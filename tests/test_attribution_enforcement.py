"""Optional enforcement of attribution (HEADROOM_REQUIRE_DIMENSIONS).

The safety-critical case is the route walk at the bottom: enabling enforcement
must never cost the operator the dashboard, health probes or metrics. It walks
the app's own route table, so a new operational endpoint cannot regress the
exemption by omission.
"""

from __future__ import annotations

import pytest

from headroom.proxy.attribution import AttributionRequest
from headroom.proxy.attribution_policy import (
    attribution_error_payload,
    is_operational_path,
    missing_dimensions,
    required_dimensions,
    requires_attribution,
)

CLI_UA = "claude-cli/2.1.0 (external, cli)"


def _req(path="/v1/messages", headers=None, **kw):
    return AttributionRequest(
        headers={k.lower(): v for k, v in (headers or {}).items()}, path=path, **kw
    )


# ── pure policy ───────────────────────────────────────────────────────


def test_operational_paths_are_exempt():
    for path in (
        "/health",
        "/livez",
        "/metrics",
        "/stats",
        "/dashboard/static/x.js",
        "/attribution/report",
        "/v1/telemetry/export",
    ):
        assert is_operational_path(path), path
        assert not requires_attribution(path)


def test_prefix_match_is_segment_aware():
    assert not is_operational_path("/statsify")
    assert not is_operational_path("/healthz-custom")


def test_provider_paths_require_attribution():
    for path in ("/v1/messages", "/v1/chat/completions", "/v1/responses", "/v1beta/models"):
        assert requires_attribution(path), path


@pytest.mark.parametrize(
    ("configured", "env", "expected"),
    [
        (None, {}, ()),
        (None, {"HEADROOM_REQUIRE_DIMENSIONS": "host, Project,host,bad name"}, ("host", "project")),
        (["team"], {"HEADROOM_REQUIRE_DIMENSIONS": "host"}, ("team",)),
        ([], {"HEADROOM_REQUIRE_DIMENSIONS": "host"}, ()),
    ],
)
def test_required_dimensions(configured, env, expected):
    assert required_dimensions(configured, env) == expected


def test_nothing_required_means_nothing_missing():
    assert missing_dimensions(_req(), ()) == ()


def test_host_needs_the_header_not_the_peer_fallback():
    assert missing_dimensions(_req(peer="192.168.0.12"), ["host"]) == ("host",)
    assert missing_dimensions(_req(peer="127.0.0.1"), ["host"]) == ("host",)
    assert missing_dimensions(_req(headers={"X-Headroom-Host": "b1"}), ["host"]) == ()


def test_project_prefix_satisfies_project():
    assert missing_dimensions(_req(), ["project"]) == ("project",)
    assert missing_dimensions(_req(path_project="repo"), ["project"]) == ()


def test_unknown_traffic_class_does_not_count():
    assert missing_dimensions(_req(), ["traffic_class"]) == ("traffic_class",)
    assert missing_dimensions(_req(headers={"User-Agent": CLI_UA}), ["traffic_class"]) == ()


def test_custom_dimension_via_labels_header():
    assert missing_dimensions(_req(), ["team"]) == ("team",)
    assert missing_dimensions(_req(headers={"X-Headroom-Labels": "team=infra"}), ["team"]) == ()


def test_operational_path_never_missing():
    assert missing_dimensions(_req(path="/stats"), ["host", "project"]) == ()


def test_error_payload_names_what_to_send():
    body = attribution_error_payload(("host", "project"))
    assert body["error"]["missing_dimensions"] == ["host", "project"]
    message = body["error"]["message"]
    assert "X-Headroom-Host" in message and "/p/<project-name>" in message
    assert body["error"]["type"] == "invalid_request_error"


# ── through the proxy ─────────────────────────────────────────────────


def _client(tmp_path, monkeypatch, required=None):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from headroom.proxy.server import ProxyConfig, create_app

    monkeypatch.setenv("HEADROOM_SAVINGS_PATH", str(tmp_path / "savings.json"))
    monkeypatch.setenv("HEADROOM_SAVINGS_EVENTS_PATH", str(tmp_path / "events.jsonl"))
    monkeypatch.delenv("HEADROOM_REQUIRE_DIMENSIONS", raising=False)
    config = ProxyConfig(
        cache_enabled=False,
        rate_limit_enabled=False,
        log_requests=False,
        require_dimensions=required,
    )
    return TestClient(create_app(config), base_url="http://127.0.0.1", client=("127.0.0.1", 1))


def test_off_by_default(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        resp = client.post("/v1/messages", json={"model": "x", "messages": []})
        assert "missing_dimensions" not in resp.text


def test_refuses_unattributed_provider_request(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch, ["host"]) as client:
        resp = client.post("/v1/messages", json={"model": "x", "messages": []})
        assert resp.status_code == 400
        assert resp.json()["error"]["missing_dimensions"] == ["host"]


def test_attributed_request_passes_the_gate(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch, ["host", "project"]) as client:
        resp = client.post(
            "/p/my-repo/v1/messages",
            json={"model": "x", "messages": []},
            headers={"X-Headroom-Host": "build01"},
        )
        assert "missing_dimensions" not in resp.text


def test_websocket_upgrade_is_refused_with_1008(tmp_path, monkeypatch):
    from starlette.websockets import WebSocketDisconnect

    with _client(tmp_path, monkeypatch, ["host"]) as client:
        with pytest.raises(WebSocketDisconnect) as exc:
            with client.websocket_connect("/v1/responses") as ws:
                ws.receive_text()
        assert exc.value.code == 1008


def test_enforcement_never_blocks_the_operator_surface(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch, ["host", "project", "team"]) as client:
        for path in ("/health", "/livez", "/readyz", "/metrics", "/stats", "/dashboard"):
            resp = client.get(path)
            assert "missing_dimensions" not in resp.text, path


def test_every_operational_get_route_is_exempt(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch, ["host"]) as client:
        app = client.app
    provider_prefixes = ("/v1/", "/v1beta/", "/v1internal", "/model/", "/chat/", "/backend-api/")
    operational = [
        route.path
        for route in app.routes
        if "GET" in (getattr(route, "methods", set()) or set())
        and "{" not in getattr(route, "path", "{")
        and not any(route.path.startswith(p) for p in provider_prefixes)
    ]
    assert operational
    not_exempt = [p for p in operational if requires_attribution(p)]
    assert not not_exempt, f"add to OPERATIONAL_PATH_PREFIXES: {not_exempt}"
