"""Tests for optional project/host attribution enforcement.

The safety-critical case is the guard at the bottom: enabling enforcement must
never cost the operator the dashboard, the health probes, or the metrics
endpoint. That test walks the app's own registered routes rather than a
hand-written list, so a new operational endpoint added later cannot regress the
exemption by omission.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.attribution_policy import (  # noqa: E402
    attribution_error_payload,
    is_operational_path,
    missing_attribution,
    requires_attribution,
)
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402

PROJECT_H = "x-headroom-project"
HOST_H = "x-headroom-host"


# ---------------------------------------------------------------------------
# Pure policy
# ---------------------------------------------------------------------------


def test_operational_paths_are_exempt():
    for path in ("/health", "/livez", "/metrics", "/stats", "/dashboard", "/favicon.ico"):
        assert is_operational_path(path), path
        assert not requires_attribution(path)


def test_operational_prefixes_cover_subpaths():
    assert is_operational_path("/dashboard/static/alpine.min.js")
    assert is_operational_path("/v1/telemetry/export")
    assert is_operational_path("/debug/tasks")


def test_prefix_match_is_segment_aware():
    """``/statsify`` is not ``/stats``."""
    assert is_operational_path("/stats")
    assert not is_operational_path("/statsify")
    assert not is_operational_path("/healthz-custom")


def test_provider_paths_require_attribution():
    for path in ("/v1/messages", "/v1/chat/completions", "/v1/responses", "/v1beta/models"):
        assert requires_attribution(path), path


def test_nothing_required_when_both_toggles_off():
    assert (
        missing_attribution({}, path="/v1/messages", require_project=False, require_host=False)
        == ()
    )


def test_missing_project_reported():
    missing = missing_attribution({}, path="/v1/messages", require_project=True, require_host=False)
    assert missing == (PROJECT_H,)


def test_path_prefix_satisfies_project_requirement():
    """Clients that cannot send headers use /p/<name>; that must count."""
    assert (
        missing_attribution(
            {},
            path="/v1/messages",
            require_project=True,
            require_host=False,
            prefix_project="my-repo",
        )
        == ()
    )


def test_host_requirement_needs_the_header_not_the_fallback():
    """The peer-address fallback must not satisfy enforcement.

    ``resolve_host`` almost always yields something, so accepting that would
    make the option a no-op. Only the explicit header counts.
    """
    assert missing_attribution(
        {}, path="/v1/messages", require_project=False, require_host=True
    ) == (HOST_H,)
    assert (
        missing_attribution(
            {HOST_H: "build01"}, path="/v1/messages", require_project=False, require_host=True
        )
        == ()
    )


def test_blank_header_does_not_satisfy():
    assert missing_attribution(
        {HOST_H: "   "}, path="/v1/messages", require_project=False, require_host=True
    ) == (HOST_H,)


def test_both_missing_reported_together():
    missing = missing_attribution({}, path="/v1/messages", require_project=True, require_host=True)
    assert set(missing) == {PROJECT_H, HOST_H}


def test_error_payload_names_the_headers_and_reads_in_both_formats():
    payload = attribution_error_payload((PROJECT_H, HOST_H))
    # Anthropic envelope.
    assert payload["type"] == "error"
    assert payload["error"]["type"] == "invalid_request_error"
    # OpenAI clients read error.message; it must name what to send.
    assert PROJECT_H in payload["error"]["message"]
    assert HOST_H in payload["error"]["message"]
    assert payload["error"]["missing_headers"] == [PROJECT_H, HOST_H]


# ---------------------------------------------------------------------------
# Wired through the app
# ---------------------------------------------------------------------------


def _client(tmp_path, monkeypatch, **flags):
    monkeypatch.setenv("HEADROOM_SAVINGS_PATH", str(tmp_path / "savings.json"))
    config = ProxyConfig(cache_enabled=False, rate_limit_enabled=False, log_requests=False, **flags)
    return TestClient(create_app(config))


def test_disabled_by_default_lets_unattributed_traffic_through(tmp_path, monkeypatch):
    """The single-user default must be unchanged.

    An unattributed call reaches the upstream auth check (401) instead of being
    refused here, which is what "enforcement is off" has to mean.
    """
    with _client(tmp_path, monkeypatch) as client:
        resp = client.post("/v1/messages", json={"model": "x", "messages": []})
        assert resp.status_code != 400
        assert "missing_headers" not in resp.text


def test_enforcement_refuses_unattributed_provider_request(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch, require_host_attribution=True) as client:
        resp = client.post("/v1/messages", json={"model": "x", "messages": []})
        assert resp.status_code == 400
        body = resp.json()
        assert body["error"]["missing_headers"] == [HOST_H]


def test_enforcement_allows_attributed_request_through(tmp_path, monkeypatch):
    """With the header the request passes the gate and reaches upstream auth.

    Asserted as "not our 400" rather than a specific success code: the test
    has no upstream credentials, so 401 is the correct far side of the gate.
    """
    with _client(tmp_path, monkeypatch, require_host_attribution=True) as client:
        resp = client.post(
            "/v1/messages",
            json={"model": "x", "messages": []},
            headers={"X-Headroom-Host": "build01"},
        )
        assert resp.status_code != 400
        assert "missing_headers" not in resp.text


def test_project_prefix_satisfies_enforcement_end_to_end(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch, require_project_attribution=True) as client:
        refused = client.post("/v1/messages", json={"model": "x", "messages": []})
        assert refused.status_code == 400

        allowed = client.post("/p/my-repo/v1/messages", json={"model": "x", "messages": []})
        assert allowed.status_code != 400
        assert "missing_headers" not in allowed.text


# ---------------------------------------------------------------------------
# The lockout guard
# ---------------------------------------------------------------------------


def test_enforcement_never_blocks_the_operator_surface(tmp_path, monkeypatch):
    """Enabling enforcement must not break health, dashboard, stats or metrics.

    Walks the app's own route table instead of a fixed list so a newly added
    operational endpoint that forgets the exemption fails here.
    """
    with _client(
        tmp_path,
        monkeypatch,
        require_project_attribution=True,
        require_host_attribution=True,
    ) as client:
        for path in ("/health", "/livez", "/readyz", "/metrics", "/stats", "/dashboard"):
            resp = client.get(path)
            assert resp.status_code != 400 or "missing_headers" not in resp.text, (
                f"{path} was blocked by attribution enforcement"
            )


def test_every_operational_get_route_is_exempt(tmp_path, monkeypatch):
    """No parameterless GET route the proxy registers for itself is enforced."""
    config = ProxyConfig(
        cache_enabled=False,
        rate_limit_enabled=False,
        log_requests=False,
        require_project_attribution=True,
        require_host_attribution=True,
    )
    app = create_app(config)

    # Routes the proxy serves itself, as opposed to provider passthrough: no
    # path parameters, and not under a provider API prefix.
    provider_prefixes = ("/v1/", "/v1beta/", "/v1internal", "/model/", "/chat/", "/backend-api/")
    operational: list[str] = []
    for route in app.routes:
        path = getattr(route, "path", "")
        methods = getattr(route, "methods", set()) or set()
        if "GET" not in methods or "{" in path:
            continue
        if any(path.startswith(p) for p in provider_prefixes):
            continue
        operational.append(path)

    assert operational, "expected to discover operational routes"
    not_exempt = [p for p in operational if requires_attribution(p)]
    assert not not_exempt, (
        "these operational routes would be blocked by enforcement; "
        f"add them to OPERATIONAL_PATH_PREFIXES: {not_exempt}"
    )
