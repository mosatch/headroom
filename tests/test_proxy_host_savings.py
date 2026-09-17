"""Tests for per-host savings attribution (X-Headroom-Host).

The host axis exists for one deployment the project axis cannot describe: a
proxy bound to a LAN address, serving several machines. These tests cover the
funnel, the stats payloads, the cardinality cap, and — the reason the axis keys
on a header rather than the peer address — that one machine arriving from
several addresses stays one host.
"""

import asyncio

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.host_context import (  # noqa: E402
    get_current_host,
    set_current_host,
)
from headroom.proxy.outcome import RequestOutcome, emit_request_outcome  # noqa: E402
from headroom.proxy.savings_tracker import (  # noqa: E402
    DEFAULT_MAX_HOSTS,
    SavingsTracker,
)
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402


def test_host_contextvar_roundtrip():
    set_current_host("MBMini.local")
    try:
        # Sanitized on the way in, so every spelling of one machine keys alike.
        assert get_current_host() == "mbmini"
    finally:
        set_current_host(None)
    assert get_current_host() is None


def test_tracker_accumulates_per_host_and_persists(tmp_path):
    path = tmp_path / "savings.json"
    tracker = SavingsTracker(path=path)

    tracker.record_request(model="gpt-4o", input_tokens=600, tokens_saved=400, host="mbmini")
    tracker.record_request(model="gpt-4o", input_tokens=300, tokens_saved=100, host="mbmini")
    tracker.record_request(model="gpt-4o", input_tokens=200, tokens_saved=50, host="build01")
    tracker.flush()

    hosts = SavingsTracker(path=path).snapshot()["hosts"]
    assert hosts["mbmini"]["requests"] == 2
    assert hosts["mbmini"]["tokens_saved"] == 500
    assert hosts["build01"]["tokens_saved"] == 50
    # Ranked by savings, like projects.
    assert list(hosts) == ["mbmini", "build01"]


def test_one_machine_on_several_addresses_stays_one_host(tmp_path):
    """The bug that motivates the header: loopback + LAN IP are one machine."""
    tracker = SavingsTracker(path=tmp_path / "savings.json")

    for _ in range(3):
        tracker.record_request(model="gpt-4o", input_tokens=100, tokens_saved=10, host="MBMini")
    # Same machine, reported with the Bonjour spelling and a trailing root dot.
    tracker.record_request(model="gpt-4o", input_tokens=100, tokens_saved=10, host="mbmini.local.")

    hosts = tracker.snapshot()["hosts"]
    assert list(hosts) == ["mbmini"]
    assert hosts["mbmini"]["requests"] == 4


def test_tracker_migrates_state_without_hosts(tmp_path):
    """State written before this feature loads without a hosts key."""
    path = tmp_path / "savings.json"
    tracker = SavingsTracker(path=path)
    tracker.record_request(model="gpt-4o", input_tokens=100, tokens_saved=10)
    tracker.flush()

    import json

    raw = json.loads(path.read_text())
    raw.pop("hosts", None)
    path.write_text(json.dumps(raw))

    reloaded = SavingsTracker(path=path)
    assert reloaded.snapshot()["hosts"] == {}


def test_tracker_caps_host_cardinality(tmp_path):
    tracker = SavingsTracker(path=tmp_path / "savings.json")
    for i in range(DEFAULT_MAX_HOSTS + 10):
        tracker.record_request(
            model="gpt-4o",
            input_tokens=10,
            # Later hosts save more, so the cap keeps them and evicts the small.
            tokens_saved=i + 1,
            host=f"host{i:03d}",
        )
    hosts = tracker.snapshot()["hosts"]
    assert len(hosts) <= DEFAULT_MAX_HOSTS


def _emit_outcome(proxy, *, host_field=None):
    outcome = RequestOutcome(
        request_id="req-1",
        provider="openai",
        model="gpt-4o",
        original_tokens=1000,
        optimized_tokens=600,
        output_tokens=20,
        tokens_saved=400,
        attempted_input_tokens=1000,
        host=host_field,
    )
    asyncio.run(emit_request_outcome(proxy, outcome))


def test_funnel_attributes_savings_from_context_and_stats_exposes_them(tmp_path, monkeypatch):
    monkeypatch.setenv("HEADROOM_SAVINGS_PATH", str(tmp_path / "savings.json"))
    config = ProxyConfig(cache_enabled=False, rate_limit_enabled=False, log_requests=False)

    with TestClient(create_app(config)) as client:
        proxy = client.app.state.proxy

        set_current_host("ctx-host")
        try:
            _emit_outcome(proxy)
        finally:
            set_current_host(None)

        # Explicit outcome.host wins over the bound context.
        _emit_outcome(proxy, host_field="field-host")

        stats = client.get("/stats").json()
        per_host = stats["savings"]["per_host"]
        assert per_host["ctx-host"]["tokens_saved"] == 400
        assert per_host["field-host"]["tokens_saved"] == 400
        assert stats["persistent_savings"]["hosts"] == per_host
        assert stats["persistent_savings"]["hosts_limit"] == DEFAULT_MAX_HOSTS

        history = client.get("/stats-history").json()
        assert history["hosts"]["ctx-host"]["requests"] == 1


def test_record_request_without_host_matches_legacy_totals(tmp_path):
    """Unattributed traffic is skipped, leaving aggregates untouched."""
    tracker = SavingsTracker(path=tmp_path / "savings.json")
    tracker.record_request(model="gpt-4o", input_tokens=600, tokens_saved=400)

    snapshot = tracker.snapshot()
    assert snapshot["hosts"] == {}
    assert snapshot["lifetime"]["tokens_saved"] == 400
    assert snapshot["lifetime"]["requests"] == 1


def test_middleware_binds_host_header_to_context(tmp_path, monkeypatch):
    monkeypatch.setenv("HEADROOM_SAVINGS_PATH", str(tmp_path / "savings.json"))
    config = ProxyConfig(cache_enabled=False, rate_limit_enabled=False, log_requests=False)

    captured: list[str | None] = []

    import headroom.proxy.server as server_module

    real_bind = server_module.bind_host_from_request

    def _capture(headers, peer_address):
        bound = real_bind(headers, peer_address)
        captured.append(bound)
        return bound

    monkeypatch.setattr(server_module, "bind_host_from_request", _capture)

    with TestClient(create_app(config)) as client:
        assert (
            client.get("/health", headers={"X-Headroom-Host": " Build01.local "}).status_code == 200
        )
        # An explicit header wins over the peer address.
        assert captured[-1] == "build01"

        # No header: falls back to the peer address rather than going
        # unattributed, so a client that cannot send headers still lands in a
        # bucket. TestClient reports its peer as "testclient".
        assert client.get("/health").status_code == 200
        assert captured[-1] is not None
        assert captured[-1] != "build01"
