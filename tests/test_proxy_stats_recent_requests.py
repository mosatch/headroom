from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from headroom.proxy.models import ProxyConfig
from headroom.proxy.server import create_app
from headroom.rollout import resolve_rollout


class FakeRequestLogger:
    def __init__(self) -> None:
        self._logs: list[dict[str, object]] = []

    @property
    def logs(self) -> list[dict[str, object]]:
        return self._logs

    @logs.setter
    def logs(self, value: list[dict[str, object]]) -> None:
        self._logs = value

    def get_recent(self, limit: int) -> list[dict[str, object]]:
        return self._logs[-limit:]


class FakeLogEntry(dict[str, object]):
    def __getattr__(self, name: str) -> object:
        return self.get(name)


def test_stats_exposes_actual_running_rollout_snapshot() -> None:
    rollout = resolve_rollout(
        {
            "HEADROOM_ROLLOUT_CHANNEL": "canary",
            "HEADROOM_FEATURES": "tool_result_interceptors",
        }
    )
    app = create_app(
        ProxyConfig(
            rollout=rollout,
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
            ccr_inject_tool=False,
            ccr_handle_responses=False,
            ccr_context_tracking=False,
            http2=False,
        )
    )

    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 12345)) as client:
        payload = client.get("/stats").json()["rollout"]

    assert payload == rollout.to_dict()
    assert payload["qualification_eligible"] is True


def test_stats_refreshes_recent_requests_when_cached() -> None:
    app = create_app(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
            ccr_inject_tool=False,
            ccr_handle_responses=False,
            ccr_context_tracking=False,
            http2=False,
        )
    )
    logger = FakeRequestLogger()
    app.state.proxy.logger = logger

    first_log = FakeLogEntry(
        {
            "timestamp": "2026-06-11T10:00:00Z",
            "provider": "openai",
            "model": "gpt-4.1",
            "input_tokens_original": 100,
            "input_tokens_optimized": 60,
            "tokens_saved": 40,
            "savings_percent": 40.0,
            "savings_breakdown": [
                {
                    "source": "tool_search",
                    "tokens": 40,
                    "usd": 0.0,
                    "realized": True,
                }
            ],
        }
    )
    second_log = FakeLogEntry(
        {
            "timestamp": "2026-06-11T10:01:00Z",
            "provider": "anthropic",
            "model": "claude-sonnet",
            "input_tokens_original": 200,
            "input_tokens_optimized": 120,
            "tokens_saved": 80,
            "savings_percent": 40.0,
        }
    )

    # Loopback client/Host: recent_requests is served only to loopback callers.
    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 12345)) as client:
        logger.logs = [first_log]
        first_response = client.get("/stats?cached=1")
        assert first_response.status_code == 200
        assert first_response.json()["recent_requests"][-1]["model"] == "gpt-4.1"
        assert first_response.json()["recent_requests"][-1]["savings_breakdown"] == [
            {
                "source": "tool_search",
                "tokens": 40,
                "usd": 0.0,
                "realized": True,
            }
        ]

        logger.logs = [first_log, second_log]
        second_response = client.get("/stats?cached=1")
        assert second_response.status_code == 200
        second_payload = second_response.json()

    assert second_payload["recent_requests"][0]["model"] == "claude-sonnet"
    assert second_payload["request_logs"][-1]["model"] == "claude-sonnet"


def test_stats_recent_requests_includes_token_incomplete_requests() -> None:
    app = create_app(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
            ccr_inject_tool=False,
            ccr_handle_responses=False,
            ccr_context_tracking=False,
            http2=False,
        )
    )
    logger = FakeRequestLogger()
    app.state.proxy.logger = logger

    logger.logs = [
        FakeLogEntry(
            {
                "request_id": "req-haiku-1",
                "timestamp": "2026-07-09T10:00:00Z",
                "provider": "anthropic",
                "model": "claude-haiku",
                "transforms_applied": [],
            }
        ),
        FakeLogEntry(
            {
                "request_id": "req-haiku-2",
                "timestamp": "2026-07-09T10:01:00Z",
                "provider": "anthropic",
                "model": "claude-haiku",
                "input_tokens_original": None,
                "input_tokens_optimized": None,
                "output_tokens": None,
                "tokens_saved": 0,
                "savings_percent": 0.0,
                "transforms_applied": [],
            }
        ),
        FakeLogEntry(
            {
                "request_id": "req-sonnet-1",
                "timestamp": "2026-07-09T10:02:00Z",
                "provider": "anthropic",
                "model": "claude-sonnet",
                "input_tokens_original": 200,
                "input_tokens_optimized": 120,
                "output_tokens": 40,
                "tokens_saved": 80,
                "savings_percent": 40.0,
                "transforms_applied": ["smart_crusher"],
            }
        ),
    ]

    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 12345)) as client:
        response = client.get("/stats")

    assert response.status_code == 200
    payload = response.json()
    assert [req["model"] for req in payload["recent_requests"]] == [
        "claude-sonnet",
        "claude-haiku",
        "claude-haiku",
    ]
    assert payload["recent_requests"][0]["token_accounting_status"] == "complete"
    assert payload["recent_requests"][0]["has_exact_tokens"] is True
    assert payload["recent_requests"][1]["output_tokens"] is None
    assert payload["recent_requests"][1]["token_accounting_status"] == "partial"
    assert payload["recent_requests"][1]["tokens_saved"] == 0
    assert payload["recent_requests"][2]["input_tokens_optimized"] is None
    assert payload["recent_requests"][2]["token_accounting_status"] == "missing"
    assert payload["recent_requests"][2]["has_exact_tokens"] is False
    assert payload["summary"]["uncompressed_requests"]["unknown_token_accounting"] == 2
    assert payload["request_logs"][-1]["model"] == "claude-sonnet"


def test_agent_usage_totals_use_proxy_only_savings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_REQUIRE_RUST_CORE", "false")
    app = create_app(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
            ccr_inject_tool=False,
            ccr_handle_responses=False,
            ccr_context_tracking=False,
            http2=False,
        )
    )
    logger = FakeRequestLogger()
    app.state.proxy.logger = logger

    logger.logs = [
        FakeLogEntry(
            {
                "timestamp": "2026-06-11T10:00:00Z",
                "provider": "openai",
                "model": "gpt-5.2-codex",
                "tags": {"client": "codex"},
                "input_tokens_original": 1000,
                "input_tokens_optimized": 900,
                "output_tokens": 50,
                "tokens_saved": 100,
                "savings_percent": 10.0,
            }
        )
    ]

    with TestClient(app) as client:
        proxy = client.app.state.proxy
        proxy.metrics.tokens_input_total = 900
        proxy.metrics.tokens_saved_total = 100
        proxy.metrics.tokens_output_total = 50

        response = client.get("/stats")

    assert response.status_code == 200
    payload = response.json()

    assert payload["tokens"]["saved"] == 100
    assert payload["agent_usage"]["totals"]["before_tokens"] == 1000
    assert payload["agent_usage"]["totals"]["tokens_saved"] == 100
    assert payload["agent_usage"]["totals"]["savings_percent"] == 10.0
    assert payload["agent_usage"]["agents"][0]["share_of_saved_percent"] == 100.0


def test_stats_preserves_default_smart_crusher_compaction_state() -> None:
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    # Loopback client/Host: the `config` block is served only to loopback callers.
    client = TestClient(
        create_app(config), base_url="http://127.0.0.1", client=("127.0.0.1", 12345)
    )

    response = client.get("/stats")

    assert response.status_code == 200
    assert response.json()["config"]["smart_crusher_with_compaction"] is None


def _read_maturation_app(mode: str, quiesce_turns: int = 5):
    return create_app(
        ProxyConfig(
            mode=mode,
            read_maturation=True,
            read_maturation_quiesce_turns=quiesce_turns,
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
            ccr_inject_tool=False,
            ccr_handle_responses=False,
            ccr_context_tracking=False,
            http2=False,
        )
    )


@pytest.mark.parametrize(
    ("mode", "quiesce_turns", "effective", "startup_level"),
    [
        ("cache", 5, False, "WARNING"),
        ("token", 5, True, "INFO"),
        # A zero threshold matures a Read in the still-mutable latest turn.
        ("cache", 0, True, "INFO"),
    ],
)
def test_read_maturation_reports_whether_it_can_fire(
    mode, quiesce_turns, effective, startup_level, caplog
):
    """Cache mode freezes a Read before it can quiesce, so enabled maturation
    never fires there; startup and /stats must say so instead of reporting a
    silently inert feature as on."""
    app = _read_maturation_app(mode, quiesce_turns)

    with caplog.at_level("INFO", logger="headroom.proxy"):
        with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 12345)) as client:
            payload = client.get("/stats").json()["read_maturation"]

    assert payload["enabled"] is True
    assert payload["effective"] is effective
    assert (payload["inert_reason"] is None) is effective
    banner = [r for r in caplog.records if r.getMessage().startswith("Read maturation:")]
    assert [r.levelname for r in banner] == [startup_level]


def _stats_app(mode: str):
    return create_app(
        ProxyConfig(
            mode=mode,
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
            ccr_inject_tool=False,
            ccr_handle_responses=False,
            ccr_context_tracking=False,
            http2=False,
        )
    )


def _unsaved(request_id: str, **extra: object) -> FakeLogEntry:
    entry: dict[str, object] = {
        "request_id": request_id,
        "timestamp": "2026-11-07T10:00:00Z",
        "provider": "anthropic",
        "model": "claude-opus",
        "input_tokens_original": 50_000,
        "input_tokens_optimized": 50_000,
        "output_tokens": 10,
        "tokens_saved": 0,
        "savings_percent": 0.0,
        "transforms_applied": [],
        "tags": {},
    }
    entry.update(extra)
    return FakeLogEntry(entry)


def test_uncompressed_reasons_follow_handler_scope_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cache mode: an empty transforms list is not evidence of a frozen history.

    The handler tags what the pipeline could touch. A skipped turn is a
    passthrough with its reason kept, a turn whose new tail had nothing to
    fold is ``no_compressible_content``, and only a turn with no un-frozen
    tail at all is ``prefix_frozen``. Untagged legacy entries keep the old
    heuristic.
    """
    monkeypatch.setenv("HEADROOM_REQUIRE_RUST_CORE", "false")
    app = _stats_app("cache")
    logger = FakeRequestLogger()
    app.state.proxy.logger = logger
    logger.logs = [
        _unsaved("skip", tags={"passthrough_reason": "cache_mode_prefix_mismatch"}),
        _unsaved("delta-empty", tags={"compression_scope": "delta"}),
        _unsaved(
            "delta-noop",
            tags={"compression_scope": "delta"},
            transforms_applied=["anthropic:tool_schema_compaction"],
        ),
        _unsaved("full-small", tags={"compression_scope": "full"}, input_tokens_original=120),
        _unsaved("frozen", tags={"compression_scope": "frozen"}),
        _unsaved("legacy"),
    ]

    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 12345)) as client:
        summary = client.get("/stats").json()["summary"]

    assert summary["uncompressed_requests"] == {
        "passthrough": 1,
        "no_compressible_content": 2,
        "too_small": 1,
        "prefix_frozen": 2,
    }
    assert summary["passthrough_reasons"] == {"cache_mode_prefix_mismatch": 1}
    # Two genuinely frozen turns do not earn the "switch to token mode" tip.
    assert "tip" not in summary


def test_token_mode_tip_needs_frozen_turns_not_empty_deltas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HEADROOM_REQUIRE_RUST_CORE", "false")
    app = _stats_app("cache")
    logger = FakeRequestLogger()
    app.state.proxy.logger = logger
    logger.logs = [_unsaved(f"delta-{i}", tags={"compression_scope": "delta"}) for i in range(12)]
    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 12345)) as client:
        summary = client.get("/stats").json()["summary"]
    assert summary["uncompressed_requests"] == {"no_compressible_content": 12}
    assert "tip" not in summary

    logger.logs = [_unsaved(f"frozen-{i}", tags={"compression_scope": "frozen"}) for i in range(12)]
    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 12345)) as client:
        summary = client.get("/stats").json()["summary"]
    assert summary["uncompressed_requests"] == {"prefix_frozen": 12}
    assert "HEADROOM_MODE=token" in summary["tip"]


def test_stats_report_who_deferred_tool_schemas(monkeypatch: pytest.MonkeyPatch) -> None:
    """A client that defers its own tools must not read as deferral being off.

    ``tool_schema_tokens_saved`` stays 0 by design when the client already
    sent ``defer_loading``; the summary now says so beside it.
    """
    import asyncio

    monkeypatch.setenv("HEADROOM_REQUIRE_RUST_CORE", "false")
    app = _stats_app("cache")
    metrics = app.state.proxy.metrics
    asyncio.run(metrics.record_tool_search_mode("client", 7_000))
    asyncio.run(metrics.record_tool_search_mode("client", 7_400))
    asyncio.run(metrics.record_tool_search_mode("headroom", 0))
    asyncio.run(metrics.record_tool_search_mode("none"))

    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 12345)) as client:
        compression = client.get("/stats").json()["summary"]["compression"]

    assert compression["tool_schema_tokens_saved"] == 0
    assert compression["tool_search_modes"] == {"client": 2, "headroom": 1, "none": 1}
    assert compression["tool_schema_tokens_deferred_by_client"] == 14_400
    # Reported beside the savings, never folded into them.
    assert compression["total_tokens_saved_all_layers"] == 0


@pytest.mark.parametrize("path", ["/stats?cached=1", "/stats"])
def test_recent_requests_served_to_token_holder_behind_auth_proxy(
    monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    """A dashboard fronted by an authenticating reverse proxy reaches /stats with
    a domain Host from a non-loopback peer, carrying HEADROOM_PROXY_TOKEN. The
    Host/CIDR rule can never admit it; the verified token must."""
    monkeypatch.setenv("HEADROOM_PROXY_TOKEN", "s3cret-token")
    monkeypatch.setenv("HEADROOM_HOST", "0.0.0.0")
    app = create_app(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
            ccr_inject_tool=False,
            ccr_handle_responses=False,
            ccr_context_tracking=False,
            http2=False,
        )
    )
    logger = FakeRequestLogger()
    app.state.proxy.logger = logger
    logger.logs = [
        FakeLogEntry(
            {
                "request_id": "req-1",
                "timestamp": "2026-10-12T10:00:00Z",
                "provider": "anthropic",
                "model": "claude-sonnet",
            }
        )
    ]
    token = {"x-headroom-proxy-token": "s3cret-token"}

    with TestClient(app, base_url="http://headroom", client=("10.0.4.7", 5000)) as client:
        payload = client.get(path, headers=token).json()
        assert [r["request_id"] for r in payload["recent_requests"]] == ["req-1"]
        assert payload["request_logs"]

        # The gate still refuses callers without (or with a wrong) token.
        assert client.get(path).status_code == 401
        assert client.get(path, headers={"x-headroom-proxy-token": "nope"}).status_code == 401
