"""The attribution report extension: cache economics, top sessions, git labels."""

from __future__ import annotations

import pytest

from headroom.pipeline import OutcomeSnapshot, PipelineEvent, PipelineStage
from headroom.proxy import attribution_report as ar
from headroom.proxy.attribution import (
    AttributionRequest,
    resolve_attribution,
    set_current_attribution,
    unregister_dimension,
)


@pytest.fixture(autouse=True)
def _free_pricing(monkeypatch):
    # Deterministic cost: $1 per input token, $0 output; no price table needed.
    monkeypatch.setattr(
        ar, "_estimate_costs", lambda snap: (float(snap.input_tokens), 0.0), raising=True
    )


def _snap(conv="c1", *, read=0, write=0, inp=1000, labels=None, saved=0, model="m"):
    return OutcomeSnapshot(
        request_id="r",
        provider="anthropic",
        model=model,
        input_tokens=inp,
        cache_read_tokens=read,
        cache_write_tokens=write,
        novel_tokens_saved=saved,
        conversation_key=conv,
        attribution=labels or {},
    )


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        self.t += 1.0
        return self.t


def test_bust_detection_counts_a_lost_cache_only():
    report = ar.AttributionReport(bust_min_write_tokens=100, clock=_Clock())
    labels = {"host": "a"}
    report.observe(_snap(write=5000, labels=labels))  # cold start: not a bust
    report.observe(_snap(read=5000, write=50, labels=labels))  # reading
    report.observe(_snap(read=5000, write=500, labels=labels))  # growth while reading
    report.observe(_snap(read=0, write=5000, labels=labels))  # lost it: bust
    report.observe(_snap(read=0, write=5000, labels=labels))  # still cold: not a 2nd
    report.observe(_snap(read=5000, labels=labels))
    out = report.report()
    assert out["totals"]["busts"] == 1
    row = out["cache_by_dimension"]["host"]["a"]
    assert row["busts"] == 1
    assert row["bust_write_tokens"] == 5000
    assert row["requests"] == 6
    session = out["top_sessions"][0]
    assert session["busts"] == 1
    assert session["requests"] == 6


def test_top_sessions_ranked_by_cost_and_bounded():
    report = ar.AttributionReport(max_sessions=3, clock=_Clock())
    for i, cost in enumerate([10, 500, 30, 400]):
        report.observe(_snap(conv=f"c{i}", inp=cost, labels={"host": f"h{i}"}))
    out = report.report(top=2)
    assert [s["conversation"] for s in out["top_sessions"]] == ["c1", "c3"]
    assert out["sessions_tracked"] == 3
    assert out["top_sessions"][0]["labels"] == {"host": "h1"}


def test_requests_without_conversation_still_count_per_label():
    report = ar.AttributionReport(clock=_Clock())
    report.observe(_snap(conv=None, labels={"project": "p"}))
    out = report.report()
    assert out["sessions_tracked"] == 0
    assert out["cache_by_dimension"]["project"]["p"]["requests"] == 1


def test_labels_per_dimension_bounded():
    report = ar.AttributionReport(max_labels=2, clock=_Clock())
    for i, cost in enumerate([5, 50, 500]):
        report.observe(_snap(conv=None, inp=cost, labels={"host": f"h{i}"}))
    assert set(report.report()["cache_by_dimension"]["host"]) == {"h1", "h2"}


def test_pipeline_event_entry_point():
    report = ar.AttributionReport(clock=_Clock())
    event = PipelineEvent(
        stage=PipelineStage.OUTCOME_OBSERVED, operation="proxy.outcome", outcome=_snap()
    )
    assert report.on_pipeline_event(event) is None
    report.on_pipeline_event(PipelineEvent(stage=PipelineStage.PRE_SEND, operation="x"))
    assert report.report()["totals"]["requests"] == 1


def test_insights_flag_a_label_that_busts_far_more_than_the_fleet():
    totals = {"requests": 100, "cost_usd": 100.0, "bust_rate": 0.02}
    cache = {
        "host": {
            "noisy": {
                "requests": 40,
                "cost_usd": 60.0,
                "busts": 8,
                "bust_rate": 0.2,
                "bust_write_tokens": 400_000,
                "cache_read_share": 0.3,
                "cache_write_tokens": 500_000,
            },
            "quiet": {"requests": 60, "cost_usd": 40.0, "busts": 0, "bust_rate": 0.0},
        }
    }
    found = ar.insights(totals, cache)
    assert any("host=noisy loses the prompt cache on 20%" in f for f in found)
    assert any("host=noisy writes more cache than it reads" in f for f in found)
    assert not any("quiet" in f for f in found)


def test_insights_need_enough_traffic():
    assert ar.insights({"requests": 5, "cost_usd": 1.0}, {"host": {}}) == []


# ── git labels ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("git@github.com:headroomlabs-ai/headroom.git", "headroomlabs-ai/headroom"),
        ("https://github.com/mosatch/headroom.git", "mosatch/headroom"),
        ("https://user:ghp_SECRET@github.com/o/r.git", "o/r"),
        ("ssh://git@host:2222/team/proj", "team/proj"),
        ("/srv/git/solo.git", "git/solo"),
        ("", None),
    ],
)
def test_repo_from_remote_url_drops_credentials(url, expected):
    result = ar.repo_from_remote_url(url)
    assert result == expected
    assert "SECRET" not in (result or "")


def _make_repo(root, *, branch="main", remote="git@github.com:o/r.git"):
    git = root / ".git"
    git.mkdir(parents=True)
    (git / "HEAD").write_text(f"ref: refs/heads/{branch}\n")
    (git / "config").write_text(f'[core]\n\tbare = false\n[remote "origin"]\n\turl = {remote}\n')
    return git


def test_git_labels_read_branch_repo_and_refresh(tmp_path):
    git = _make_repo(tmp_path / "proj", branch="feat/x")
    clock = [0.0]
    labels = ar.GitLabels(ttl_s=5.0, clock=lambda: clock[0])
    assert labels.lookup(str(tmp_path / "proj" / "sub")) == ("o/r", "feat/x")
    (git / "HEAD").write_text("ref: refs/heads/main\n")
    assert labels.lookup(str(tmp_path / "proj" / "sub"))[1] == "feat/x"  # cached
    clock[0] = 6.0
    assert labels.lookup(str(tmp_path / "proj" / "sub"))[1] == "main"
    (git / "HEAD").write_text("0123456789abcdef0123\n")
    clock[0] = 12.0
    assert labels.lookup(str(tmp_path / "proj" / "sub"))[1] == "0123456789ab"
    assert labels.lookup(None) == (None, None)


def test_git_labels_follow_a_worktree_to_its_common_dir(tmp_path):
    main_git = _make_repo(tmp_path / "main", remote="https://x/acme/app.git")
    wt_git = main_git / "worktrees" / "wt"
    wt_git.mkdir(parents=True)
    (wt_git / "HEAD").write_text("ref: refs/heads/session/abc\n")
    (wt_git / "commondir").write_text("../..\n")
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / ".git").write_text(f"gitdir: {wt_git}\n")
    assert ar.GitLabels().lookup(str(wt)) == ("acme/app", "session/abc")


def test_git_labels_without_repo(tmp_path):
    assert ar.GitLabels().lookup(str(tmp_path)) == (None, None)


def test_branch_dimension_prefers_header_then_registered_workspace(tmp_path, monkeypatch):
    from headroom.proxy import project_context

    _make_repo(tmp_path / "ws", branch="dev")
    monkeypatch.setattr(ar, "_git_labels", ar.GitLabels())
    from headroom.proxy.attribution import register_dimension

    register_dimension("repo", ar.resolve_repo, replace=True)
    register_dimension("branch", ar.resolve_branch, replace=True)
    try:
        project_context.set_registered_cwd(str(tmp_path / "ws"))
        req = AttributionRequest(headers={})
        assert resolve_attribution(req, ["repo", "branch"]) == {"repo": "o/r", "branch": "dev"}
        req = AttributionRequest(headers={"x-headroom-branch": "pr-12"})
        assert resolve_attribution(req, ["branch"]) == {"branch": "pr-12"}
    finally:
        project_context.set_registered_cwd(None)
        unregister_dimension("repo")
        unregister_dimension("branch")


# ── installed in the proxy ────────────────────────────────────────────


def test_install_serves_report_to_operators_only(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    import asyncio

    from fastapi.testclient import TestClient

    from headroom.proxy.outcome import RequestOutcome, emit_request_outcome
    from headroom.proxy.server import ProxyConfig, create_app

    monkeypatch.setenv("HEADROOM_SAVINGS_PATH", str(tmp_path / "savings.json"))
    monkeypatch.setenv("HEADROOM_SAVINGS_EVENTS_PATH", str(tmp_path / "events.jsonl"))
    monkeypatch.setenv("HEADROOM_ATTRIBUTION_GIT", "0")
    monkeypatch.setattr(
        "headroom.proxy.extensions.discover",
        lambda: iter([("attribution", ar.install)]),
    )
    config = ProxyConfig(
        cache_enabled=False,
        rate_limit_enabled=False,
        log_requests=False,
        proxy_extensions=["attribution"],
    )
    app = create_app(config)
    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 1)) as client:
        set_current_attribution({"host": "mbmini"})
        asyncio.run(
            emit_request_outcome(
                client.app.state.proxy,
                RequestOutcome(
                    request_id="r1",
                    provider="openai",
                    model="gpt-4o",
                    original_tokens=1000,
                    optimized_tokens=600,
                    output_tokens=20,
                    tokens_saved=400,
                    attempted_input_tokens=1000,
                    conversation_key="conv-9",
                ),
            )
        )
        set_current_attribution(None)
        body = client.get("/attribution/report").json()
        assert body["top_sessions"][0]["conversation"] == "conv-9"
        assert body["top_sessions"][0]["labels"] == {"host": "mbmini"}
        assert "mbmini" in body["cache_by_dimension"]["host"]
        assert body["savings_by_dimension"]["host"]["mbmini"]["tokens_saved"] == 400
    with TestClient(app, base_url="http://10.0.0.5", client=("10.0.0.9", 1)) as remote:
        assert remote.get("/attribution/report").status_code in (401, 403, 404)
