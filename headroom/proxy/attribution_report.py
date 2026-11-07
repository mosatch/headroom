"""Attribution report: where the money goes, by label.

A proxy extension (``headroom.proxy_extension`` entry point ``attribution``,
enable with ``--proxy-extension attribution``) built only on public seams: it
reads each request's :class:`~headroom.pipeline.OutcomeSnapshot` at
``OUTCOME_OBSERVED`` and the labels :mod:`headroom.proxy.attribution` put on it.
It answers questions the per-label savings totals cannot:

**Which label pays for the prompt cache?** Cache reads are cheap and writes
carry a premium, so two clients with the same token volume can cost very
different amounts. Per label it tracks reads, writes, the read share of cached
input, and *busts*: a turn of a conversation that had been reading the cache
and suddenly re-writes it. A bust is the expensive event (the whole prefix is
billed again at the write rate), and a label that busts far more often than the
rest is usually one misbehaving client, plugin or prompt template.

**Which conversations cost the most?** A bounded table of the most expensive
conversations (``conversation_key``), each with its labels, so "what was that
$4 session on the build box" has an answer.

**What did this branch cost?** The extension registers ``repo`` and ``branch``
dimensions. A client can send ``X-Headroom-Repo`` / ``X-Headroom-Branch``;
otherwise, for a ``headroom wrap`` session on this machine, they are read from
the workspace's git checkout (the server-verified root, never a header path),
so a branch switch mid-session is picked up within seconds. With these on,
``headroom savings --by branch`` and the per-label tables split by branch.

Everything is in memory and bounded; ``GET /attribution/report`` returns it to
the same callers that may read ``/stats-history``. Costs are estimates from the
same price table the savings figures use.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

try:  # Only the route needs it; the rest of the module works without the proxy extra.
    from fastapi import HTTPException, Request
except ImportError:  # pragma: no cover - fastapi ships with the proxy extra
    HTTPException = None  # type: ignore[assignment,misc]
    Request = Any  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

#: Conversations kept for the top-sessions table (least recently active evicted).
MAX_SESSIONS: Final[int] = 1000
#: Labels kept per dimension in the cache table.
MAX_LABELS: Final[int] = 50
#: A turn that writes at least this many cache tokens after the conversation
#: had been reading counts as a bust. Below it, a write is the normal growth of
#: the cached prefix by the latest turn rather than a re-write of it.
BUST_MIN_WRITE_TOKENS: Final[int] = 4096
#: Seconds a git HEAD read is reused before the checkout is looked at again.
GIT_CACHE_TTL_S: Final[float] = 5.0

REPO_HEADER: Final[str] = "x-headroom-repo"
BRANCH_HEADER: Final[str] = "x-headroom-branch"


# ── git-derived dimensions ────────────────────────────────────────────


def _git_dir(root: Path) -> Path | None:
    """The git directory for a checkout or worktree rooted at or above ``root``."""
    for candidate in (root, *root.parents):
        dot_git = candidate / ".git"
        if dot_git.is_dir():
            return dot_git
        if dot_git.is_file():
            # Worktree / submodule: ".git" is a file naming the real gitdir.
            try:
                text = dot_git.read_text(encoding="utf-8").strip()
            except OSError:
                return None
            if text.startswith("gitdir:"):
                target = Path(text[len("gitdir:") :].strip())
                return target if target.is_absolute() else (candidate / target).resolve()
            return None
    return None


def _common_dir(git_dir: Path) -> Path:
    """Where config lives: a worktree's gitdir points back via ``commondir``."""
    try:
        rel = (git_dir / "commondir").read_text(encoding="utf-8").strip()
    except OSError:
        return git_dir
    path = Path(rel)
    return path if path.is_absolute() else (git_dir / path).resolve()


def read_branch(git_dir: Path) -> str | None:
    """Current branch name, or a short sha when HEAD is detached."""
    try:
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if head.startswith("ref:"):
        ref = head[len("ref:") :].strip()
        return ref[len("refs/heads/") :] if ref.startswith("refs/heads/") else ref
    return head[:12] or None


def repo_from_remote_url(url: str) -> str | None:
    """``owner/name`` from a git remote URL, with any credentials dropped.

    Handles ``https://user:token@host/owner/name.git``, ``ssh://git@host/...``
    and scp-like ``git@host:owner/name.git``. The userinfo part is discarded
    before anything else so a token embedded in a remote can never become a
    label.
    """
    raw = url.strip()
    if not raw:
        return None
    if "://" in raw:
        rest = raw.split("://", 1)[1]
        rest = rest.split("@", 1)[1] if "@" in rest.split("/", 1)[0] else rest
        path = rest.split("/", 1)[1] if "/" in rest else ""
    else:
        rest = raw.split("@", 1)[1] if "@" in raw else raw
        path = rest.split(":", 1)[1] if ":" in rest else rest
    parts = [p for p in path.replace("\\", "/").split("/") if p]
    if not parts:
        return None
    name = parts[-1]
    if name.endswith(".git"):
        name = name[: -len(".git")]
    return "/".join([*parts[-2:-1], name]) or None


def read_origin_repo(git_dir: Path) -> str | None:
    """``owner/name`` of the ``origin`` remote, from the repository config."""
    try:
        lines = (_common_dir(git_dir) / "config").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    in_origin = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("["):
            in_origin = stripped.replace(" ", "") in ('[remote"origin"]',)
            continue
        if in_origin and stripped.startswith("url"):
            _, sep, value = stripped.partition("=")
            if sep:
                return repo_from_remote_url(value)
    return None


@dataclass
class _GitCacheEntry:
    at: float
    repo: str | None
    branch: str | None


class GitLabels:
    """Repo/branch for a workspace root, cached briefly per root."""

    def __init__(self, ttl_s: float = GIT_CACHE_TTL_S, clock: Any = time.monotonic) -> None:
        self._ttl = ttl_s
        self._clock = clock
        self._cache: dict[str, _GitCacheEntry] = {}
        self._lock = threading.Lock()

    def lookup(self, root: str | None) -> tuple[str | None, str | None]:
        if not root:
            return None, None
        now = self._clock()
        with self._lock:
            hit = self._cache.get(root)
            if hit is not None and now - hit.at < self._ttl:
                return hit.repo, hit.branch
        git_dir = _git_dir(Path(root))
        repo = branch = None
        if git_dir is not None:
            branch = read_branch(git_dir)
            repo = read_origin_repo(git_dir) or Path(root).name or None
        with self._lock:
            if len(self._cache) > 256:
                self._cache.clear()
            self._cache[root] = _GitCacheEntry(at=now, repo=repo, branch=branch)
        return repo, branch


_git_labels = GitLabels()


def _registered_root() -> str | None:
    from headroom.proxy.project_context import get_registered_cwd

    return get_registered_cwd()


def resolve_repo(request: Any) -> str | None:
    return request.header(REPO_HEADER) or _git_labels.lookup(_registered_root())[0]


def resolve_branch(request: Any) -> str | None:
    return request.header(BRANCH_HEADER) or _git_labels.lookup(_registered_root())[1]


# ── report accumulation ───────────────────────────────────────────────


def _estimate_costs(snapshot: Any) -> tuple[float, float]:
    """(input_usd, output_usd) for one request, from the shared price table."""
    from headroom.proxy.savings_tracker import (
        _estimate_input_cost_usd,
        _estimate_output_cost_usd,
    )

    read = max(int(snapshot.cache_read_tokens or 0), 0)
    write = max(int(snapshot.cache_write_tokens or 0), 0)
    total = max(int(snapshot.input_tokens or 0), 0)
    uncached = max(total - read - write, 0)
    try:
        input_usd = _estimate_input_cost_usd(
            snapshot.model,
            total,
            cache_read_tokens=read,
            cache_write_tokens=write,
            uncached_input_tokens=uncached,
        )
        output_usd = _estimate_output_cost_usd(snapshot.model, int(snapshot.output_tokens or 0))
    except Exception:  # noqa: BLE001 - an unpriced model costs 0, never fails
        return 0.0, 0.0
    return float(input_usd or 0.0), float(output_usd or 0.0)


@dataclass
class CacheRow:
    requests: int = 0
    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    busts: int = 0
    bust_write_tokens: int = 0
    cost_usd: float = 0.0
    last_seen: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        cached = self.cache_read_tokens + self.cache_write_tokens
        return {
            "requests": self.requests,
            "input_tokens": self.input_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cache_read_share": round(self.cache_read_tokens / cached, 4) if cached else None,
            "busts": self.busts,
            "bust_rate": round(self.busts / self.requests, 4) if self.requests else 0.0,
            "bust_write_tokens": self.bust_write_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }


@dataclass
class SessionRow:
    key: str
    labels: dict[str, str] = field(default_factory=dict)
    models: set[str] = field(default_factory=set)
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tokens_saved: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    busts: int = 0
    cost_usd: float = 0.0
    first_seen: float = 0.0
    last_seen: float = 0.0
    reading: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation": self.key,
            "labels": dict(self.labels),
            "models": sorted(self.models),
            "requests": self.requests,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "tokens_saved": self.tokens_saved,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "busts": self.busts,
            "cost_usd": round(self.cost_usd, 6),
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "duration_s": round(max(self.last_seen - self.first_seen, 0.0), 1),
        }


class AttributionReport:
    """Pipeline extension accumulating the report. Thread-safe, bounded."""

    def __init__(
        self,
        *,
        max_sessions: int = MAX_SESSIONS,
        max_labels: int = MAX_LABELS,
        bust_min_write_tokens: int = BUST_MIN_WRITE_TOKENS,
        clock: Any = time.time,
    ) -> None:
        self._max_sessions = max_sessions
        self._max_labels = max_labels
        self._bust_min = bust_min_write_tokens
        self._clock = clock
        self._lock = threading.Lock()
        self._sessions: OrderedDict[str, SessionRow] = OrderedDict()
        self._cache: dict[str, dict[str, CacheRow]] = {}
        self._totals = CacheRow()
        self.started_at = clock()

    # PipelineExtension protocol
    def on_pipeline_event(self, event: Any) -> None:
        from headroom.pipeline import PipelineStage

        if event.stage is PipelineStage.OUTCOME_OBSERVED and event.outcome is not None:
            self.observe(event.outcome)
        return None

    def observe(self, snapshot: Any) -> None:
        input_usd, output_usd = _estimate_costs(snapshot)
        cost = input_usd + output_usd
        read = max(int(snapshot.cache_read_tokens or 0), 0)
        write = max(int(snapshot.cache_write_tokens or 0), 0)
        labels: Mapping[str, str] = snapshot.attribution or {}
        now = self._clock()
        with self._lock:
            bust = False
            key = snapshot.conversation_key
            if key:
                session = self._sessions.get(key)
                if session is None:
                    session = SessionRow(key=key, first_seen=now)
                    self._sessions[key] = session
                    while len(self._sessions) > self._max_sessions:
                        self._sessions.popitem(last=False)
                else:
                    self._sessions.move_to_end(key)
                # A conversation that was reading its prefix and now re-writes
                # a large one has lost the cache: the expensive event.
                bust = session.reading and read == 0 and write >= self._bust_min
                session.reading = read > 0 or (session.reading and write < self._bust_min)
                session.labels.update(labels)
                if snapshot.model:
                    session.models.add(str(snapshot.model))
                session.requests += 1
                session.input_tokens += max(int(snapshot.input_tokens or 0), 0)
                session.output_tokens += max(int(snapshot.output_tokens or 0), 0)
                session.tokens_saved += max(int(snapshot.novel_tokens_saved or 0), 0)
                session.cache_read_tokens += read
                session.cache_write_tokens += write
                session.busts += int(bust)
                session.cost_usd += cost
                session.last_seen = now
            rows = [self._totals]
            for dim, label in labels.items():
                per_dim = self._cache.setdefault(dim, {})
                row = per_dim.get(label)
                if row is None:
                    if len(per_dim) >= self._max_labels:
                        evict = min(
                            per_dim, key=lambda k: (per_dim[k].cost_usd, per_dim[k].last_seen)
                        )
                        del per_dim[evict]
                    row = per_dim[label] = CacheRow()
                rows.append(row)
            for row in rows:
                row.requests += 1
                row.input_tokens += max(int(snapshot.input_tokens or 0), 0)
                row.cache_read_tokens += read
                row.cache_write_tokens += write
                row.busts += int(bust)
                row.bust_write_tokens += write if bust else 0
                row.cost_usd += cost
                row.last_seen = now

    def report(self, *, top: int = 20) -> dict[str, Any]:
        with self._lock:
            sessions = sorted(self._sessions.values(), key=lambda s: s.cost_usd, reverse=True)
            cache = {
                dim: {
                    label: row.to_dict()
                    for label, row in sorted(
                        rows.items(), key=lambda kv: kv[1].cost_usd, reverse=True
                    )
                }
                for dim, rows in self._cache.items()
            }
            totals = self._totals.to_dict()
            top_sessions = [s.to_dict() for s in sessions[: max(top, 0)]]
            tracked = len(self._sessions)
        return {
            "since": self.started_at,
            "totals": totals,
            "cache_by_dimension": cache,
            "top_sessions": top_sessions,
            "sessions_tracked": tracked,
            "insights": insights(totals, cache),
            "notes": [
                "Costs are estimates from the shared price table, input and output.",
                "Labels are self-reported by clients; they attribute usage, they "
                "do not authenticate anyone.",
            ],
        }


def insights(totals: Mapping[str, Any], cache: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Plain-language findings worth acting on, most expensive first.

    Deliberately conservative: a finding needs enough requests behind it to be
    more than noise, and is phrased as a comparison against the whole fleet so
    it reads as "look here", never as a verdict.
    """
    found: list[tuple[float, str]] = []
    fleet_requests = int(totals.get("requests") or 0)
    fleet_cost = float(totals.get("cost_usd") or 0.0)
    fleet_bust = float(totals.get("bust_rate") or 0.0)
    if fleet_requests < 20 or fleet_cost <= 0:
        return []
    for dim, rows in cache.items():
        if len(rows) < 2:
            continue
        for label, row in rows.items():
            requests = int(row.get("requests") or 0)
            if requests < 10:
                continue
            cost = float(row.get("cost_usd") or 0.0)
            share = cost / fleet_cost
            bust_rate = float(row.get("bust_rate") or 0.0)
            if bust_rate >= max(0.05, 2 * fleet_bust) and int(row.get("busts") or 0) >= 3:
                found.append(
                    (
                        cost,
                        f"{dim}={label} loses the prompt cache on {bust_rate:.0%} of requests "
                        f"(fleet {fleet_bust:.0%}); {int(row['bust_write_tokens']):,} tokens "
                        "were re-written at the cache-write rate.",
                    )
                )
            read_share = row.get("cache_read_share")
            if (
                read_share is not None
                and read_share < 0.5
                and int(row.get("cache_write_tokens") or 0) > 100_000
            ):
                found.append(
                    (
                        cost,
                        f"{dim}={label} writes more cache than it reads (read share "
                        f"{read_share:.0%}): its prefix rarely survives to the next turn.",
                    )
                )
            if share >= 0.5 and len(rows) >= 3:
                found.append((cost, f"{dim}={label} accounts for {share:.0%} of estimated spend."))
    found.sort(key=lambda item: item[0], reverse=True)
    return [text for _, text in found[:10]]


# ── extension entry point ─────────────────────────────────────────────

_report: AttributionReport | None = None


def get_report() -> AttributionReport | None:
    return _report


def install(app: Any, config: Any) -> None:
    """``headroom.proxy_extension`` entry point."""
    global _report
    from headroom.proxy.attribution import register_dimension

    if os.environ.get("HEADROOM_ATTRIBUTION_GIT", "1").strip().lower() not in ("0", "false", "off"):
        register_dimension("repo", resolve_repo, replace=True)
        register_dimension("branch", resolve_branch, replace=True)

    proxy = getattr(app.state, "proxy", None)
    manager = getattr(proxy, "pipeline_extensions", None)
    if manager is None or not callable(getattr(manager, "add", None)):
        raise RuntimeError("attribution report needs the proxy's pipeline extension manager")
    report = AttributionReport()
    manager.add(report)
    _report = report

    @app.get("/attribution/report")
    async def attribution_report(request: Request, top: int = 20) -> dict[str, Any]:
        gate = getattr(request.app.state, "require_operator_read", None)
        if gate is None:
            raise HTTPException(status_code=404)
        gate(request)
        payload = report.report(top=min(max(top, 0), 200))
        tracker = getattr(getattr(proxy, "metrics", None), "savings_tracker", None)
        if tracker is not None:
            payload["savings_by_dimension"] = tracker.snapshot().get("by_dimension", {})
        return payload


__all__ = [
    "AttributionReport",
    "GitLabels",
    "get_report",
    "insights",
    "install",
    "read_branch",
    "read_origin_repo",
    "repo_from_remote_url",
    "resolve_branch",
    "resolve_repo",
]
