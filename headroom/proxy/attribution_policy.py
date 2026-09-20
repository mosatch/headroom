"""Optional enforcement of project / host attribution on inbound requests.

Attribution is best-effort by default: a caller that sends no
``X-Headroom-Project`` or ``X-Headroom-Host`` is recorded as unattributed and
its savings land only in the aggregates. That is the right default for the
single-user local proxy, where there is nothing to tell apart.

An operator running a shared gateway needs the opposite: if any caller can opt
out of identifying itself, the per-project and per-host tables are advisory
rather than complete, and no chargeback or capacity question can be answered
from them. Enabling enforcement makes attribution a precondition for service —
an unattributed request is refused rather than silently pooled.

Two properties matter for safety:

**Failing closed.** Every path that is not explicitly operational requires
attribution. A provider route added later is therefore covered automatically;
the alternative (an allowlist of provider paths) would silently leak new routes
into the unattributed bucket, which is exactly the hole enforcement exists to
close.

**Never locking out the operator.** The operational surface — health probes,
the dashboard, stats, metrics — is exempt unconditionally, so turning
enforcement on cannot cost you the UI you would use to see its effect, and
cannot fail a container health check. ``tests/test_attribution_enforcement.py``
asserts that every route the proxy registers for its own operation is exempt,
so a new operational endpoint cannot regress this by omission.

Host enforcement requires the *header* specifically. ``resolve_host`` falls back
to the peer address, so a host value is almost always available and requiring
"a host" would be satisfied by every request — a no-op. Requiring the header is
the only form of the option that means anything.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from headroom.proxy.host_policy import HOST_HEADER
from headroom.proxy.project_policy import PROJECT_HEADER

#: Paths the proxy serves for its own operation rather than to proxy a model
#: call. Matched as prefixes so sub-paths (``/dashboard/static/...``,
#: ``/v1/telemetry/export``) inherit the exemption.
OPERATIONAL_PATH_PREFIXES: tuple[str, ...] = (
    "/health",
    "/livez",
    "/readyz",
    "/metrics",
    "/quota",
    "/subscription-window",
    "/stats",
    "/stats-history",
    "/stats-lifetime",
    "/dashboard",
    "/settings",
    "/favicon.ico",
    "/admin/",
    "/debug/",
    "/transformations/",
    # Headroom's own service APIs. These are tools acting on behalf of an
    # already-attributed session, not billable model traffic of their own.
    "/v1/compress",
    "/v1/usage",
    "/v1/retrieve",
    "/v1/feedback",
    "/v1/telemetry",
    "/v1/toin",
    # OpenAPI/docs, served by FastAPI itself.
    "/docs",
    "/redoc",
    "/openapi.json",
)


def is_operational_path(path: str) -> bool:
    """Whether ``path`` is served for the proxy's own operation.

    Exact match or a ``/``-delimited prefix match, so ``/statsify`` is not
    mistaken for ``/stats``.
    """
    if not isinstance(path, str) or not path:
        return False
    candidate = path.rstrip("/") or "/"
    for prefix in OPERATIONAL_PATH_PREFIXES:
        trimmed = prefix.rstrip("/")
        if candidate == trimmed or candidate.startswith(trimmed + "/"):
            return True
    return False


def requires_attribution(path: str) -> bool:
    """Whether a request to ``path`` is subject to enforcement when enabled."""
    return not is_operational_path(path)


def _has_header(headers: Mapping[str, Any] | Any, name: str) -> bool:
    get = getattr(headers, "get", None)
    if get is None:
        return False
    value = get(name) or get(name.title()) or get(name.upper())
    return bool(isinstance(value, str) and value.strip())


def missing_attribution(
    headers: Mapping[str, Any] | Any,
    *,
    path: str,
    require_project: bool,
    require_host: bool,
    prefix_project: str | None = None,
) -> tuple[str, ...]:
    """Attribution requirements this request fails to satisfy.

    Empty when the request is acceptable — either because nothing is required,
    the path is operational, or every required identifier is present.
    ``prefix_project`` is the ``/p/<name>`` value the scope strip produced; it
    satisfies the project requirement for clients that cannot send headers.
    """
    if not (require_project or require_host):
        return ()
    if not requires_attribution(path):
        return ()
    missing: list[str] = []
    if require_project and not (_has_header(headers, PROJECT_HEADER) or prefix_project):
        missing.append(PROJECT_HEADER)
    if require_host and not _has_header(headers, HOST_HEADER):
        missing.append(HOST_HEADER)
    return tuple(missing)


def attribution_error_payload(missing: tuple[str, ...]) -> dict[str, Any]:
    """Provider-shaped error body naming exactly what the caller must send.

    Uses the Anthropic error envelope, whose ``error.message`` an OpenAI-format
    client also reads, so the refusal renders as a real message in either
    harness instead of an opaque failure.
    """
    names = ", ".join(missing)
    hint = (
        "This proxy requires attribution headers. Launch the client with "
        "`headroom wrap`, which sends them automatically, or set them "
        "explicitly."
    )
    if PROJECT_HEADER in missing:
        hint += " A /p/<project-name> base-URL prefix also satisfies the project requirement."
    return {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": f"Missing required attribution header(s): {names}. {hint}",
            "missing_headers": list(missing),
        },
    }


__all__ = [
    "OPERATIONAL_PATH_PREFIXES",
    "attribution_error_payload",
    "is_operational_path",
    "missing_attribution",
    "requires_attribution",
]
