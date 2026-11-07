"""Optional enforcement: refuse requests that do not say who they are.

Attribution is best-effort by default. A caller that sends no label is simply
left out of the per-label tables and counted only in the totals. That is right
for a single-user proxy, where there is nothing to tell apart.

An operator running a shared gateway needs the opposite. If any caller can opt
out of identifying itself, the per-host or per-team tables are advisory rather
than complete, and no chargeback question can be answered from them. Naming a
dimension in ``HEADROOM_REQUIRE_DIMENSIONS`` (``--require-dimension``) makes
its label a precondition for service: a request without one is refused with a
provider-shaped 400 (WebSocket: close 1008) that says exactly what to send.

**What counts as present.** The label must come from what the *client* sent. A
fallback the proxy derives on its own would satisfy every request and make the
requirement a no-op, so it is not consulted: ``host`` needs ``X-Headroom-Host``
(the peer address does not count), and ``traffic_class`` needs a client that is
identified (``unknown`` does not count).

**Failing closed.** Every path that is not explicitly operational is covered,
so a provider route added later is covered automatically.

**Never locking out the operator.** Health probes, the dashboard, stats and
metrics are exempt unconditionally, so turning enforcement on cannot cost you
the UI you would use to see its effect.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from typing import Any, Final

from headroom.proxy.attribution import (
    AttributionRequest,
    header_name_for,
    resolve_attribution,
    sanitize_dimension_name,
)

REQUIRE_DIMENSIONS_ENV_VAR: Final[str] = "HEADROOM_REQUIRE_DIMENSIONS"

#: Labels that mean "the proxy could not tell", which never satisfy a requirement.
_UNKNOWN_LABELS: Final[frozenset[str]] = frozenset({"unknown"})

#: Paths the proxy serves for its own operation rather than to proxy a model
#: call. Matched as prefixes so sub-paths (``/dashboard/static/...``) inherit
#: the exemption.
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
    "/attribution",
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
    # Anthropic's connectivity canary, forwarded upstream: a reachability probe,
    # not billable traffic, and refusing it reads as "proxy unreachable".
    "/api/hello",
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


def required_dimensions(
    configured: Iterable[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Required dimensions from config, else ``HEADROOM_REQUIRE_DIMENSIONS``."""
    if configured is None:
        env = os.environ if environ is None else environ
        configured = (env.get(REQUIRE_DIMENSIONS_ENV_VAR) or "").split(",")
    names = (sanitize_dimension_name(n) for n in configured)
    return tuple(dict.fromkeys(n for n in names if n))


def missing_dimensions(
    request: AttributionRequest,
    required: Iterable[str],
) -> tuple[str, ...]:
    """Required dimensions this request carries no client-sent label for.

    Resolved with the peer address withheld, so no proxy-side fallback can
    stand in for what the client was asked to send.
    """
    required = tuple(required)
    if not required or not requires_attribution(request.path):
        return ()
    explicit = AttributionRequest(
        headers=request.headers, path=request.path, peer=None, path_project=request.path_project
    )
    labels = resolve_attribution(explicit, required)
    return tuple(name for name in required if labels.get(name) in (None, *_UNKNOWN_LABELS))


def attribution_error_payload(missing: tuple[str, ...]) -> dict[str, Any]:
    """Provider-shaped error body naming exactly what the caller must send.

    Uses the Anthropic error envelope, whose ``error.message`` an OpenAI-format
    client also reads, so the refusal renders as a real message in either
    harness instead of an opaque failure.
    """
    headers = [header_name_for(name).title() for name in missing]
    hint = (
        "This proxy requires attribution. Launch the client with `headroom wrap`, "
        "which sends the host and project, or send the header(s) yourself; "
        "X-Headroom-Labels: name=value; ... also works for custom dimensions."
    )
    if "project" in missing:
        hint += " A /p/<project-name> base-URL prefix also satisfies project."
    return {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": (
                f"Missing required attribution: {', '.join(missing)} ({', '.join(headers)}). {hint}"
            ),
            "missing_dimensions": list(missing),
        },
    }


__all__ = [
    "OPERATIONAL_PATH_PREFIXES",
    "REQUIRE_DIMENSIONS_ENV_VAR",
    "attribution_error_payload",
    "is_operational_path",
    "missing_dimensions",
    "required_dimensions",
    "requires_attribution",
]
