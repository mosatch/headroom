"""Built-in attribution dimensions.

Each resolver reads a signal the proxy already understands; none of them adds a
new trust assumption:

``project``
    ``X-Headroom-Project`` (sent by ``headroom wrap``) or a ``/p/<name>``
    base-URL prefix, exactly as :mod:`headroom.proxy.project_context` reads it.
``client``
    The harness, from :func:`headroom.proxy.auth_mode.classify_client`.
``host``
    The calling machine: ``X-Headroom-Host`` (sent by ``headroom wrap``), else
    the peer address with this machine's own addresses folded onto its name.
    See :mod:`headroom.proxy.host_policy`.
``traffic_class``
    ``interactive`` for a person at an agent, ``automation`` for headless SDK and
    ``-p`` runs, ``unknown`` otherwise. Splitting these keeps a nightly batch job
    from reading as the heaviest developer.
``tenant``
    The tenant-id header TOIN partitions on (``HEADROOM_TENANT_KEY_HEADER``),
    only when a caller sent one.
"""

from __future__ import annotations

import threading
from typing import Final

from headroom.proxy.attribution import AttributionRequest, _register_if_absent

TRAFFIC_INTERACTIVE: Final[str] = "interactive"
TRAFFIC_AUTOMATION: Final[str] = "automation"
TRAFFIC_UNKNOWN: Final[str] = "unknown"

#: Explicit override, for a client that knows what it is (a CI job, a cron).
TRAFFIC_CLASS_HEADER: Final[str] = "x-headroom-traffic-class"

#: User-agent fragments that mark a headless run of an otherwise interactive
#: harness. Claude Code reports its entrypoint in the UA comment:
#: ``claude-cli/2.1.0 (external, cli)`` for a person at the terminal,
#: ``(external, sdk-cli)`` / ``sdk-ts`` / ``sdk-py`` for ``-p`` and the Agent SDK.
_AUTOMATION_UA_MARKERS: Final[tuple[str, ...]] = (
    ", sdk-",
    "(sdk-",
    "claude-agent-sdk",
    "github-actions",
)

_local_addresses: frozenset[str] | None = None
_local_addresses_lock = threading.Lock()


def _cached_local_addresses() -> frozenset[str]:
    # Interface enumeration does a resolver call; once per process is enough.
    global _local_addresses
    if _local_addresses is None:
        with _local_addresses_lock:
            if _local_addresses is None:
                from headroom.proxy.host_policy import local_addresses

                _local_addresses = local_addresses()
    return _local_addresses


def resolve_project(request: AttributionRequest) -> str | None:
    from headroom.proxy.project_policy import classify_project

    return classify_project(request.headers) or request.path_project


def resolve_client(request: AttributionRequest) -> str | None:
    from headroom.proxy.auth_mode import classify_client

    return classify_client(request.headers)


def resolve_host(request: AttributionRequest) -> str | None:
    from headroom.proxy.host_policy import resolve_host as _resolve_host

    return _resolve_host(
        request.headers,
        peer_address=request.peer,
        local_addresses=_cached_local_addresses(),
    )


def classify_traffic(user_agent: str | None, client: str | None, explicit: str | None) -> str:
    """``interactive`` / ``automation`` / ``unknown`` for one request."""
    if explicit:
        cleaned = explicit.strip().lower()
        if cleaned in (TRAFFIC_INTERACTIVE, TRAFFIC_AUTOMATION):
            return cleaned
    ua = (user_agent or "").lower()
    if any(marker in ua for marker in _AUTOMATION_UA_MARKERS):
        return TRAFFIC_AUTOMATION
    if client:
        return TRAFFIC_INTERACTIVE
    return TRAFFIC_UNKNOWN


def resolve_traffic_class(request: AttributionRequest) -> str:
    return classify_traffic(
        request.header("user-agent"),
        resolve_client(request),
        request.header(TRAFFIC_CLASS_HEADER),
    )


def resolve_tenant(request: AttributionRequest) -> str | None:
    import os

    from headroom.proxy.tenant_key import (
        DEFAULT_TENANT_KEY_HEADER,
        TENANT_KEY_HEADER_ENV_VAR,
        _sanitize_tenant_key,
    )

    header = os.environ.get(TENANT_KEY_HEADER_ENV_VAR, "").strip() or DEFAULT_TENANT_KEY_HEADER
    raw = request.header(header)
    if not raw:
        return None
    return _sanitize_tenant_key(raw) or None


BUILTIN_RESOLVERS = (
    ("project", resolve_project),
    ("client", resolve_client),
    ("host", resolve_host),
    ("traffic_class", resolve_traffic_class),
    ("tenant", resolve_tenant),
)


def install_builtin_dimensions() -> None:
    """Register the built-ins. Idempotent; an extension's resolver is kept."""
    for name, resolver in BUILTIN_RESOLVERS:
        _register_if_absent(name, resolver)


__all__ = [
    "TRAFFIC_AUTOMATION",
    "TRAFFIC_CLASS_HEADER",
    "TRAFFIC_INTERACTIVE",
    "TRAFFIC_UNKNOWN",
    "BUILTIN_RESOLVERS",
    "classify_traffic",
    "install_builtin_dimensions",
]
