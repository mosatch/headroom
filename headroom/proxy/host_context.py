"""Per-request host attribution for the proxy.

``headroom wrap`` launches agents with an ``X-Headroom-Host`` header (via
``ANTHROPIC_CUSTOM_HEADERS`` for Claude Code and ``env_http_headers`` for Codex)
naming the machine the agent runs on. The proxy captures that header once per
request — in the HTTP middleware for regular requests and at the WebSocket
accept for Codex responses-WS sessions — into a :mod:`contextvars` variable, so
the outcome funnel can attribute savings to a host without threading a
parameter through every handler.

This mirrors :mod:`headroom.proxy.project_context` exactly; see
:mod:`headroom.proxy.host_policy` for why the header is the primary key and the
peer address only the fallback.
"""

from __future__ import annotations

from contextvars import ContextVar

from headroom.proxy.host_policy import (
    HOST_HEADER,
    classify_host,
    local_addresses,
    resolve_host,
    sanitize_host_name,
)

_current_host: ContextVar[str | None] = ContextVar("headroom_current_host", default=None)

# Enumerated once per process: the interface list is stable for the proxy's
# lifetime, and getaddrinfo is far too slow to call on every request.
_LOCAL_ADDRESSES: frozenset[str] | None = None


def get_local_addresses() -> frozenset[str]:
    """This machine's addresses, enumerated once and reused."""
    global _LOCAL_ADDRESSES
    if _LOCAL_ADDRESSES is None:
        _LOCAL_ADDRESSES = local_addresses()
    return _LOCAL_ADDRESSES


def reset_local_addresses() -> None:
    """Drop the cached address set. For tests, and after a network change."""
    global _LOCAL_ADDRESSES
    _LOCAL_ADDRESSES = None


def set_current_host(host: str | None) -> None:
    """Bind the active request's host for downstream outcome recording."""
    _current_host.set(sanitize_host_name(host))


def get_current_host() -> str | None:
    """Host bound to the current request context, or ``None``."""
    return _current_host.get()


def bind_host_from_request(headers, peer_address: str | None) -> str | None:
    """Resolve and bind the host for a request, returning what was bound."""
    host = resolve_host(
        headers,
        peer_address=peer_address,
        local_addresses=get_local_addresses(),
    )
    set_current_host(host)
    return host


__all__ = [
    "HOST_HEADER",
    "bind_host_from_request",
    "classify_host",
    "get_current_host",
    "get_local_addresses",
    "reset_local_addresses",
    "set_current_host",
]
