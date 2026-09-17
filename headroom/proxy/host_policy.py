"""Pure host attribution policy helpers.

A Headroom proxy bound to anything other than loopback serves more than one
machine: a shared LAN gateway is a supported deployment, but every aggregate it
reports — savings, spend, request counts — collapses those machines into one
number. Host attribution adds the missing axis, mirroring the project axis in
:mod:`headroom.proxy.project_policy`.

The identity is *self-reported*. ``headroom wrap`` sends ``X-Headroom-Host``
carrying the client machine's own hostname, exactly as it already sends
``X-Headroom-Project``. Reverse DNS is deliberately not consulted: a proxy on a
home or office LAN typically has no PTR records to read (and mDNS publishes
forward names only), so a resolver-based scheme degrades to bare octets on the
deployments that need this most, while adding a blocking network call to the
request path.

The peer address is the fallback, not the primary key, because one machine
routinely reaches the proxy from several addresses — loopback plus one address
per interface. Keying on the address alone reports a single host several times
over, so :func:`coalesce_local_address` folds the local machine's own addresses
onto its hostname before the address is ever used as a label.
"""

from __future__ import annotations

import socket
from collections.abc import Mapping
from typing import Any

HOST_HEADER = "x-headroom-host"

#: Matches ``PROJECT_NAME_MAX_LENGTH``. Long enough for any FQDN a client can
#: legitimately report (RFC 1035 caps a name at 255), short enough that a
#: misbehaving client cannot bloat persisted state or the dashboard payload.
HOST_NAME_MAX_LENGTH = 128

#: Suffixes stripped before keying. Bonjour hands the same machine back as
#: ``mbmini.local`` while ``scutil --get LocalHostName`` calls it ``mbmini``;
#: without this the two spellings accumulate as separate hosts.
_STRIPPED_SUFFIXES: tuple[str, ...] = (".local.", ".local")

_LOOPBACK_ADDRESSES = frozenset({"127.0.0.1", "::1", "localhost"})


def sanitize_host_name(value: Any) -> str | None:
    """Normalize a client-supplied host name; ``None`` when unusable.

    Lowercased because DNS names are case-insensitive: a machine that reports
    ``MBMini`` through one client and ``mbmini`` through another is one host,
    and case-sensitive keying would split its history in two.
    """
    if not isinstance(value, str):
        return None
    cleaned = "".join(ch for ch in value if ch.isprintable()).strip()
    # A trailing dot is the DNS root label; it names the same host. A leading
    # dot is malformed, and dropping it keeps a stray ``.local`` from keying as
    # a host distinct from every other empty-labelled spelling.
    cleaned = cleaned.strip(".").strip().lower()
    if not cleaned:
        return None
    for suffix in _STRIPPED_SUFFIXES:
        if cleaned.endswith(suffix) and len(cleaned) > len(suffix):
            cleaned = cleaned[: -len(suffix)]
            break
    cleaned = cleaned.strip()
    if not cleaned:
        return None
    return cleaned[:HOST_NAME_MAX_LENGTH]


def classify_host(headers: Mapping[str, Any] | Any) -> str | None:
    """Extract a sanitized host name from request headers, if present."""
    get = getattr(headers, "get", None)
    if get is None:
        return None
    value = get(HOST_HEADER) or get("X-Headroom-Host")
    return sanitize_host_name(value)


def local_host_name() -> str | None:
    """This machine's own short hostname, or ``None`` if it cannot be read.

    Not cached: the value is read once per proxy start in practice, and a
    module-level cache would survive a hostname change across a test run.
    """
    try:
        return sanitize_host_name(socket.gethostname())
    except Exception:  # pragma: no cover - defensive; gethostname rarely fails
        return None


def coalesce_local_address(address: str | None, *, local_addresses: frozenset[str]) -> str | None:
    """Fold an address belonging to this machine onto its hostname.

    ``local_addresses`` is supplied by the caller rather than probed here so the
    interface enumeration happens once at startup, not per request.
    """
    if not address:
        return None
    if address in _LOOPBACK_ADDRESSES or address in local_addresses:
        return local_host_name() or address
    return address


def local_addresses() -> frozenset[str]:
    """Every address this machine answers on, for :func:`coalesce_local_address`.

    Best-effort: a partial set costs attribution precision (an address of ours
    is labelled as itself instead of by name), never correctness.
    """
    found: set[str] = set(_LOOPBACK_ADDRESSES)
    try:
        hostname = socket.gethostname()
    except Exception:  # pragma: no cover - defensive
        return frozenset(found)
    for family in (socket.AF_INET, socket.AF_INET6):
        try:
            for info in socket.getaddrinfo(hostname, None, family):
                sockaddr = info[4]
                if sockaddr and isinstance(sockaddr[0], str):
                    # Strip any IPv6 scope id (``fe80::1%en0``) so the value
                    # compares equal to the peer address the server reports.
                    found.add(sockaddr[0].partition("%")[0])
        except Exception:
            continue
    return frozenset(found)


def resolve_host(
    headers: Mapping[str, Any] | Any,
    *,
    peer_address: str | None = None,
    local_addresses: frozenset[str] | None = None,
) -> str | None:
    """Best available host identity for a request.

    Order: the explicit ``X-Headroom-Host`` header, then the peer address with
    this machine's own addresses folded onto its hostname. ``None`` when neither
    yields anything usable, which leaves attribution off for that request —
    matching how unattributed project traffic is skipped.
    """
    named = classify_host(headers)
    if named is not None:
        return named
    coalesced = coalesce_local_address(
        peer_address,
        local_addresses=local_addresses if local_addresses is not None else frozenset(),
    )
    return sanitize_host_name(coalesced)


__all__ = [
    "HOST_HEADER",
    "HOST_NAME_MAX_LENGTH",
    "classify_host",
    "coalesce_local_address",
    "local_addresses",
    "local_host_name",
    "resolve_host",
    "sanitize_host_name",
]
