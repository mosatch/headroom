"""Tests for pure host attribution policy helpers."""

from __future__ import annotations

from headroom.proxy.host_policy import (
    classify_host,
    coalesce_local_address,
    resolve_host,
    sanitize_host_name,
)


def test_sanitize_host_name_normalizes_case_and_whitespace() -> None:
    assert sanitize_host_name(" MBMini ") == "mbmini"
    assert sanitize_host_name("MBMini") == sanitize_host_name("mbmini")
    assert sanitize_host_name("") is None
    assert sanitize_host_name("   ") is None
    assert sanitize_host_name(None) is None
    assert sanitize_host_name(42) is None


def test_sanitize_host_name_folds_bonjour_and_root_labels() -> None:
    # The same machine, spelled three ways by three resolvers.
    assert sanitize_host_name("MBMini.local") == "mbmini"
    assert sanitize_host_name("mbmini.local.") == "mbmini"
    assert sanitize_host_name("mbmini") == "mbmini"
    # A real domain is not a Bonjour suffix and must survive intact.
    assert sanitize_host_name("build01.corp.example.com") == "build01.corp.example.com"
    # ``.local`` alone is the suffix, not a name; stripping it would empty it.
    assert sanitize_host_name(".local") == "local"


def test_sanitize_host_name_caps_length() -> None:
    assert len(sanitize_host_name("h" * 500)) == 128


def test_classify_host_reads_host_header() -> None:
    assert classify_host({"x-headroom-host": " MBMini "}) == "mbmini"
    assert classify_host({"X-Headroom-Host": "build01"}) == "build01"
    assert classify_host({"user-agent": "claude-cli"}) is None
    assert classify_host(object()) is None


def test_coalesce_local_address_folds_loopback_and_own_addresses() -> None:
    local = frozenset({"127.0.0.1", "::1", "192.168.0.156"})

    # Loopback and a real interface address are the same machine.
    assert coalesce_local_address("127.0.0.1", local_addresses=local) == coalesce_local_address(
        "192.168.0.156", local_addresses=local
    )
    # A peer that is not us is left alone.
    assert coalesce_local_address("192.168.0.12", local_addresses=local) == "192.168.0.12"
    assert coalesce_local_address(None, local_addresses=local) is None


def test_resolve_host_prefers_header_over_peer_address() -> None:
    local = frozenset({"127.0.0.1"})

    assert (
        resolve_host(
            {"x-headroom-host": "build01"},
            peer_address="192.168.0.12",
            local_addresses=local,
        )
        == "build01"
    )


def test_resolve_host_falls_back_to_peer_address() -> None:
    local = frozenset({"127.0.0.1"})

    assert resolve_host({}, peer_address="192.168.0.12", local_addresses=local) == "192.168.0.12"
    # Nothing to attribute to: stays unattributed rather than inventing a name.
    assert resolve_host({}, peer_address=None, local_addresses=local) is None
