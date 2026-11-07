"""`headroom wrap` sends attribution headers: host and user labels."""

from __future__ import annotations

from headroom.cli import wrap as wrap_mod


def test_host_and_labels_appended_to_custom_headers(monkeypatch):
    monkeypatch.setenv("HEADROOM_HOST_LABEL", "build01")
    monkeypatch.setenv("HEADROOM_LABELS", "team=infra, owner=sam")
    env = {"ANTHROPIC_CUSTOM_HEADERS": "X-Foo: 1"}
    wrap_mod._apply_attribution_header_env(env)
    assert env["ANTHROPIC_CUSTOM_HEADERS"].splitlines() == [
        "X-Foo: 1",
        "X-Headroom-Host: build01",
        "X-Headroom-Labels: team=infra; owner=sam",
    ]


def test_user_supplied_header_wins(monkeypatch):
    monkeypatch.delenv("HEADROOM_LABELS", raising=False)
    env = {"ANTHROPIC_CUSTOM_HEADERS": "x-headroom-host: mine"}
    wrap_mod._apply_attribution_header_env(env)
    assert env["ANTHROPIC_CUSTOM_HEADERS"] == "x-headroom-host: mine"


def test_host_defaults_to_hostname_and_is_header_safe(monkeypatch):
    monkeypatch.delenv("HEADROOM_HOST_LABEL", raising=False)
    monkeypatch.setattr(wrap_mod.socket, "gethostname", lambda: "Büro Mac")
    assert wrap_mod._host_name_for_header() == "B%C3%BCro Mac"
    monkeypatch.setattr(wrap_mod.socket, "gethostname", lambda: "  ")
    assert wrap_mod._host_name_for_header() is None


def test_no_labels_means_no_labels_header(monkeypatch):
    monkeypatch.setenv("HEADROOM_LABELS", "garbage")
    monkeypatch.setenv("HEADROOM_HOST_LABEL", "h")
    env: dict[str, str] = {}
    wrap_mod._apply_attribution_header_env(env)
    assert env["ANTHROPIC_CUSTOM_HEADERS"] == "X-Headroom-Host: h"


def test_proxy_decodes_what_wrap_sends(monkeypatch):
    from headroom.proxy.attribution import AttributionRequest, resolve_attribution

    monkeypatch.setattr(wrap_mod.socket, "gethostname", lambda: "Büro-Mac.local")
    monkeypatch.delenv("HEADROOM_HOST_LABEL", raising=False)
    sent = wrap_mod._host_name_for_header()
    labels = resolve_attribution(AttributionRequest(headers={"x-headroom-host": sent}), ["host"])
    assert labels == {"host": "büro-mac"}
