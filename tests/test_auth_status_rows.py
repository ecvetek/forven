"""Regression: the Diagnostics/CLI auth status table was frozen on a hardcoded
``["openai", "minimax", "lmstudio"]`` provider list — every provider added
since (Z.AI, OpenRouter, Anthropic, ..., Omniroute) never appeared here even
when fully connected, so ``check_auth_providers()`` kept reporting "no
provider has a valid token" regardless of what was actually configured.
"""

from __future__ import annotations

from unittest.mock import patch

from forven.auth import store as auth_store
from forven import diagnostics


def test_get_status_rows_covers_every_supported_provider():
    with patch("forven.auth.store.load_auth", return_value={"version": 1, "profiles": {}}):
        rows = auth_store.get_status_rows()
    providers = {row[0] for row in rows}
    assert providers == set(auth_store._SUPPORTED_AUTH_PROVIDERS)
    assert "omniroute" in providers


def test_get_status_rows_is_stably_ordered():
    with patch("forven.auth.store.load_auth", return_value={"version": 1, "profiles": {}}):
        rows_a = auth_store.get_status_rows()
        rows_b = auth_store.get_status_rows()
    assert [r[0] for r in rows_a] == [r[0] for r in rows_b] == sorted(auth_store._SUPPORTED_AUTH_PROVIDERS)


def test_get_status_rows_connected_token_provider_with_base_url_shows_active():
    fake_store = {
        "version": 1,
        "profiles": {
            "omniroute:default": {
                "provider": "omniroute",
                "access": "sk-test",
                "base_url": "http://10.0.0.50:20128/v1",
            },
        },
    }
    with patch("forven.auth.store.load_auth", return_value=fake_store):
        rows = auth_store.get_status_rows()
    row = next(r for r in rows if r[0] == "omniroute")
    assert "Active" in row[1]
    assert row[2] == "http://10.0.0.50:20128/v1"


def test_get_status_rows_unconfigured_provider_shows_not_configured():
    with patch("forven.auth.store.load_auth", return_value={"version": 1, "profiles": {}}):
        rows = auth_store.get_status_rows()
    row = next(r for r in rows if r[0] == "openrouter")
    assert "Not configured" in row[1]


def test_get_status_rows_lmstudio_behavior_unchanged():
    fake_store = {
        "version": 1,
        "profiles": {
            "lmstudio:default": {"provider": "lmstudio", "base_url": "http://127.0.0.1:1234"},
        },
    }
    with patch("forven.auth.store.load_auth", return_value=fake_store):
        rows = auth_store.get_status_rows()
    row = next(r for r in rows if r[0] == "lmstudio")
    assert "Active" in row[1]
    assert row[2] == "http://127.0.0.1:1234"

    with patch("forven.auth.store.load_auth", return_value={"version": 1, "profiles": {}}):
        rows = auth_store.get_status_rows()
    row = next(r for r in rows if r[0] == "lmstudio")
    assert "Not configured" in row[1]


def test_diagnostics_check_auth_providers_passes_with_only_omniroute_connected():
    fake_rows = [
        (p, "[red]Not configured[/red]", "-")
        for p in sorted(auth_store._SUPPORTED_AUTH_PROVIDERS)
        if p != "omniroute"
    ] + [("omniroute", "[green]Active[/green]", "http://10.0.0.50:20128/v1")]

    with patch("forven.auth.store.get_status_rows", return_value=fake_rows):
        result = diagnostics.check_auth_providers()

    assert result.status == diagnostics.PASS
    assert "omniroute" in result.summary
