"""Omniroute provider wiring — local self-hosted OpenAI-compatible router.

Mirrors test_provider_nvidia.py: verifies the adapter, the ai.py/api_core.py/
model_routing.py registries, the discovery belong-rule (accept everything,
like LM Studio — Omniroute's model list is already curated by the operator
on the Omniroute side), and that an explicit omniroute route is never
hijacked by model NAME (its ids embed other vendors' names as path segments,
e.g. "openrouter/openai/gpt-5.6-luna").

Also covers the local-provider connect/test flow (required base_url AND
required token — a combination no existing provider needed before this one),
following the upsert_auth_provider / test_auth_provider patterns in
test_api_core_auth.py.
"""

from __future__ import annotations

import httpx
import pytest

import forven.ai as ai
from forven import api_core as ac
from forven import model_routing as mr
from forven.auth import store as auth_store
from forven.agents.providers import (
    OmniRouteProvider,
    ToolCallProvider,
    get_provider,
)


def test_factory_resolves_omniroute():
    inst = get_provider("omniroute")
    assert isinstance(inst, OmniRouteProvider)
    assert issubclass(OmniRouteProvider, ToolCallProvider)


def test_omniroute_requires_base_url_at_call_time():
    inst = OmniRouteProvider()
    with pytest.raises(ValueError):
        inst._get_base_url()


def test_prefix_and_routing_defaults():
    assert "omniroute" not in ai.ENDPOINTS  # no fixed endpoint — base_url is per-profile
    assert "omniroute" in ai._KNOWN_PROVIDER_PREFIXES
    assert "omniroute" in ai._PROVIDER_PASSTHROUGH
    assert "omniroute" in mr._SUPPORTED_PROVIDERS
    assert mr.get_default_model_for_provider("omniroute")


def test_omniroute_persisted_by_auth_store():
    # The auth store keeps its OWN allowlist; load_auth() silently DROPS any
    # profile whose provider isn't in it. If omniroute is missing here, the
    # token+base_url save (POST 200, "Connected" flash) but are stripped on
    # the next read, so the provider reverts to "Not connected".
    assert "omniroute" in auth_store._SUPPORTED_AUTH_PROVIDERS
    assert auth_store._ENV_ACCESS_TOKEN_KEYS["omniroute"] == ("OMNIROUTE_API_KEY",)
    assert auth_store._ENV_BASE_URL_KEYS["omniroute"] == ("OMNIROUTE_BASE_URL",)


def test_auth_store_allowlist_covers_every_connectable_provider():
    api_core_providers = set(ac._SUPPORTED_AUTH_PROVIDERS)
    store_providers = set(auth_store._SUPPORTED_AUTH_PROVIDERS)
    missing = api_core_providers - store_providers
    assert not missing, f"providers connectable but not persistable: {sorted(missing)}"


def test_registered_in_api_core():
    assert "omniroute" in ac._SUPPORTED_AUTH_PROVIDERS
    assert ac._AUTH_PROVIDER_ENV_VARS["omniroute"] == "OMNIROUTE_API_KEY"
    assert ac._MODEL_PROVIDER_DISPLAY_NAMES["omniroute"] == "Omniroute"
    assert ac._LOCAL_PROVIDER_DEFAULT_BASE_URLS["omniroute"] == ""
    # No curated fallback catalog — Omniroute's models are entirely
    # dependent on the operator's own connector config.
    assert not [m for m in ac._AGENT_MODEL_CATALOG if m["provider"] == "omniroute"]


def test_discovery_belong_rule_accepts_everything():
    # Like LM Studio: the operator already curates their connector list on
    # the Omniroute side, so Forven doesn't second-guess the ids it returns.
    assert ac._discovery_model_should_belong("omniroute", "openrouter/openai/gpt-5.6-luna")
    assert ac._discovery_model_should_belong("omniroute", "claude/claude-opus-4-7")
    assert ac._discovery_model_should_belong("omniroute", "anything-goes")
    assert not ac._discovery_model_should_belong("omniroute", "")


def test_explicit_omniroute_not_hijacked_by_model_name():
    # Omniroute ids embed other vendors' names as path segments
    # ("openrouter/openai/...", "claude/..."). An EXPLICIT omniroute
    # selection must pass through unchanged, never re-routed by the legacy
    # looks-like-openai/zai/minimax model-name heuristics.
    assert ai.normalize_provider_and_model("omniroute", "openrouter/openai/gpt-5.6-luna") == (
        "omniroute", "openrouter/openai/gpt-5.6-luna",
    )
    assert ai.normalize_provider_and_model("omniroute", "claude/claude-opus-4-7") == (
        "omniroute", "claude/claude-opus-4-7",
    )


def test_omniroute_prefix_parsing():
    assert ai._split_provider_model_prefix("omniroute:openrouter/openai/gpt-5.6-luna") == (
        "omniroute", "openrouter/openai/gpt-5.6-luna",
    )


def test_upsert_auth_provider_omniroute_requires_base_url(monkeypatch):
    monkeypatch.setattr(ac, "get_profile", lambda provider: None)
    monkeypatch.setattr(ac, "upsert_profile", lambda provider, profile: None)

    with pytest.raises(Exception) as exc_info:
        ac.upsert_auth_provider(
            "omniroute",
            ac.AuthProviderProfileBody(api_key="sk-test"),
        )
    assert "base_url" in str(exc_info.value)


def test_upsert_auth_provider_omniroute_requires_token(monkeypatch):
    monkeypatch.setattr(ac, "get_profile", lambda provider: None)
    monkeypatch.setattr(ac, "upsert_profile", lambda provider, profile: None)

    with pytest.raises(Exception) as exc_info:
        ac.upsert_auth_provider(
            "omniroute",
            ac.AuthProviderProfileBody(base_url="http://127.0.0.1:8787"),
        )
    assert "access token" in str(exc_info.value)


def test_upsert_auth_provider_omniroute_accepts_both(monkeypatch):
    saved_profiles: dict[str, dict] = {}

    def _fake_get_profile(provider: str) -> dict | None:
        return saved_profiles.get(provider)

    def _fake_upsert_profile(provider: str, profile: dict) -> None:
        saved_profiles[provider] = dict(profile)

    monkeypatch.setattr(ac, "get_profile", _fake_get_profile)
    monkeypatch.setattr(ac, "upsert_profile", _fake_upsert_profile)

    result = ac.upsert_auth_provider(
        "omniroute",
        ac.AuthProviderProfileBody(api_key="sk-test", base_url="http://127.0.0.1:8787"),
    )

    assert result == {"ok": True, "provider": "omniroute"}
    assert saved_profiles["omniroute"]["access"] == "sk-test"
    assert saved_profiles["omniroute"]["base_url"] == "http://127.0.0.1:8787"


def test_omniroute_test_provider_calls_local_models_endpoint(monkeypatch):
    profile = {"base_url": "http://127.0.0.1:8787", "access": "sk-test"}

    class _FakeClient:
        def __init__(self, timeout: float | None = None):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url: str, headers: dict | None = None):
            assert url == "http://127.0.0.1:8787/v1/models"
            assert headers == {"Authorization": "Bearer sk-test"}
            return httpx.Response(
                200,
                json={"data": [{"id": "openrouter/openai/gpt-5.6-luna"}, {"id": "claude/claude-opus-4-7"}]},
                request=httpx.Request("GET", url),
            )

    monkeypatch.setattr(ac, "get_profile", lambda provider: profile if provider == "omniroute" else None)
    monkeypatch.setattr(ac.httpx, "Client", _FakeClient)

    result = ac.test_auth_provider("omniroute")

    assert result["ok"] is True
    assert result["provider"] == "omniroute"
    assert "2 models discovered" in str(result["message"])


def test_discover_provider_models_omniroute(monkeypatch):
    from forven.providers import discovery

    profile = {"base_url": "http://127.0.0.1:8787", "access": "sk-test"}

    class _FakeClient:
        def __init__(self, timeout: float | None = None):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url: str, headers: dict | None = None):
            assert url == "http://127.0.0.1:8787/v1/models"
            assert headers == {"Authorization": "Bearer sk-test"}
            return httpx.Response(
                200,
                json={"data": [{"id": "openrouter/openai/gpt-5.6-luna"}]},
                request=httpx.Request("GET", url),
            )

    monkeypatch.setattr(discovery, "get_profile", lambda provider: profile if provider == "omniroute" else None)
    monkeypatch.setattr(discovery.httpx, "Client", _FakeClient)
    discovery._AGENT_MODEL_LIST_CACHE.pop("omniroute", None)

    models, error = discovery._discover_provider_models("omniroute", force_refresh=True)

    assert error is None
    assert any(m["model_id"] == "openrouter/openai/gpt-5.6-luna" for m in models)
