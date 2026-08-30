from __future__ import annotations

import asyncio

from conftest import make_decision
from reinsurance_classifier.extraction import EvidencePack
from reinsurance_classifier.provider import OpenRouterClassifier, ROUTES


class _Usage:
    input_tokens = 12
    output_tokens = 34


class _Result:
    output = make_decision()
    usage = _Usage()

    @staticmethod
    def all_messages():
        return []


def test_openrouter_settings_lock_first_party_and_disable_fallbacks(monkeypatch) -> None:
    captured: list[dict] = []

    class FakeProvider:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeModel:
        def __init__(self, model_id, *, provider):
            self.model_id = model_id
            self.provider = provider

    def fake_settings(**kwargs):
        captured.append(kwargs)
        return kwargs

    class FakeAgent:
        def __init__(self, model, **kwargs):
            self.model = model
            self.kwargs = kwargs

        async def run(self, text):
            return _Result()

    monkeypatch.setattr("pydantic_ai.Agent", FakeAgent)
    monkeypatch.setattr("pydantic_ai.models.openrouter.OpenRouterModel", FakeModel)
    monkeypatch.setattr(
        "pydantic_ai.models.openrouter.OpenRouterModelSettings", fake_settings
    )
    monkeypatch.setattr(
        "pydantic_ai.providers.openrouter.OpenRouterProvider", FakeProvider
    )
    pack = EvidencePack(
        text="[L000001] operative provision",
        selected_line_numbers=frozenset({1}),
        selected_ranges=((1, 1),),
        categories_found=(),
        truncated=False,
        normalized_chars=28,
        estimated_input_tokens=10,
    )
    classifier = OpenRouterClassifier("test-key", app_title="test")

    asyncio.run(
        classifier.classify(
            pack,
            ROUTES[0],
            prompt_text="prompt",
            max_output_tokens=500,
        )
    )
    asyncio.run(
        classifier.classify(
            pack,
            ROUTES[2],
            prompt_text="prompt",
            max_output_tokens=500,
        )
    )

    assert captured[0]["openrouter_provider"] == {
        "only": ["alibaba"],
        "allow_fallbacks": False,
    }
    assert captured[0]["thinking"] is False
    assert captured[1]["openrouter_provider"] == {
        "only": ["z-ai"],
        "allow_fallbacks": False,
    }
    assert captured[1]["thinking"] == "minimal"


def test_invalid_evidence_is_audited_without_discarding_paid_decision(monkeypatch) -> None:
    class FakeProvider:
        def __init__(self, **kwargs):
            pass

    class FakeModel:
        def __init__(self, model_id, *, provider):
            pass

    class FakeAgent:
        def __init__(self, model, **kwargs):
            pass

        async def run(self, text):
            return _Result()

    monkeypatch.setattr("pydantic_ai.Agent", FakeAgent)
    monkeypatch.setattr("pydantic_ai.models.openrouter.OpenRouterModel", FakeModel)
    monkeypatch.setattr(
        "pydantic_ai.models.openrouter.OpenRouterModelSettings", lambda **kwargs: kwargs
    )
    monkeypatch.setattr(
        "pydantic_ai.providers.openrouter.OpenRouterProvider", FakeProvider
    )
    pack = EvidencePack(
        text="[L000002] different text",
        selected_line_numbers=frozenset({2}),
        selected_ranges=((2, 2),),
        categories_found=(),
        truncated=False,
        normalized_chars=22,
        estimated_input_tokens=8,
    )
    result = asyncio.run(
        OpenRouterClassifier("test-key", app_title="test").classify(
            pack, ROUTES[0], prompt_text="prompt", max_output_tokens=500
        )
    )
    assert result.evidence_lines_valid is False
    assert result.evidence_quotes_valid is False
    assert result.evidence_validation_errors
