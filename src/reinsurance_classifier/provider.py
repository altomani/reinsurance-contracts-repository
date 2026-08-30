"""First-party-only OpenRouter model calls through Pydantic AI."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from pydantic import Field

from .extraction import EvidencePack
from .models import (
    ClassificationDecision,
    StrictModel,
    validate_evidence_lines,
    validate_evidence_quotes,
)


@dataclass(frozen=True)
class ModelRoute:
    alias: str
    model_id: str
    provider_slug: str


ROUTES: tuple[ModelRoute, ...] = (
    ModelRoute("qwen", "qwen/qwen3.8-flash", "alibaba"),
    ModelRoute(
        "deepseek", "deepseek/deepseek-v4-flash-vision-exp", "deepseek"
    ),
    ModelRoute("glm", "z-ai/glm-5.3-flash", "z-ai"),
)
ROUTE_BY_NAME = {
    name: route
    for route in ROUTES
    for name in (route.alias, route.model_id)
}


class ProviderCallResult(StrictModel):
    model_id: str
    required_provider: str
    downstream_provider: str | None = None
    decision: ClassificationDecision
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cost_usd: float | None = Field(default=None, ge=0)
    latency_seconds: float = Field(ge=0)
    retry_count: int = Field(default=0, ge=0)
    evidence_lines_valid: bool = True
    evidence_quotes_valid: bool = True
    evidence_validation_errors: list[str] = Field(default_factory=list)


class ClassifierBackend(Protocol):
    async def classify(
        self,
        evidence_pack: EvidencePack,
        route: ModelRoute,
        *,
        prompt_text: str,
        max_output_tokens: int,
    ) -> ProviderCallResult: ...


class OpenRouterClassifier:
    def __init__(self, api_key: str, *, app_title: str) -> None:
        self.api_key = api_key
        self.app_title = app_title

    async def classify(
        self,
        evidence_pack: EvidencePack,
        route: ModelRoute,
        *,
        prompt_text: str,
        max_output_tokens: int,
    ) -> ProviderCallResult:
        # Keep the paid provider dependency out of local extraction/dry-run paths.
        from pydantic_ai import Agent, PromptedOutput
        from pydantic_ai.messages import ModelResponse
        from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
        from pydantic_ai.providers.openrouter import OpenRouterProvider

        model = OpenRouterModel(
            route.model_id,
            provider=OpenRouterProvider(
                api_key=self.api_key,
                app_title=self.app_title,
            ),
        )
        settings = OpenRouterModelSettings(
            temperature=0,
            max_tokens=max_output_tokens,
            thinking="minimal" if route.provider_slug == "z-ai" else False,
            openrouter_provider={
                "only": [route.provider_slug],
                "allow_fallbacks": False,
            },
            openrouter_usage={"include": True},
        )
        agent = Agent(
            model,
            output_type=PromptedOutput(ClassificationDecision),
            instructions=prompt_text,
            model_settings=settings,
            retries=2,
        )
        started = time.perf_counter()
        result = await agent.run(evidence_pack.text)
        latency = time.perf_counter() - started
        validation_errors: list[str] = []
        lines_valid = True
        quotes_valid = True
        try:
            validate_evidence_lines(
                result.output, set(evidence_pack.selected_line_numbers)
            )
        except ValueError as exc:
            lines_valid = False
            validation_errors.append(str(exc))
        try:
            validate_evidence_quotes(result.output, evidence_pack.text)
        except ValueError as exc:
            quotes_valid = False
            validation_errors.append(str(exc))

        model_responses = [
            message
            for message in result.all_messages()
            if isinstance(message, ModelResponse)
        ]
        costs: list[float] = []
        downstream: str | None = None
        for response in model_responses:
            details = response.provider_details or {}
            cost = details.get("cost")
            if isinstance(cost, int | float):
                costs.append(float(cost))
            candidate = details.get("downstream_provider")
            if isinstance(candidate, str):
                downstream = candidate
        usage = result.usage
        return ProviderCallResult(
            model_id=route.model_id,
            required_provider=route.provider_slug,
            downstream_provider=downstream,
            decision=result.output,
            input_tokens=usage.input_tokens or 0,
            output_tokens=usage.output_tokens or 0,
            cost_usd=sum(costs) if costs else None,
            latency_seconds=latency,
            retry_count=max(0, len(model_responses) - 1),
            evidence_lines_valid=lines_valid,
            evidence_quotes_valid=quotes_valid,
            evidence_validation_errors=validation_errors,
        )


def resolve_routes(names: list[str] | None) -> tuple[ModelRoute, ...]:
    if not names:
        return ROUTES
    routes: list[ModelRoute] = []
    for name in names:
        try:
            route = ROUTE_BY_NAME[name]
        except KeyError as exc:
            available = ", ".join(sorted(ROUTE_BY_NAME))
            raise ValueError(f"unknown model {name!r}; choose one of: {available}") from exc
        if route not in routes:
            routes.append(route)
    return tuple(routes)
