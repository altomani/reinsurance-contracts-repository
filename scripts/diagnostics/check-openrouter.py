"""Low-cost connectivity check for the configured OpenRouter models."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent, PromptedOutput
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
from pydantic_ai.providers.openrouter import OpenRouterProvider


@dataclass(frozen=True)
class ModelRoute:
    model: str
    provider: str


class SmokeTestResult(BaseModel):
    status: str
    checksum: int


ROUTES = (
    ModelRoute("qwen/qwen3.8-flash", "alibaba"),
    ModelRoute("deepseek/deepseek-v4-flash-vision-exp", "deepseek"),
    ModelRoute("z-ai/glm-5.3-flash", "z-ai"),
)


def check_route(route: ModelRoute, api_key: str) -> None:
    model = OpenRouterModel(
        route.model,
        provider=OpenRouterProvider(
            api_key=api_key,
            app_title="EDGAR reinsurance contract classifier",
        ),
    )
    settings = OpenRouterModelSettings(
        temperature=0,
        max_tokens=128,
        # GLM 5.3 requires reasoning; Alibaba rejects forced output tools when
        # reasoning is enabled. Keep the smoke test compatible with both.
        thinking="minimal" if route.provider == "z-ai" else False,
        openrouter_provider={
            "only": [route.provider],
            "allow_fallbacks": False,
        },
        openrouter_usage={"include": True},
    )
    agent = Agent(
        model,
        # Prompted JSON avoids provider-specific forced-tool restrictions while
        # retaining Pydantic validation and automatic retry on malformed output.
        output_type=PromptedOutput(SmokeTestResult),
        instructions=(
            "This is a connectivity test. Return status='ok' and checksum=7. "
            "Do not add commentary."
        ),
        model_settings=settings,
    )
    result = agent.run_sync("Return the requested connectivity-test result.")
    if result.output != SmokeTestResult(status="ok", checksum=7):
        raise RuntimeError(f"Unexpected response from {route.model}: {result.output!r}")

    details = result.response.provider_details or {}
    usage = result.usage
    cost = details.get("cost")
    cost_text = f", ${cost:.8f}" if isinstance(cost, int | float) else ""
    print(
        f"OK {route.model} via {details.get('downstream_provider', 'unknown')} "
        f"({usage.input_tokens} input, {usage.output_tokens} output tokens{cost_text})"
    )


def main() -> None:
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set")

    for route in ROUTES:
        check_route(route, api_key)


if __name__ == "__main__":
    main()
