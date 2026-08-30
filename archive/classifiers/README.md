# Historical classifier archive

This directory preserves superseded classifier implementations and their generated outputs for reproducibility and provenance. Nothing here is part of the maintained classifier workflow.

## Contents

| Path | Historical model or purpose |
| --- | --- |
| `scripts/classify-contracts.py` | OpenAI `gpt-4o-mini` classifier |
| `scripts/classify-contracts-gemini.py` | Google `gemini-2.0-flash` classifier through its OpenAI-compatible endpoint |
| `scripts/classify-contracts-openrouter.py` | OpenRouter batch classifier used for the later three-model run |
| `scripts/join-results.py` | Join utility for historical per-model outputs |
| `results/gpt-4o-mini/` | Yearly `gpt-4o-mini` classifications |
| `results/gemini-2.0-flash/` | Yearly `gemini-2.0-flash` classifications |
| `results/qwen3-235b-a22b-2507/` | Yearly Qwen classifications |
| `results/gpt-oss-120b/` | Yearly GPT-OSS classifications |
| `results/gemini-2.5-flash-lite/` | Yearly Gemini Flash Lite classifications |
| `results/joined/` | Historical joined model output |

The archived scripts predate the package in `src/reinsurance_classifier/`. They use separate dependencies such as pandas, OpenAI-compatible clients, tiktoken, html2text, and provider-specific API keys. Their hard-coded year ranges, paths, prompts, and model names should be reviewed before any reproduction attempt.

For current classification, testing, and output formats, use the repository root README and the `reinsurance-classifier` command.
