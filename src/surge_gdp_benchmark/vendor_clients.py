"""Direct vendor API calls and local usage/cost normalization."""

from __future__ import annotations

import dataclasses
from typing import Literal, cast

from surge_gdp_benchmark.common import JsonValue, ModelKey, Sample

Provider = Literal["openai", "anthropic", "google"]


@dataclasses.dataclass(frozen=True)
class PricingTier:
    label: str
    input_cost_per_million: float
    output_cost_per_million: float
    max_input_tokens: int | None = None

    def matches(self, input_tokens: int) -> bool:
        return self.max_input_tokens is None or input_tokens <= self.max_input_tokens


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    key: ModelKey
    label: str
    provider: Provider
    model_name: str
    pricing_tiers: tuple[PricingTier, ...]
    output_tokens_include_reasoning: bool = True
    input_tokens_include_cache_segments: bool = True
    max_tokens: int | None = None
    timeout_s: int = 900

    def pricing_for_input_tokens(self, input_tokens: int) -> PricingTier:
        for tier in self.pricing_tiers:
            if tier.matches(input_tokens):
                return tier
        raise ValueError(f"No pricing tier for {self.model_name} with {input_tokens} input tokens")


MODEL_SPECS: dict[ModelKey, ModelSpec] = {
    # Prices:
    # https://openai.com/api/pricing/
    # Long-context tier and reasoning token billing:
    # https://developers.openai.com/api/docs/models/gpt-5.5
    # https://developers.openai.com/api/docs/guides/reasoning
    "gpt_5_5": ModelSpec(
        key="gpt_5_5",
        label="GPT 5.5 xHigh Reasoning",
        provider="openai",
        model_name="gpt-5.5",
        pricing_tiers=(
            PricingTier(
                label="standard_under_272k_input_tokens",
                max_input_tokens=272_000,
                input_cost_per_million=5.0,
                output_cost_per_million=30.0,
            ),
            PricingTier(
                label="long_context_over_272k_input_tokens",
                input_cost_per_million=10.0,
                output_cost_per_million=45.0,
            ),
        ),
    ),
    # Prices:
    # https://platform.claude.com/docs/en/about-claude/pricing
    # Thinking token accounting:
    # https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking
    # Prompt-cache usage fields:
    # https://platform.claude.com/docs/en/build-with-claude/prompt-caching
    "opus_4_8": ModelSpec(
        key="opus_4_8",
        label="Claude Opus 4.8 Adaptive Max",
        provider="anthropic",
        model_name="claude-opus-4-8",
        pricing_tiers=(
            PricingTier(
                label="standard",
                input_cost_per_million=5.0,
                output_cost_per_million=25.0,
            ),
        ),
        input_tokens_include_cache_segments=False,
        max_tokens=128_000,
        timeout_s=1_800,
    ),
    # Prices:
    # https://ai.google.dev/gemini-api/docs/pricing
    # Token accounting and thinking-token billing:
    # https://ai.google.dev/gemini-api/docs/tokens
    # https://ai.google.dev/gemini-api/docs/thinking
    "gemini_3_1": ModelSpec(
        key="gemini_3_1",
        label="Gemini 3.1 Pro",
        provider="google",
        model_name="gemini-3.1-pro-preview",
        pricing_tiers=(
            PricingTier(
                label="standard_up_to_200k_input_tokens",
                max_input_tokens=200_000,
                input_cost_per_million=2.0,
                output_cost_per_million=12.0,
            ),
            PricingTier(
                label="long_context_over_200k_input_tokens",
                input_cost_per_million=4.0,
                output_cost_per_million=18.0,
            ),
        ),
        output_tokens_include_reasoning=False,
    ),
}


@dataclasses.dataclass(frozen=True)
class TargetResponse:
    answer: str
    requested_model: str
    resolved_model: str
    finish_reason: str | None
    usage: dict[str, JsonValue]
    raw_response: JsonValue
    methodology: dict[str, JsonValue]
    latency_s: float


@dataclasses.dataclass(frozen=True)
class UploadedPdf:
    provider: Provider
    name: str
    uri: str
    mime_type: str
    source_path: str
    provider_file: object | None = dataclasses.field(default=None, repr=False, compare=False)


def require_dict(value: object, context: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise TypeError(f"Expected {context} to be a dict, got {type(value).__name__}")
    return cast(dict[str, JsonValue], value)


def require_key(mapping: dict[str, JsonValue], key: str, context: str) -> JsonValue:
    if key not in mapping:
        raise KeyError(f"Missing {context}.{key}")
    return mapping[key]


def require_int(mapping: dict[str, JsonValue], key: str, context: str) -> int:
    value = require_key(mapping, key, context)
    if isinstance(value, bool):
        raise TypeError(f"Expected {context}.{key} to be an int, got bool")
    if not isinstance(value, int):
        raise TypeError(f"Expected {context}.{key} to be an int, got {type(value).__name__}")
    return value


# Token accounting sources:
# - OpenAI `output_tokens` includes reasoning tokens billed as output:
#   https://developers.openai.com/api/docs/guides/reasoning
# - OpenAI cached input tokens are reported in `input_tokens_details.cached_tokens`:
#   https://platform.openai.com/docs/guides/prompt-caching
# - Anthropic `output_tokens` is the inclusive billing total and `thinking_tokens` is a
#   breakdown of that total:
#   https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking
# - Anthropic reports uncached input, cache reads, and cache writes as separate fields:
#   https://platform.claude.com/docs/en/build-with-claude/prompt-caching
# - Gemini bills response output as candidates plus thinking tokens:
#   https://ai.google.dev/gemini-api/docs/thinking
# - Gemini usage metadata reports prompt, cached, candidate, and thought token counts:
#   https://ai.google.dev/gemini-api/docs/tokens
# Benchmark pricing policy: do not apply provider cache discounts or cache-write premiums.
# Any provider-reported cache reads/writes are charged as standard input tokens.
def cost_payload(
    model_key: ModelKey, usage: dict[str, JsonValue], latency_s: float
) -> dict[str, JsonValue]:
    spec = MODEL_SPECS[model_key]
    input_tokens = require_int(usage, "input_tokens", "normalized usage")
    output_tokens = require_int(usage, "output_tokens", "normalized usage")
    reasoning_tokens = require_int(usage, "reasoning_tokens", "normalized usage")
    cached_tokens = require_int(usage, "cached_tokens", "normalized usage")
    cache_creation_tokens = require_int(
        usage, "cache_creation_input_tokens", "normalized usage"
    )
    billable_input_tokens = input_tokens
    if not spec.input_tokens_include_cache_segments:
        billable_input_tokens += cached_tokens + cache_creation_tokens
    tier = spec.pricing_for_input_tokens(billable_input_tokens)
    billable_output_tokens = output_tokens
    if not spec.output_tokens_include_reasoning:
        billable_output_tokens += reasoning_tokens
    input_cost = billable_input_tokens * tier.input_cost_per_million / 1_000_000
    output_cost = billable_output_tokens * tier.output_cost_per_million / 1_000_000
    reasoning_cost = reasoning_tokens * tier.output_cost_per_million / 1_000_000
    total_cost = input_cost + output_cost
    return {
        "model": spec.model_name,
        "provider": spec.provider,
        "input_tokens": input_tokens,
        "cached_tokens": cached_tokens,
        "cache_creation_input_tokens": cache_creation_tokens,
        "billable_input_tokens": billable_input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "billable_output_tokens": billable_output_tokens,
        "input_cost_per_million": tier.input_cost_per_million,
        "output_cost_per_million": tier.output_cost_per_million,
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "reasoning_cost_usd": reasoning_cost,
        "reasoning_cost_policy": "included_in_output_cost",
        "total_cost_usd": total_cost,
        "pricing_tier": tier.label,
        "pricing_policy": "cache_tokens_charged_as_standard_input",
        "latency_s": latency_s,
    }


def target_artifact(
    *,
    sample: Sample,
    arm: str,
    model_key: ModelKey,
    response: TargetResponse,
) -> dict[str, JsonValue]:
    spec = MODEL_SPECS[model_key]
    cost = cost_payload(model_key, response.usage, response.latency_s)
    return {
        "sample_idx": sample.sample_idx,
        "task_id": sample.task_id,
        "arm": arm,
        "model_key": model_key,
        "model_label": spec.label,
        "requested_model": response.requested_model,
        "resolved_model": response.resolved_model,
        "answer_chars": len(response.answer),
        "latency_s": response.latency_s,
        "finish_reason": response.finish_reason,
        "usage": response.usage,
        "cost": cost,
        "raw_response": response.raw_response,
        "methodology": response.methodology,
        "generation_config": generation_config_json(model_key),
    }


def generation_config_json(model_key: ModelKey) -> dict[str, JsonValue]:
    spec = MODEL_SPECS[model_key]
    if model_key == "gpt_5_5":
        return {
            "temperature": None,
            "reasoning": {"effort": "xhigh"},
            "timeout_s": spec.timeout_s,
        }
    if model_key == "opus_4_8":
        return {
            "temperature": None,
            "max_tokens": spec.max_tokens,
            "timeout_s": spec.timeout_s,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "max"},
        }
    return {
        "temperature": None,
        "timeout_s": spec.timeout_s,
        "thinking_config": {"thinkingLevel": "HIGH"},
    }


def call_target_model(
    *,
    sample: Sample,
    model_key: ModelKey,
    prompt_text: str,
    uploaded_pdf: UploadedPdf | None = None,
) -> TargetResponse:
    if model_key == "gpt_5_5":
        from surge_gdp_benchmark.clients.openai_client import call_openai

        return call_openai(sample=sample, prompt_text=prompt_text)
    if model_key == "opus_4_8":
        from surge_gdp_benchmark.clients.anthropic_client import call_anthropic

        return call_anthropic(sample=sample, prompt_text=prompt_text)
    from surge_gdp_benchmark.clients.google_client import call_google

    return call_google(sample=sample, prompt_text=prompt_text, uploaded_pdf=uploaded_pdf)
