"""OpenRouter judge calls and usage normalization."""

from __future__ import annotations

import dataclasses
import os
import time

import requests

from common import JsonValue, Sample, jsonable
from vendor_clients import require_dict, require_int, require_key

OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_DOCS: dict[str, str] = {
    "chat_completions": "https://openrouter.ai/docs/api-reference/chat-completion",
    "reasoning": "https://openrouter.ai/docs/guides/best-practices/reasoning-tokens",
}
OPENROUTER_JUDGE_PROVIDER_ORDER = ["Fireworks"]
OPENROUTER_REASONING_CONFIG: dict[str, JsonValue] = {"enabled": True, "exclude": True}
DEFAULT_OPENROUTER_JUDGE_MODEL = "deepseek/deepseek-v4-pro"
DEFAULT_OPENROUTER_TIMEOUT_S = 900
DEFAULT_OPENROUTER_MAX_TOKENS = 16384


@dataclasses.dataclass(frozen=True)
class OpenRouterJudgeConfig:
    """Configuration for an OpenRouter-backed rubric judge call."""

    model: str
    timeout_s: int
    max_tokens: int


@dataclasses.dataclass(frozen=True)
class OpenRouterJudgeResponse:
    """Normalized OpenRouter judge response plus benchmark metadata."""

    output_text: str
    requested_model: str
    resolved_model: str
    finish_reason: str
    usage: dict[str, JsonValue]
    raw_response: JsonValue
    methodology: dict[str, JsonValue]
    latency_s: float
    cost: dict[str, JsonValue]


def openrouter_judge_config_from_env() -> OpenRouterJudgeConfig:
    """Read the judge model settings from environment variables."""

    return OpenRouterJudgeConfig(
        model=os.getenv("GDP_PDF_OPENROUTER_JUDGE_MODEL", DEFAULT_OPENROUTER_JUDGE_MODEL),
        timeout_s=env_int("GDP_PDF_OPENROUTER_TIMEOUT_S", DEFAULT_OPENROUTER_TIMEOUT_S),
        max_tokens=env_int("GDP_PDF_OPENROUTER_MAX_TOKENS", DEFAULT_OPENROUTER_MAX_TOKENS),
    )


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def openrouter_api_key() -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY must be set for OpenRouter judge calls")
    return api_key


def openrouter_judge_methodology(
    sample: Sample, config: OpenRouterJudgeConfig
) -> dict[str, JsonValue]:
    return {
        "provider_api": "OpenRouter Chat Completions API",
        "requested_model": config.model,
        "input_method": "text_only_message_content",
        "source_pdf_included": False,
        "pdf_handling_policy": "no_pdf_or_parser_context_rubric_text_only_judging",
        "provider_routing": {
            "order": OPENROUTER_JUDGE_PROVIDER_ORDER,
            "allow_fallbacks": False,
            "require_parameters": True,
        },
        "provider_routing_policy": (
            "pin_single_structured_output_capable_provider_for_reproducible_judging"
        ),
        "response_format": "json_object",
        "max_tokens": config.max_tokens,
        "reasoning": OPENROUTER_REASONING_CONFIG,
        "sampling_config": "provider_defaults",
        "determinism_policy": "temperature_and_sampling_parameters_intentionally_unset",
        "api_docs": OPENROUTER_API_DOCS,
    }


def call_openrouter_judge(
    *,
    sample: Sample,
    judge_prompt: str,
    config: OpenRouterJudgeConfig | None = None,
) -> OpenRouterJudgeResponse:
    judge_config = config or openrouter_judge_config_from_env()
    payload = openrouter_judge_payload(sample, judge_prompt, judge_config)
    headers = {
        "Authorization": f"Bearer {openrouter_api_key()}",
        "Content-Type": "application/json",
    }
    started = time.perf_counter()
    response = requests.post(
        OPENROUTER_CHAT_COMPLETIONS_URL,
        headers=headers,
        json=payload,
        timeout=judge_config.timeout_s,
    )
    latency_s = time.perf_counter() - started
    if not response.ok:
        raise requests.HTTPError(
            f"OpenRouter judge call failed with status {response.status_code}: {response.text}",
            response=response,
        )
    raw = jsonable(response.json())
    raw_dict = require_dict(raw, "OpenRouter response")
    text = openrouter_message_text(raw_dict)
    finish_reason = openrouter_finish_reason(raw_dict)
    if finish_reason != "stop":
        raise ValueError(
            f"OpenRouter judge finish_reason must be 'stop', got {finish_reason!r}"
        )
    usage = normalize_openrouter_usage(raw_dict)
    return OpenRouterJudgeResponse(
        output_text=text,
        requested_model=judge_config.model,
        resolved_model=openrouter_resolved_model(raw_dict),
        finish_reason=finish_reason,
        usage=usage,
        raw_response=raw,
        methodology=openrouter_judge_methodology(sample, judge_config),
        latency_s=latency_s,
        cost=openrouter_judge_cost(judge_config.model, usage, latency_s),
    )


def openrouter_judge_payload(
    sample: Sample, judge_prompt: str, config: OpenRouterJudgeConfig
) -> dict[str, object]:
    return {
        "model": config.model,
        "messages": [
            {
                "role": "user",
                "content": judge_prompt,
            }
        ],
        "provider": {
            "order": OPENROUTER_JUDGE_PROVIDER_ORDER,
            "allow_fallbacks": False,
            "require_parameters": True,
        },
        "response_format": {"type": "json_object"},
        "reasoning": OPENROUTER_REASONING_CONFIG,
        "max_tokens": config.max_tokens,
    }


def openrouter_message_text(raw_response: dict[str, JsonValue]) -> str:
    choice = openrouter_first_choice(raw_response)
    message = require_dict(
        require_key(choice, "message", "OpenRouter choice"), "OpenRouter message"
    )
    content = require_key(message, "content", "OpenRouter message")
    if content is None:
        raise ValueError("OpenRouter judge response contained null message.content")
    if not isinstance(content, str):
        raise TypeError(
            f"Expected OpenRouter message.content to be a str, got {type(content).__name__}"
        )
    if not content.strip():
        raise ValueError("OpenRouter judge response did not contain any text content")
    return content


def openrouter_first_choice(raw_response: dict[str, JsonValue]) -> dict[str, JsonValue]:
    choices = require_list(require_key(raw_response, "choices", "OpenRouter response"), "choices")
    if not choices:
        raise ValueError("OpenRouter response did not contain any choices")
    return require_dict(choices[0], "OpenRouter choice")


def require_list(value: JsonValue, context: str) -> list[JsonValue]:
    if not isinstance(value, list):
        raise TypeError(f"Expected OpenRouter {context} to be a list, got {type(value).__name__}")
    return value


def openrouter_resolved_model(raw_response: dict[str, JsonValue]) -> str:
    value = require_key(raw_response, "model", "OpenRouter response")
    if not isinstance(value, str):
        raise TypeError(
            f"Expected OpenRouter response.model to be a str, got {type(value).__name__}"
        )
    return value


def openrouter_finish_reason(raw_response: dict[str, JsonValue]) -> str:
    choice = openrouter_first_choice(raw_response)
    value = require_key(choice, "finish_reason", "OpenRouter choice")
    if not isinstance(value, str):
        raise TypeError(
            f"Expected OpenRouter choice.finish_reason to be a str, got {type(value).__name__}"
        )
    return value


def normalize_openrouter_usage(raw_response: dict[str, JsonValue]) -> dict[str, JsonValue]:
    usage = require_dict(
        require_key(raw_response, "usage", "OpenRouter response"), "OpenRouter usage"
    )
    payload: dict[str, JsonValue] = {
        "input_tokens": require_int(usage, "prompt_tokens", "OpenRouter usage"),
        "output_tokens": require_int(usage, "completion_tokens", "OpenRouter usage"),
        "total_tokens": require_int(usage, "total_tokens", "OpenRouter usage"),
        "provider_reported_cost_usd": require_float(
            require_key(usage, "cost", "OpenRouter usage"), "OpenRouter usage.cost"
        ),
    }
    add_optional_token_detail(
        payload=payload,
        usage=usage,
        details_key="prompt_tokens_details",
        token_key="cached_tokens",
        output_key="cached_tokens",
    )
    add_optional_token_detail(
        payload=payload,
        usage=usage,
        details_key="completion_tokens_details",
        token_key="reasoning_tokens",
        output_key="reasoning_tokens",
    )
    return payload


def add_optional_token_detail(
    *,
    payload: dict[str, JsonValue],
    usage: dict[str, JsonValue],
    details_key: str,
    token_key: str,
    output_key: str,
) -> None:
    details = optional_dict(usage.get(details_key), details_key)
    if details is None:
        return
    if token_key not in details:
        return
    payload[output_key] = require_int(details, token_key, f"OpenRouter {details_key}")


def optional_dict(value: JsonValue | None, context: str) -> dict[str, JsonValue] | None:
    if value is None:
        return None
    return require_dict(value, f"OpenRouter {context}")


def require_float(value: JsonValue, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"Expected {context} to be a number, got {type(value).__name__}")
    return float(value)


def openrouter_judge_cost(
    model: str, usage: dict[str, JsonValue], latency_s: float
) -> dict[str, JsonValue]:
    input_tokens = require_int(usage, "input_tokens", "OpenRouter normalized usage")
    output_tokens = require_int(usage, "output_tokens", "OpenRouter normalized usage")
    total_tokens = require_int(usage, "total_tokens", "OpenRouter normalized usage")
    provider_cost = require_float(
        require_key(usage, "provider_reported_cost_usd", "OpenRouter normalized usage"),
        "OpenRouter normalized usage.provider_reported_cost_usd",
    )
    payload: dict[str, JsonValue] = {
        "model": model,
        "provider": "openrouter",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "provider_reported_cost_usd": provider_cost,
        "total_cost_usd": provider_cost,
        "pricing_policy": "openrouter_usage_cost_authoritative",
        "latency_s": latency_s,
    }
    copy_optional_usage_field(payload, usage, "cached_tokens")
    copy_optional_usage_field(payload, usage, "reasoning_tokens")
    return payload


def copy_optional_usage_field(
    payload: dict[str, JsonValue], usage: dict[str, JsonValue], key: str
) -> None:
    if key not in usage:
        return
    payload[key] = require_int(usage, key, "OpenRouter normalized usage")
