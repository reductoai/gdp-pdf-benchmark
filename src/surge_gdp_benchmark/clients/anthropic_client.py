"""Anthropic target calls and usage normalization."""

from __future__ import annotations

import time

import anthropic
from anthropic.types import Message

from surge_gdp_benchmark.common import JsonValue, Sample, jsonable
from surge_gdp_benchmark.vendor_clients import (
    MODEL_SPECS,
    TargetResponse,
    require_dict,
    require_int,
    require_key,
)

ANTHROPIC_API_DOCS: dict[str, str] = {
    "pdf_support": "https://platform.claude.com/docs/en/build-with-claude/pdf-support",
    "pricing": "https://platform.claude.com/docs/en/about-claude/pricing",
    "adaptive_thinking": "https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking",
    "prompt_caching": "https://platform.claude.com/docs/en/build-with-claude/prompt-caching",
}


def anthropic_text_content(message: Message) -> str:
    text_blocks = [block.text for block in message.content if block.type == "text"]
    if not text_blocks:
        raise ValueError(f"Anthropic response: no text blocks, stop_reason={message.stop_reason}")
    return "".join(text_blocks)


def anthropic_pdf_methodology(sample: Sample) -> dict[str, JsonValue]:
    return {
        "provider_api": "Anthropic Messages API",
        "input_method": "document.source.url",
        "pdf_source": "url",
        "file_upload_fallback": False,
        "pdf_url": sample.hf_resolve_url,
        "api_docs": ANTHROPIC_API_DOCS,
    }


def anthropic_max_tokens() -> int:
    max_tokens = MODEL_SPECS["opus_4_8"].max_tokens
    if max_tokens is None:
        raise ValueError("Claude Opus 4.8 max_tokens must be configured")
    return max_tokens


def call_anthropic(
    *,
    sample: Sample,
    prompt_text: str,
) -> TargetResponse:
    spec = MODEL_SPECS["opus_4_8"]
    client = anthropic.Anthropic(timeout=spec.timeout_s, max_retries=0)
    started = time.perf_counter()
    message = client.messages.create(
        model=spec.model_name,
        max_tokens=anthropic_max_tokens(),
        thinking={"type": "adaptive"},
        output_config={"effort": "max"},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {"type": "url", "url": sample.hf_resolve_url},
                    },
                    {"type": "text", "text": prompt_text},
                ],
            }
        ],
    )
    latency_s = time.perf_counter() - started
    text = anthropic_text_content(message)
    raw = jsonable(message)
    return TargetResponse(
        answer=text,
        requested_model=spec.model_name,
        resolved_model=spec.model_name,
        finish_reason=message.stop_reason,
        usage=normalize_anthropic_usage(raw),
        raw_response=raw,
        methodology=anthropic_pdf_methodology(sample),
        latency_s=latency_s,
    )


def normalize_anthropic_usage(raw: JsonValue) -> dict[str, JsonValue]:
    raw_dict = require_dict(raw, "Anthropic response")
    usage = require_dict(require_key(raw_dict, "usage", "Anthropic response"), "Anthropic usage")
    details = require_dict(
        require_key(usage, "output_tokens_details", "Anthropic usage"),
        "Anthropic output_tokens_details",
    )
    input_tokens = require_int(usage, "input_tokens", "Anthropic usage")
    output_tokens = require_int(usage, "output_tokens", "Anthropic usage")
    cache_read_tokens = require_int(usage, "cache_read_input_tokens", "Anthropic usage")
    cache_creation_tokens = require_int(usage, "cache_creation_input_tokens", "Anthropic usage")
    total_input_tokens = input_tokens + cache_read_tokens + cache_creation_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_input_tokens + output_tokens,
        "cached_tokens": cache_read_tokens,
        "cache_creation_input_tokens": cache_creation_tokens,
        "reasoning_tokens": require_int(
            details, "thinking_tokens", "Anthropic output_tokens_details"
        ),
    }
