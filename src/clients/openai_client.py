"""OpenAI target calls and usage normalization."""

from __future__ import annotations

import time

from openai import OpenAI

from common import JsonValue, Sample, jsonable
from vendor_clients import (
    MODEL_SPECS,
    TargetResponse,
    require_dict,
    require_int,
    require_key,
)

OPENAI_API_DOCS: dict[str, str] = {
    "responses_api": "https://developers.openai.com/api/reference/resources/responses/",
    "file_inputs": "https://developers.openai.com/api/docs/guides/file-inputs",
    "reasoning": "https://developers.openai.com/api/docs/guides/reasoning",
    "pricing": "https://openai.com/api/pricing/",
}


def openai_pdf_methodology(sample: Sample) -> dict[str, JsonValue]:
    return {
        "provider_api": "OpenAI Responses API",
        "input_method": "input_file.file_url",
        "pdf_url": sample.hf_resolve_url,
        "file_upload_fallback": False,
        "api_docs": OPENAI_API_DOCS,
    }


def call_openai(
    *,
    sample: Sample,
    prompt_text: str,
) -> TargetResponse:
    client = OpenAI(timeout=MODEL_SPECS["gpt_5_5"].timeout_s, max_retries=0)
    started = time.perf_counter()
    response = client.responses.create(
        model=MODEL_SPECS["gpt_5_5"].model_name,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_file", "file_url": sample.hf_resolve_url},
                    {"type": "input_text", "text": prompt_text},
                ],
            }
        ],
        reasoning={"effort": "xhigh"},
    )
    latency_s = time.perf_counter() - started
    raw = jsonable(response)
    raw_dict = require_dict(raw, "OpenAI response")
    usage_payload = require_dict(require_key(raw_dict, "usage", "OpenAI response"), "OpenAI usage")
    usage = normalize_openai_usage(usage_payload)
    return TargetResponse(
        answer=response.output_text,
        requested_model=MODEL_SPECS["gpt_5_5"].model_name,
        resolved_model=str(require_key(raw_dict, "model", "OpenAI response")),
        finish_reason=str(require_key(raw_dict, "status", "OpenAI response")),
        usage=usage,
        raw_response=raw,
        methodology=openai_pdf_methodology(sample),
        latency_s=latency_s,
    )


def normalize_openai_usage(payload: object) -> dict[str, JsonValue]:
    usage = require_dict(payload, "OpenAI usage")
    input_details = require_dict(
        require_key(usage, "input_tokens_details", "OpenAI usage"),
        "OpenAI input_tokens_details",
    )
    output_details = require_dict(
        require_key(usage, "output_tokens_details", "OpenAI usage"),
        "OpenAI output_tokens_details",
    )
    cached_tokens = require_int(input_details, "cached_tokens", "OpenAI input_tokens_details")
    return {
        "input_tokens": require_int(usage, "input_tokens", "OpenAI usage"),
        "output_tokens": require_int(usage, "output_tokens", "OpenAI usage"),
        "total_tokens": require_int(usage, "total_tokens", "OpenAI usage"),
        "cached_tokens": cached_tokens,
        "cache_creation_input_tokens": 0,
        "reasoning_tokens": require_int(
            output_details, "reasoning_tokens", "OpenAI output_tokens_details"
        ),
    }
