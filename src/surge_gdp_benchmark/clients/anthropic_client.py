"""Anthropic target calls and usage normalization.

Anthropic requests use the Files API (PDF uploaded once and referenced by
file_id, bypassing the ~32MB URL/base64 limit) plus streaming (so long
effort=max generations report progress and never hang silently on retries).
"""

from __future__ import annotations

import threading
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

FILES_BETA = "files-api-2025-04-14"
PROGRESS_LOG_INTERVAL_S = 30.0

ANTHROPIC_API_DOCS: dict[str, str] = {
    "pdf_support": "https://platform.claude.com/docs/en/build-with-claude/pdf-support",
    "files_api": "https://platform.claude.com/docs/en/build-with-claude/files",
    "streaming": "https://platform.claude.com/docs/en/build-with-claude/streaming",
    "pricing": "https://platform.claude.com/docs/en/about-claude/pricing",
    "adaptive_thinking": "https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking",
    "prompt_caching": "https://platform.claude.com/docs/en/build-with-claude/prompt-caching",
}

_file_id_lock = threading.Lock()
_file_id_cache: dict[str, str] = {}


def upload_pdf_cached(client: anthropic.Anthropic, sample: Sample) -> str:
    """Upload the sample PDF via the Files API once, caching the file_id by path."""
    key = str(sample.pdf_local)
    with _file_id_lock:
        cached = _file_id_cache.get(key)
    if cached is not None:
        return cached
    with sample.pdf_local.open("rb") as handle:
        uploaded = client.beta.files.upload(
            file=(sample.pdf_local.name, handle, "application/pdf"),
        )
    with _file_id_lock:
        _file_id_cache[key] = uploaded.id
    print(f"uploaded sample={sample.sample_idx} file_id={uploaded.id}", flush=True)
    return uploaded.id


def anthropic_text_content(message: Message) -> str:
    text_blocks = [block.text for block in message.content if block.type == "text"]
    if not text_blocks:
        raise ValueError(f"Anthropic response: no text blocks, stop_reason={message.stop_reason}")
    return "".join(text_blocks)


def anthropic_pdf_methodology(sample: Sample, file_id: str) -> dict[str, JsonValue]:
    return {
        "provider_api": "Anthropic Messages API + Files API (streaming)",
        "input_method": "document.source.file_id",
        "pdf_source": "files_api",
        "uploaded_file_id": file_id,
        "beta": FILES_BETA,
        "streamed": True,
        "pdf_path": sample.pdf_path,
        "api_docs": ANTHROPIC_API_DOCS,
    }


def anthropic_max_tokens() -> int:
    max_tokens = MODEL_SPECS["opus_4_8"].max_tokens
    if max_tokens is None:
        raise ValueError("Claude Opus 4.8 max_tokens must be configured")
    return max_tokens


def stream_final_message(
    client: anthropic.Anthropic,
    sample_idx: int,
    file_id: str,
    prompt_text: str,
) -> Message:
    """Stream the response, logging token progress so a stall is visible."""
    last_log = time.perf_counter()
    text_chars = 0
    thinking_chars = 0
    with client.beta.messages.stream(
        model=MODEL_SPECS["opus_4_8"].model_name,
        max_tokens=anthropic_max_tokens(),
        thinking={"type": "adaptive"},
        output_config={"effort": "max"},
        betas=[FILES_BETA],
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "document", "source": {"type": "file", "file_id": file_id}},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ],
    ) as stream:
        for event in stream:
            if event.type == "content_block_delta":
                delta = event.delta
                if delta.type == "text_delta":
                    text_chars += len(delta.text)
                elif delta.type == "thinking_delta":
                    thinking_chars += len(delta.thinking)
            now = time.perf_counter()
            if now - last_log >= PROGRESS_LOG_INTERVAL_S:
                print(
                    f"progress sample={sample_idx} thinking_chars={thinking_chars} "
                    f"answer_chars={text_chars}",
                    flush=True,
                )
                last_log = now
        return stream.get_final_message()


def call_anthropic(
    *,
    sample: Sample,
    prompt_text: str,
) -> TargetResponse:
    spec = MODEL_SPECS["opus_4_8"]
    client = anthropic.Anthropic(timeout=spec.timeout_s, max_retries=0)
    file_id = upload_pdf_cached(client, sample)
    started = time.perf_counter()
    message = stream_final_message(client, sample.sample_idx, file_id, prompt_text)
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
        methodology=anthropic_pdf_methodology(sample, file_id),
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
