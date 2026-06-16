"""Google Gemini target calls and usage normalization."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol

from google import genai
from google.genai import types

from surge_gdp_benchmark.common import JsonValue, Sample, jsonable
from surge_gdp_benchmark.vendor_clients import (
    MODEL_SPECS,
    TargetResponse,
    UploadedPdf,
    require_dict,
    require_int,
    require_key,
)

GOOGLE_API_DOCS: dict[str, str] = {
    "document_processing": "https://ai.google.dev/gemini-api/docs/interactions/document-processing",
    "file_input_methods": "https://ai.google.dev/gemini-api/docs/interactions/file-input-methods",
    "thinking": "https://ai.google.dev/gemini-api/docs/thinking",
    "tokens": "https://ai.google.dev/gemini-api/docs/tokens",
    "pricing": "https://ai.google.dev/gemini-api/docs/pricing",
}


class GoogleTextResponse(Protocol):
    text: str | None


def gemini_thinking_config() -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.HIGH)
    )


def require_str(value: object, context: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"Expected {context} to be a str, got {type(value).__name__}")
    return value


def upload_google_pdf(
    sample: Sample, *, wait_timeout_s: int = 300, poll_interval_s: float = 2.0
) -> UploadedPdf:
    client = genai.Client(
        http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1))
    )
    uploaded = client.files.upload(
        file=sample.pdf_local,
        config=types.UploadFileConfig(
            display_name=Path(sample.pdf_path).name,
            mime_type="application/pdf",
        ),
    )
    active = wait_for_google_file(client, uploaded, wait_timeout_s, poll_interval_s)
    return UploadedPdf(
        provider="google",
        name=require_str(active.name, "Google uploaded file.name"),
        uri=require_str(active.uri, "Google uploaded file.uri"),
        mime_type=require_str(active.mime_type, "Google uploaded file.mime_type"),
        source_path=str(sample.pdf_local),
        provider_file=active,
    )


def wait_for_google_file(
    client: genai.Client,
    file: types.File,
    wait_timeout_s: int,
    poll_interval_s: float,
) -> types.File:
    name = require_str(file.name, "Google uploaded file.name")
    deadline = time.monotonic() + wait_timeout_s
    current = file
    while current.state == types.FileState.PROCESSING:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for Gemini file upload {name} to become active")
        time.sleep(poll_interval_s)
        current = client.files.get(name=name)
    if current.state == types.FileState.FAILED:
        raise ValueError(f"Gemini file upload failed for {name}")
    if current.state != types.FileState.ACTIVE:
        raise ValueError(f"Unexpected Gemini file upload state for {name}: {current.state}")
    return current


def google_response_text(response: GoogleTextResponse) -> str:
    text = response.text
    if not isinstance(text, str):
        raise TypeError(f"Expected Google response.text to be a str, got {type(text).__name__}")
    return text


def google_finish_reason(raw_dict: dict[str, JsonValue]) -> str:
    candidates = require_key(raw_dict, "candidates", "Google response")
    if not isinstance(candidates, list):
        raise TypeError(
            f"Expected Google response.candidates to be a list, got {type(candidates).__name__}"
        )
    if not candidates:
        raise ValueError("Google response did not contain any candidates")
    candidate = require_dict(candidates[0], "Google response.candidates[0]")
    return require_str(
        require_key(candidate, "finish_reason", "Google response.candidates[0]"),
        "Google response.candidates[0].finish_reason",
    )


def google_nullable_int(mapping: dict[str, JsonValue], key: str, context: str) -> int:
    value = require_key(mapping, key, context)
    if value is None:
        return 0
    return require_int(mapping, key, context)


def google_file_content(uploaded_pdf: UploadedPdf) -> types.File:
    if not isinstance(uploaded_pdf.provider_file, types.File):
        raise TypeError(
            "Expected Google uploaded PDF provider_file to be google.genai.types.File"
        )
    return uploaded_pdf.provider_file


def google_pdf_methodology(sample: Sample, uploaded_pdf: UploadedPdf) -> dict[str, JsonValue]:
    return {
        "provider_api": "Gemini generate_content API",
        "input_method": "client.files.upload then uploaded File object in contents",
        "pdf_url": sample.hf_resolve_url,
        "file_upload": True,
        "uploaded_file_name": uploaded_pdf.name,
        "uploaded_file_uri": uploaded_pdf.uri,
        "uploaded_file_mime_type": uploaded_pdf.mime_type,
        "upload_source_path": uploaded_pdf.source_path,
        "api_docs": GOOGLE_API_DOCS,
    }


def call_google(
    *, sample: Sample, prompt_text: str, uploaded_pdf: UploadedPdf | None = None
) -> TargetResponse:
    spec = MODEL_SPECS["gemini_3_1"]
    pdf_file = uploaded_pdf or upload_google_pdf(sample)
    if pdf_file.provider != "google":
        raise ValueError(f"Expected a Google uploaded PDF, got provider={pdf_file.provider}")
    client = genai.Client(
        http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1))
    )
    started = time.perf_counter()
    response = client.models.generate_content(
        model=spec.model_name,
        contents=[google_file_content(pdf_file), prompt_text],
        config=gemini_thinking_config(),
    )
    latency_s = time.perf_counter() - started
    raw = jsonable(response)
    raw_dict = require_dict(raw, "Google response")
    return TargetResponse(
        answer=google_response_text(response),
        requested_model=spec.model_name,
        resolved_model=spec.model_name,
        finish_reason=google_finish_reason(raw_dict),
        usage=normalize_google_usage(raw),
        raw_response=raw,
        methodology=google_pdf_methodology(sample, pdf_file),
        latency_s=latency_s,
    )


def normalize_google_usage(raw: JsonValue) -> dict[str, JsonValue]:
    raw_dict = require_dict(raw, "Google response")
    usage = require_dict(
        require_key(raw_dict, "usage_metadata", "Google response"),
        "Google usage_metadata",
    )
    prompt = require_int(usage, "prompt_token_count", "Google usage_metadata")
    candidates = require_int(usage, "candidates_token_count", "Google usage_metadata")
    cached_tokens = google_nullable_int(
        usage, "cached_content_token_count", "Google usage_metadata"
    )
    return {
        "input_tokens": prompt,
        "output_tokens": candidates,
        "total_tokens": require_int(usage, "total_token_count", "Google usage_metadata"),
        "cached_tokens": cached_tokens,
        "cache_creation_input_tokens": 0,
        "reasoning_tokens": google_nullable_int(
            usage, "thoughts_token_count", "Google usage_metadata"
        ),
    }
