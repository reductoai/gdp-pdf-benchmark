"""Run target calls with both native PDF input and Reducto parsed text."""

from __future__ import annotations

import argparse
import dataclasses
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import cast

import anthropic
import requests
from openai import OpenAIError
from reducto import Reducto

from surge_gdp_benchmark.common import (
    JsonValue,
    ModelKey,
    Sample,
    default_output_dir,
    jsonable,
    load_samples,
    parse_model_keys,
    parse_sample_indices,
    sample_dir,
    target_dir,
    write_json,
    write_sample_metadata,
    write_target_error,
    write_target_result,
    write_text,
)
from surge_gdp_benchmark.vendor_clients import (
    call_target_model,
    require_dict,
    require_key,
    target_artifact,
)

PARSE_ARTIFACT_DIR = "reducto"


@dataclasses.dataclass(frozen=True)
class HybridTargetTask:
    sample: Sample
    model_key: ModelKey
    output_dir: Path
    parsed_doc: str
    force: bool


def chunks_from_parse_result(parse_result: object) -> list[dict[str, JsonValue]]:
    parse_payload = require_dict(jsonable(parse_result), "Reducto parse result")
    result = require_dict(
        require_key(parse_payload, "result", "Reducto parse result"),
        "Reducto parse result.result",
    )
    result_type = require_key(result, "type", "Reducto parse result.result")
    if result_type == "url":
        url = require_key(result, "url", "Reducto parse result.result")
        if not isinstance(url, str):
            raise TypeError("Expected Reducto parse result.result.url to be a string")
        response = requests.get(url, timeout=300)
        response.raise_for_status()
        payload = require_dict(response.json(), "Reducto parse JSON response")
        chunks = require_key(payload, "chunks", "Reducto parse JSON response")
    else:
        chunks = require_key(result, "chunks", "Reducto parse result.result")

    if not isinstance(chunks, list):
        raise ValueError("Reducto parse response did not contain a chunk list")
    normalized: list[dict[str, JsonValue]] = []
    for chunk in chunks:
        chunk_json = jsonable(chunk)
        if not isinstance(chunk_json, dict):
            raise ValueError("Reducto parse chunk was not JSON object-like")
        normalized.append(cast(dict[str, JsonValue], chunk_json))
    return normalized


def parsed_doc_from_chunks(chunks: list[dict[str, JsonValue]]) -> str:
    parts: list[str] = []
    for index, chunk in enumerate(chunks):
        content = chunk.get("content")
        if not isinstance(content, str):
            raise ValueError(f"Chunk {index} did not contain string content")
        parts.append(f"<!-- chunk {index} -->\n{content}")
    return "\n\n".join(parts)


def parse_artifact_dir(output_dir: Path, sample_idx: int) -> Path:
    return sample_dir(output_dir, sample_idx) / PARSE_ARTIFACT_DIR


def parse_sample(*, sample: Sample, output_dir: Path, force: bool) -> str:
    directory = parse_artifact_dir(output_dir, sample.sample_idx)
    parsed_doc_path = directory / "parsed_doc.txt"
    if parsed_doc_path.exists() and not force:
        print(f"skip parse sample={sample.sample_idx}", flush=True)
        return parsed_doc_path.read_text()

    directory.mkdir(parents=True, exist_ok=True)
    client = Reducto()
    started = time.perf_counter()
    upload = client.upload(file=sample.pdf_local)
    parse_config = {
        "formatting": {"add_page_markers": True, "table_output_format": "html"},
        "retrieval": {"chunking": {"chunk_mode": "page"}},
        "enhance": {
            "agentic": [{"scope": "text"}, {"scope": "table"}, {"scope": "figure"}],
            "summarize_figures": True,
        },
        "settings": {"force_url_result": True},
    }
    parse_result = client.parse.run(input=upload.file_id, **parse_config)
    parse_latency_s = time.perf_counter() - started
    chunks = chunks_from_parse_result(parse_result)
    parsed_doc = parsed_doc_from_chunks(chunks)
    write_json(
        directory / "parse.json",
        {
            "file_id": upload.file_id,
            "parse_latency_s": parse_latency_s,
            "parse_response": jsonable(parse_result),
            "parse_config": parse_config,
        },
    )
    write_json(directory / "chunks.json", {"chunks": chunks})
    write_text(parsed_doc_path, parsed_doc)
    print(f"parsed sample={sample.sample_idx} latency={parse_latency_s:.1f}s", flush=True)
    return parsed_doc


def load_or_parse_doc(*, sample: Sample, output_dir: Path, force_parse: bool) -> str:
    parsed_doc_path = parse_artifact_dir(output_dir, sample.sample_idx) / "parsed_doc.txt"
    if parsed_doc_path.exists() and not force_parse:
        return parsed_doc_path.read_text()
    return parse_sample(sample=sample, output_dir=output_dir, force=force_parse)


def run_hybrid_target(task: HybridTargetTask) -> None:
    destination = target_dir(
        task.output_dir, task.sample.sample_idx, "reducto_parse_plus_pdf", task.model_key
    )
    if not task.force and (
        (destination / "target.json").exists() or (destination / "target_error.json").exists()
    ):
        print(
            f"skip target sample={task.sample.sample_idx} arm=reducto_parse_plus_pdf "
            f"model={task.model_key}",
            flush=True,
        )
        return

    prompt_text = f"Document:\n{task.parsed_doc}\n\n{task.sample.prompt}"
    started = time.perf_counter()
    try:
        response = call_target_model(
            sample=task.sample,
            model_key=task.model_key,
            prompt_text=prompt_text,
        )
    except (OpenAIError, anthropic.APIError, requests.RequestException, OSError, ValueError) as exc:
        write_target_error(
            destination,
            sample=task.sample,
            arm="reducto_parse_plus_pdf",
            model_key=task.model_key,
            error_type=type(exc).__name__,
            error_message=str(exc),
            latency_s=time.perf_counter() - started,
        )
        print(
            f"failed target sample={task.sample.sample_idx} arm=reducto_parse_plus_pdf "
            f"model={task.model_key} error={type(exc).__name__}",
            flush=True,
        )
        return

    artifact = target_artifact(
        sample=task.sample,
        arm="reducto_parse_plus_pdf",
        model_key=task.model_key,
        response=response,
    )
    write_target_result(destination, response.answer, artifact)
    print(
        f"done target sample={task.sample.sample_idx} arm=reducto_parse_plus_pdf "
        f"model={task.model_key} latency={response.latency_s:.1f}s",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--samples", nargs="+", required=True)
    parser.add_argument("--models", nargs="+", default=["gpt_5_5", "opus_4_8"])
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--parse-concurrency", type=int, default=4)
    parser.add_argument("--target-concurrency", type=int, default=4)
    parser.add_argument("--force-parse", action="store_true")
    parser.add_argument("--force-targets", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    samples = load_samples(parse_sample_indices(args.samples), output_dir)
    model_keys = parse_model_keys(args.models)

    with ThreadPoolExecutor(max_workers=args.parse_concurrency) as executor:
        futures = {
            executor.submit(
                load_or_parse_doc,
                sample=sample,
                output_dir=output_dir,
                force_parse=args.force_parse,
            ): sample.sample_idx
            for sample in samples
        }
        parsed_docs = {futures[future]: future.result() for future in as_completed(futures)}

    tasks: list[HybridTargetTask] = []
    for sample in samples:
        write_sample_metadata(output_dir, sample)
        for model_key in model_keys:
            tasks.append(
                HybridTargetTask(
                    sample=sample,
                    model_key=model_key,
                    output_dir=output_dir,
                    parsed_doc=parsed_docs[sample.sample_idx],
                    force=args.force_targets,
                )
            )

    with ThreadPoolExecutor(max_workers=args.target_concurrency) as executor:
        futures = [executor.submit(run_hybrid_target, task) for task in tasks]
        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
