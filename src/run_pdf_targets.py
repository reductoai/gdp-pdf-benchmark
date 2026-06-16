"""Run GDP.pdf native-PDF target model calls using direct vendor APIs."""

from __future__ import annotations

import argparse
import dataclasses
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
from openai import OpenAIError

from common import (
    ModelKey,
    Sample,
    default_output_dir,
    load_samples,
    parse_model_keys,
    parse_sample_indices,
    target_dir,
    write_sample_metadata,
    write_target_error,
    write_target_result,
)
from vendor_clients import (
    call_target_model,
    target_artifact,
)


@dataclasses.dataclass(frozen=True)
class PdfTargetTask:
    sample: Sample
    model_key: ModelKey
    output_dir: Path
    force: bool


def run_pdf_target(task: PdfTargetTask) -> None:
    destination = target_dir(task.output_dir, task.sample.sample_idx, "plain_pdf", task.model_key)
    if not task.force and (
        (destination / "target.json").exists() or (destination / "target_error.json").exists()
    ):
        print(
            f"skip sample={task.sample.sample_idx} arm=plain_pdf model={task.model_key}", flush=True
        )
        return

    print(f"start sample={task.sample.sample_idx} arm=plain_pdf model={task.model_key}", flush=True)
    started = time.perf_counter()
    try:
        response = call_target_model(
            sample=task.sample,
            model_key=task.model_key,
            prompt_text=task.sample.prompt,
        )
    except (OpenAIError, anthropic.APIError, OSError, ValueError) as exc:
        latency_s = time.perf_counter() - started
        write_target_error(
            destination,
            sample=task.sample,
            arm="plain_pdf",
            model_key=task.model_key,
            error_type=type(exc).__name__,
            error_message=str(exc),
            latency_s=latency_s,
        )
        print(
            f"failed sample={task.sample.sample_idx} arm=plain_pdf model={task.model_key} "
            f"error={type(exc).__name__}",
            flush=True,
        )
        return

    artifact = target_artifact(
        sample=task.sample,
        arm="plain_pdf",
        model_key=task.model_key,
        response=response,
    )
    write_target_result(destination, response.answer, artifact)
    print(
        f"done sample={task.sample.sample_idx} arm=plain_pdf model={task.model_key} "
        f"latency={response.latency_s:.1f}s",
        flush=True,
    )


def run_targets(
    *,
    samples: list[Sample],
    model_keys: list[ModelKey],
    output_dir: Path,
    concurrency: int,
    force: bool,
) -> None:
    tasks: list[PdfTargetTask] = []
    for sample in samples:
        write_sample_metadata(output_dir, sample)
        for model_key in model_keys:
            tasks.append(
                PdfTargetTask(
                    sample=sample,
                    model_key=model_key,
                    output_dir=output_dir,
                    force=force,
                )
            )

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(run_pdf_target, task) for task in tasks]
        for future in as_completed(futures):
            future.result()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--samples", nargs="+", required=True)
    parser.add_argument("--models", nargs="+", default=["gpt_5_5", "opus_4_8"])
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    samples = load_samples(parse_sample_indices(args.samples), output_dir)
    run_targets(
        samples=samples,
        model_keys=parse_model_keys(args.models),
        output_dir=output_dir,
        concurrency=args.concurrency,
        force=args.force,
    )


if __name__ == "__main__":
    main()
