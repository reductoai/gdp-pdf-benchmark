"""Shared data and artifact helpers for the GDP.pdf benchmark."""

from __future__ import annotations

import dataclasses
import json
import os
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Literal, TypeVar, cast

from datasets import load_dataset
from huggingface_hub import hf_hub_download

Arm = Literal["plain_pdf", "reducto_parse_plus_pdf"]
ModelKey = Literal["gpt_5_5", "opus_4_8", "gemini_3_1"]
JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]

DEFAULT_MODELS: tuple[ModelKey, ...] = ("gpt_5_5", "opus_4_8", "gemini_3_1")
ALL_MODELS: tuple[ModelKey, ...] = DEFAULT_MODELS
OUTPUT_DIR_ENV_VAR = "GDP_PDF_BENCHMARK_OUTPUT_DIR"
HF_REPO_ID = "surgeai/GDP.pdf"
HF_PDF_RESOLVE_PREFIX = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/"

T = TypeVar("T")


@dataclasses.dataclass(frozen=True)
class Rubric:
    """One GDP.pdf rubric item."""

    n: int
    criterion: str
    type: str | None
    severity: str | None
    implicitness: str | None
    subjectiveness: str | None
    failure_mode: str | None

    def to_json(self) -> dict[str, JsonValue]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class Sample:
    """One GDP.pdf dataset sample and local PDF path."""

    sample_idx: int
    task_id: str
    domain: str
    pdf_path: str
    prompt: str
    rubrics: tuple[Rubric, ...]
    pdf_local: Path

    @property
    def hf_resolve_url(self) -> str:
        return f"{HF_PDF_RESOLVE_PREFIX}{self.pdf_path}"

    def metadata_json(self) -> dict[str, JsonValue]:
        return {
            "sample_idx": self.sample_idx,
            "task_id": self.task_id,
            "domain": self.domain,
            "pdf_path": self.pdf_path,
            "prompt": self.prompt,
            "rubrics": [rubric.to_json() for rubric in self.rubrics],
            "pdf_local": str(self.pdf_local),
            "hf_resolve_url": self.hf_resolve_url,
        }


def default_output_dir() -> Path:
    configured = os.getenv(OUTPUT_DIR_ENV_VAR)
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.cwd() / ".data" / "gdp_pdf_benchmark"


def parse_sample_indices(values: Sequence[str]) -> list[int]:
    indices: list[int] = []
    for value in values:
        for part in value.split(","):
            stripped = part.strip()
            if not stripped:
                continue
            if "-" in stripped:
                start, end = stripped.split("-", 1)
                indices.extend(range(int(start), int(end) + 1))
            else:
                indices.append(int(stripped))
    return sorted(set(indices))


def parse_model_keys(values: Sequence[str]) -> list[ModelKey]:
    valid = set(ALL_MODELS)
    parsed: list[ModelKey] = []
    for value in values:
        for part in value.split(","):
            key = part.strip()
            if not key:
                continue
            if key not in valid:
                raise ValueError(f"Unknown model key {key!r}; valid keys: {sorted(valid)}")
            parsed.append(cast(ModelKey, key))
    return parsed


def parse_arms(values: Sequence[str]) -> list[Arm]:
    valid = {"plain_pdf", "reducto_parse_plus_pdf"}
    parsed: list[Arm] = []
    for value in values:
        for part in value.split(","):
            arm = part.strip()
            if not arm:
                continue
            if arm not in valid:
                raise ValueError(f"Unknown arm {arm!r}; valid arms: {sorted(valid)}")
            parsed.append(cast(Arm, arm))
    return parsed


def sample_dir(output_dir: Path, sample_idx: int) -> Path:
    return output_dir / "samples" / f"{sample_idx:03d}"


def target_dir(output_dir: Path, sample_idx: int, arm: Arm, model_key: ModelKey) -> Path:
    return sample_dir(output_dir, sample_idx) / arm / model_key


def source_pdf_path(output_dir: Path, sample_idx: int, pdf_path: str) -> Path:
    return output_dir / "source_pdfs" / f"{sample_idx:03d}_{Path(pdf_path).name}"


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(
        f"{path.name}.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.tmp"
    )
    temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    temp_path.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def read_json(path: Path) -> dict[str, JsonValue]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return cast(dict[str, JsonValue], payload)


def jsonable(value: object) -> JsonValue:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, list | tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return jsonable(model_dump(mode="json"))
    return str(value)


def rubric_from_json(payload: Mapping[str, JsonValue]) -> Rubric:
    n = payload["n"]
    if not isinstance(n, int):
        raise ValueError(f"Rubric number must be an int: {n!r}")
    return Rubric(
        n=n,
        criterion=str(payload["criterion"]),
        type=cast(str | None, payload.get("type")),
        severity=cast(str | None, payload.get("severity")),
        implicitness=cast(str | None, payload.get("implicitness")),
        subjectiveness=cast(str | None, payload.get("subjectiveness")),
        failure_mode=cast(str | None, payload.get("failure_mode")),
    )


def load_sample_from_artifacts(output_dir: Path, sample_idx: int) -> Sample | None:
    metadata_path = sample_dir(output_dir, sample_idx) / "sample.json"
    if not metadata_path.exists():
        return None
    payload = read_json(metadata_path)
    pdf_path = str(payload["pdf_path"])
    pdf_local = source_pdf_path(output_dir, sample_idx, pdf_path)
    if not pdf_local.exists():
        return None
    rubrics_value = payload.get("rubrics")
    if not isinstance(rubrics_value, list):
        raise ValueError(f"sample.json rubrics must be a list: {metadata_path}")
    return Sample(
        sample_idx=sample_idx,
        task_id=str(payload["task_id"]),
        domain=str(payload["domain"]),
        pdf_path=pdf_path,
        prompt=str(payload["prompt"]),
        rubrics=tuple(
            rubric_from_json(cast(Mapping[str, JsonValue], rubric)) for rubric in rubrics_value
        ),
        pdf_local=pdf_local,
    )


def load_samples(sample_indices: Iterable[int], output_dir: Path | None = None) -> list[Sample]:
    ordered = list(sample_indices)
    samples_by_idx: dict[int, Sample] = {}
    missing: list[int] = []
    for sample_idx in ordered:
        artifact_sample = load_sample_from_artifacts(output_dir, sample_idx) if output_dir else None
        if artifact_sample is not None:
            samples_by_idx[sample_idx] = artifact_sample
        else:
            missing.append(sample_idx)

    if missing:
        dataset = load_dataset(HF_REPO_ID)["test"]
        for sample_idx in missing:
            row = dataset[sample_idx]
            pdf_path = str(row["pdf_path"])
            pdf_local = Path(
                hf_hub_download(repo_id=HF_REPO_ID, filename=pdf_path, repo_type="dataset")
            )
            rubrics: list[Rubric] = []
            for n in range(1, 31):
                criterion = row.get(f"rubric - {n}. criterion")
                if criterion:
                    rubrics.append(
                        Rubric(
                            n=n,
                            criterion=str(criterion),
                            type=row.get(f"rubric - {n}. criterion_type"),
                            severity=row.get(f"rubric - {n}. criterion_severity"),
                            implicitness=row.get(f"rubric - {n}. criterion_implicitness"),
                            subjectiveness=row.get(f"rubric - {n}. criterion_subjectiveness"),
                            failure_mode=row.get(f"rubric - {n}. criterion_failure_mode"),
                        )
                    )
            samples_by_idx[sample_idx] = Sample(
                sample_idx=sample_idx,
                task_id=str(row["task_id"]),
                domain=str(row["domain"]),
                pdf_path=pdf_path,
                prompt=str(row["prompt"]),
                rubrics=tuple(rubrics),
                pdf_local=pdf_local,
            )

    return [samples_by_idx[sample_idx] for sample_idx in ordered]


def write_sample_metadata(output_dir: Path, sample: Sample) -> None:
    write_json(sample_dir(output_dir, sample.sample_idx) / "sample.json", sample.metadata_json())


def write_target_result(destination: Path, answer: str, artifact: Mapping[str, object]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    write_text(destination / "answer.txt", answer)
    write_json(destination / "target.json", artifact)
    error_path = destination / "target_error.json"
    if error_path.exists():
        error_path.unlink()


def write_target_error(
    destination: Path,
    *,
    sample: Sample,
    arm: Arm,
    model_key: ModelKey,
    error_type: str,
    error_message: str,
    latency_s: float,
) -> None:
    write_json(
        destination / "target_error.json",
        {
            "sample_idx": sample.sample_idx,
            "task_id": sample.task_id,
            "arm": arm,
            "model_key": model_key,
            "status": "failed",
            "error_type": error_type,
            "error_message": error_message,
            "latency_s": latency_s,
            "started_at_unix": time.time() - latency_s,
            "finished_at_unix": time.time(),
        },
    )
