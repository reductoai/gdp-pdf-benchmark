"""Grade saved target answers against GDP.pdf rubric criteria."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from clients.openrouter_client import (
    DEFAULT_OPENROUTER_JUDGE_MODEL,
    call_openrouter_judge,
)
from common import (
    Arm,
    JsonValue,
    ModelKey,
    Rubric,
    Sample,
    default_output_dir,
    load_samples,
    parse_arms,
    parse_model_keys,
    parse_sample_indices,
    read_json,
    sample_dir,
    target_dir,
    write_json,
)
from vendor_clients import require_dict, require_key

JUDGE_PROVIDER = "openrouter"
JUDGE_MODEL = DEFAULT_OPENROUTER_JUDGE_MODEL


def answer_path(output_dir: Path, sample_idx: int, arm: Arm, model_key: ModelKey) -> Path:
    return target_dir(output_dir, sample_idx, arm, model_key) / "answer.txt"


def grade_path(output_dir: Path, sample_idx: int, arm: Arm, model_key: ModelKey) -> Path:
    return target_dir(output_dir, sample_idx, arm, model_key) / "grade.json"


def target_error_path(output_dir: Path, sample_idx: int, arm: Arm, model_key: ModelKey) -> Path:
    return target_dir(output_dir, sample_idx, arm, model_key) / "target_error.json"


def rubric_json(rubrics: list[Rubric]) -> list[dict[str, JsonValue]]:
    return [rubric.to_json() for rubric in rubrics]


def judge_prompt(sample: Sample, answer: str, rubrics: list[Rubric]) -> str:
    required_keys = [normalise_question_key(rubric) for rubric in rubrics]
    return f"""
You are grading a response for the GDP.PDF benchmark.

You are given:
1. The original user question.
2. The model response.
3. An expert rubric decomposed into atomic criteria.

Grade each rubric criterion independently.
You do not have the source PDF. Treat each rubric criterion as the authoritative
description of the expected answer or expected behavior.
Return 1 only if the model response satisfies that criterion.
Return 0 if the response is wrong, unsupported by the rubric, ambiguous, incomplete,
contradicted, or fabricates details.
For criteria about avoiding an error, return 1 only if the response avoids that error.
For unsupported-query/abstention criteria, full credit requires explicitly saying the PDF
does not support the requested conclusion and inventing no specifics.
Do not use the gold/reference answer. Grade only from the question, response, and rubric.

Question:
{sample.prompt}

Model response:
{answer}

Rubric criteria:
{json.dumps(rubric_json(rubrics), indent=2)}

Required rubrics_check keys:
{json.dumps(required_keys)}

Return JSON:
{{
  "rubrics_check": {{
    "question_1": {{"reason": "short PDF-grounded reason", "score": 0 or 1}}
  }}
}}

The final JSON must include exactly one rubrics_check item for every required key.
Do not rename, omit, or renumber the required keys. Put "reason" before "score"
inside every rubric item.
""".strip()


def strip_json(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped


def normalise_question_key(rubric: Rubric) -> str:
    return f"question_{rubric.n}"


def judge_rubric_batch(
    sample: Sample, answer: str, rubrics: list[Rubric]
) -> dict[str, JsonValue]:
    response = call_openrouter_judge(
        sample=sample,
        judge_prompt=judge_prompt(sample, answer, rubrics),
    )
    parsed = require_dict(json.loads(strip_json(response.output_text)), "judge JSON")
    checks = require_dict(require_key(parsed, "rubrics_check", "judge JSON"), "rubrics_check")
    return {
        "rubrics_check": normalise_judge_checks(rubrics, checks),
        "latency_s": response.latency_s,
        "cost": response.cost,
        "usage": response.usage,
        "raw_response": response.raw_response,
        "methodology": response.methodology,
        "requested_model": response.requested_model,
        "resolved_model": response.resolved_model,
        "finish_reason": response.finish_reason,
    }


def judge_all_rubrics(
    sample: Sample, answer: str, max_rubrics_per_call: int | None
) -> dict[str, JsonValue]:
    rubric_batches = list(batch_rubrics(list(sample.rubrics), max_rubrics_per_call))
    calls = [judge_rubric_batch(sample, answer, rubrics) for rubrics in rubric_batches]
    all_checks: dict[str, JsonValue] = {}
    total_latency = 0.0
    total_cost = 0.0
    for call in calls:
        checks = require_dict(
            require_key(call, "rubrics_check", "judge result"),
            "judge result.rubrics_check",
        )
        all_checks.update(checks)
        total_latency += required_number_value(
            require_key(call, "latency_s", "judge result"), "judge result.latency_s"
        )
        cost = require_dict(require_key(call, "cost", "judge result"), "judge result.cost")
        total_cost += required_number_value(
            require_key(cost, "total_cost_usd", "judge result.cost"),
            "judge result.cost.total_cost_usd",
        )
    return {
        "rubrics_check": normalise_judge_checks(list(sample.rubrics), all_checks),
        "calls": calls,
        "latency_s": total_latency,
        "cost": {"total_cost_usd": total_cost},
    }


def batch_rubrics(
    rubrics: list[Rubric], max_rubrics_per_call: int | None
) -> list[list[Rubric]]:
    if max_rubrics_per_call is None:
        return [rubrics]
    if max_rubrics_per_call < 1:
        raise ValueError("--max-rubrics-per-call must be at least 1")
    return [
        rubrics[index : index + max_rubrics_per_call]
        for index in range(0, len(rubrics), max_rubrics_per_call)
    ]


def normalise_judge_checks(
    rubrics: list[Rubric], checks: dict[str, JsonValue]
) -> dict[str, JsonValue]:
    normalised: dict[str, JsonValue] = {}
    for rubric in rubrics:
        key = normalise_question_key(rubric)
        item = require_dict(require_key(checks, key, "judge checks"), f"judge checks.{key}")
        normalised[key] = normalise_judge_item(item, key)
    return normalised


def normalise_judge_item(item: dict[str, JsonValue], key: str) -> dict[str, JsonValue]:
    reason = require_key(item, "reason", f"judge checks.{key}")
    if not isinstance(reason, str):
        raise TypeError(f"Expected judge checks.{key}.reason to be a string")
    score = require_key(item, "score", f"judge checks.{key}")
    if isinstance(score, bool) or not isinstance(score, int):
        raise TypeError(f"Expected judge checks.{key}.score to be 0 or 1")
    if score not in {0, 1}:
        raise ValueError(f"Expected judge checks.{key}.score to be 0 or 1, got {score}")
    return {"reason": reason, "score": score}


def score_grade(sample: Sample, checks: dict[str, JsonValue]) -> dict[str, JsonValue]:
    passed_items: list[bool] = []
    failed_questions: list[str] = []
    for rubric in sample.rubrics:
        key = normalise_question_key(rubric)
        item = require_dict(require_key(checks, key, "judge checks"), f"judge checks.{key}")
        score = require_key(normalise_judge_item(item, key), "score", f"judge checks.{key}")
        passed = score == 1
        passed_items.append(passed)
        if not passed:
            failed_questions.append(key)
    return {
        "all_rubrics_pass": all(passed_items),
        "micro_rubric_pass_rate": (
            sum(passed_items) / len(passed_items) if passed_items else 0
        ),
        "passed": sum(passed_items),
        "total": len(passed_items),
        "failed_questions": failed_questions,
    }


def failure_grade(
    sample: Sample, arm: Arm, model_key: ModelKey, error_payload: dict[str, JsonValue]
) -> dict[str, JsonValue]:
    checks: dict[str, JsonValue] = {}
    for rubric in sample.rubrics:
        error_message = require_key(error_payload, "error_message", "target error")
        checks[normalise_question_key(rubric)] = {
            "reason": f"Target generation failed: {error_message}",
            "score": 0,
        }
    return {
        "sample_idx": sample.sample_idx,
        "task_id": sample.task_id,
        "arm": arm,
        "model_key": model_key,
        "judge_provider": JUDGE_PROVIDER,
        "judge_model": JUDGE_MODEL,
        "source_pdf_included": False,
        "target_failed": True,
        "rubrics_per_call": "not_applicable_target_failed",
        "grade": {"rubrics_check": checks},
        "score": score_grade(sample, checks),
        "calls": [],
        "judge_total_latency_s": 0,
        "judge_total_cost_usd": 0,
    }


def run_grade(
    *,
    sample: Sample,
    arm: Arm,
    model_key: ModelKey,
    output_dir: Path,
    force: bool,
    max_rubrics_per_call: int | None,
) -> None:
    destination = grade_path(output_dir, sample.sample_idx, arm, model_key)
    if destination.exists() and not force:
        print(f"skip judge sample={sample.sample_idx} arm={arm} model={model_key}", flush=True)
        return

    error_path = target_error_path(output_dir, sample.sample_idx, arm, model_key)
    if error_path.exists():
        payload = failure_grade(sample, arm, model_key, read_json(error_path))
        write_json(destination, payload)
        print(f"failed-target sample={sample.sample_idx} arm={arm} model={model_key}", flush=True)
        return

    path = answer_path(output_dir, sample.sample_idx, arm, model_key)
    if not path.exists():
        raise FileNotFoundError(f"Missing target answer: {path}")
    answer = path.read_text()
    result = judge_all_rubrics(sample, answer, max_rubrics_per_call)
    all_checks = require_dict(
        require_key(result, "rubrics_check", "judge result"), "judge result.rubrics_check"
    )
    cost = require_dict(require_key(result, "cost", "judge result"), "judge result.cost")
    total_latency = required_number_value(
        require_key(result, "latency_s", "judge result"), "judge result.latency_s"
    )
    total_cost = required_number_value(
        require_key(cost, "total_cost_usd", "judge result.cost"),
        "judge result.cost.total_cost_usd",
    )

    payload = {
        "sample_idx": sample.sample_idx,
        "task_id": sample.task_id,
        "arm": arm,
        "model_key": model_key,
        "judge_provider": JUDGE_PROVIDER,
        "judge_model": JUDGE_MODEL,
        "source_pdf_included": False,
        "source_pdf_url": sample.hf_resolve_url,
        "rubrics_per_call": max_rubrics_per_call or "all",
        "rubrics_in_call": len(sample.rubrics),
        "grade": {"rubrics_check": all_checks},
        "score": score_grade(sample, all_checks),
        "calls": require_key(result, "calls", "judge result"),
        "judge_total_latency_s": total_latency,
        "judge_total_cost_usd": total_cost,
    }
    write_json(destination, payload)
    score = payload["score"]
    print(
        f"judged sample={sample.sample_idx} arm={arm} model={model_key} "
        f"pass={score['passed']}/{score['total']}",
        flush=True,
    )


def row_target_succeeded(row: dict[str, JsonValue]) -> bool:
    return row.get("target_status") == "succeeded" and row.get("target_error_type") is None


def number_value(value: JsonValue) -> float | None:
    if isinstance(value, bool):
        return None
    return float(value) if isinstance(value, int | float) else None


def required_number_value(value: JsonValue, context: str) -> float:
    number = number_value(value)
    if number is None:
        raise TypeError(f"Expected {context} to be a number")
    return number


def required_int_value(value: JsonValue, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Expected {context} to be an int")
    return value


def average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def optional_json_object(value: JsonValue | None, context: str) -> dict[str, JsonValue] | None:
    if value is None:
        return None
    return require_dict(value, context)


def optional_summary_value(
    mapping: dict[str, JsonValue] | None, key: str, context: str
) -> JsonValue:
    if mapping is None:
        return None
    return require_key(mapping, key, context)


def optional_row_number(row: dict[str, JsonValue], key: str) -> float | None:
    return number_value(require_key(row, key, "summary row"))


def row_count_value(row: dict[str, JsonValue], key: str) -> int:
    value = require_key(row, key, "summary row")
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Expected summary row {key} to be an int or None")
    return value


def build_summary(
    output_dir: Path, samples: list[Sample], arms: list[Arm], model_keys: list[ModelKey]
) -> None:
    rows: list[dict[str, JsonValue]] = []
    for sample in samples:
        for arm in arms:
            for model_key in model_keys:
                tdir = target_dir(output_dir, sample.sample_idx, arm, model_key)
                target = (
                    read_json(tdir / "target.json") if (tdir / "target.json").exists() else None
                )
                error = (
                    read_json(tdir / "target_error.json")
                    if (tdir / "target_error.json").exists()
                    else None
                )
                grade = read_json(tdir / "grade.json") if (tdir / "grade.json").exists() else None
                target_obj = optional_json_object(target, "target artifact")
                error_obj = optional_json_object(error, "target error artifact")
                grade_obj = optional_json_object(grade, "grade artifact")
                score = (
                    require_dict(require_key(grade_obj, "score", "grade artifact"), "grade score")
                    if grade_obj is not None
                    else None
                )
                cost = (
                    require_dict(require_key(target_obj, "cost", "target artifact"), "target cost")
                    if target_obj is not None
                    else None
                )
                row: dict[str, JsonValue] = {
                    "sample_idx": sample.sample_idx,
                    "task_id": sample.task_id,
                    "domain": sample.domain,
                    "arm": arm,
                    "model_key": model_key,
                    "target_status": "succeeded" if target_obj is not None else "failed",
                    "target_error_type": optional_summary_value(
                        error_obj, "error_type", "target error artifact"
                    ),
                    "target_error_message": optional_summary_value(
                        error_obj, "error_message", "target error artifact"
                    ),
                    "all_rubrics_pass": optional_summary_value(
                        score, "all_rubrics_pass", "grade score"
                    ),
                    "micro_rubric_pass_rate": optional_summary_value(
                        score, "micro_rubric_pass_rate", "grade score"
                    ),
                    "passed": optional_summary_value(score, "passed", "grade score"),
                    "total": optional_summary_value(score, "total", "grade score"),
                    "answer_generation_latency_s": (
                        require_key(target_obj, "latency_s", "target artifact")
                        if target_obj is not None
                        else optional_summary_value(error_obj, "latency_s", "target error artifact")
                    ),
                    "llm_answer_cost_usd": optional_summary_value(
                        cost, "total_cost_usd", "target cost"
                    ),
                }
                answer_latency = optional_row_number(row, "answer_generation_latency_s")
                if answer_latency is not None:
                    row["total_latency_s"] = answer_latency
                if arm == "reducto_parse_plus_pdf":
                    parse_path = (
                        sample_dir(output_dir, sample.sample_idx) / "reducto" / "parse.json"
                    )
                    if parse_path.exists():
                        parse_payload = require_dict(read_json(parse_path), "parse artifact")
                        row["parse_latency_s"] = require_key(
                            parse_payload, "parse_latency_s", "parse artifact"
                        )
                        parse_latency = optional_row_number(row, "parse_latency_s")
                        if answer_latency is not None and parse_latency is not None:
                            row["total_latency_s"] = answer_latency + parse_latency
                rows.append(row)

    write_json(
        output_dir / "summary.json",
        {
            "rows": rows,
            "aggregate": aggregate_rows(rows, arms),
            "paired_success_aggregate": paired_success_aggregate(rows, arms, model_keys),
            "target_failure_summary": target_failure_summary(rows, arms, model_keys),
            "methodology": {
                "judge_provider": JUDGE_PROVIDER,
                "judge_model": JUDGE_MODEL,
                "source_pdf_included": False,
                "source_pdf_source": "huggingface_resolve_url",
                "rubrics_per_call": "all",
                "max_rubrics_in_single_call": max(
                    (len(sample.rubrics) for sample in samples),
                    default=0,
                ),
                "paired_success_metrics": (
                    "A sample is omitted for a model unless every requested arm has a "
                    "successful target output and grade."
                ),
            },
        },
    )


def aggregate_rows(rows: list[dict[str, JsonValue]], arms: list[Arm]) -> dict[str, JsonValue]:
    payload: dict[str, JsonValue] = {}
    for arm in arms:
        arm_rows = [row for row in rows if row["arm"] == arm]
        passed = sum(row_count_value(row, "passed") for row in arm_rows)
        total = sum(row_count_value(row, "total") for row in arm_rows)
        payload[arm] = {
            "runs": len(arm_rows),
            "passed": passed,
            "total": total,
            "micro_rubric_pass_rate": passed / total if total else 0,
        }
    return payload


def paired_success_aggregate(
    rows: list[dict[str, JsonValue]], arms: list[Arm], model_keys: list[ModelKey]
) -> dict[str, JsonValue]:
    payload: dict[str, JsonValue] = {}
    for model_key in model_keys:
        sample_indices = sorted(
            {
                required_int_value(
                    require_key(row, "sample_idx", "summary row"), "summary row.sample_idx"
                )
                for row in rows
                if row["model_key"] == model_key
            }
        )
        included = [
            sample_idx
            for sample_idx in sample_indices
            if all(
                any(
                    row["sample_idx"] == sample_idx
                    and row["model_key"] == model_key
                    and row["arm"] == arm
                    and row_target_succeeded(row)
                    for row in rows
                )
                for arm in arms
            )
        ]
        arm_payload: dict[str, JsonValue] = {}
        for arm in arms:
            arm_rows = [
                row
                for row in rows
                if row["model_key"] == model_key
                and row["arm"] == arm
                and row.get("sample_idx") in included
            ]
            passed = sum(row_count_value(row, "passed") for row in arm_rows)
            total = sum(row_count_value(row, "total") for row in arm_rows)
            arm_payload[arm] = {
                "runs": len(arm_rows),
                "passed": passed,
                "total": total,
                "micro_rubric_pass_rate": passed / total if total else 0,
                "all_rubrics_pass_rate": (
                    sum(1 for row in arm_rows if row.get("all_rubrics_pass") is True)
                    / len(arm_rows)
                    if arm_rows
                    else 0
                ),
                "avg_answer_generation_latency_s": average(
                    [
                        value
                        for row in arm_rows
                        if (value := number_value(row.get("answer_generation_latency_s")))
                        is not None
                    ]
                ),
                "avg_total_latency_s": average(
                    [
                        value
                        for row in arm_rows
                        if (value := number_value(row.get("total_latency_s"))) is not None
                    ]
                ),
                "avg_parse_latency_s": average(
                    [
                        value
                        for row in arm_rows
                        if (value := number_value(row.get("parse_latency_s"))) is not None
                    ]
                ),
                "avg_llm_answer_cost_usd": average(
                    [
                        value
                        for row in arm_rows
                        if (value := number_value(row.get("llm_answer_cost_usd"))) is not None
                    ]
                ),
            }
        payload[model_key] = {"included_samples": included, "arms": arm_payload}
    return payload


def target_failure_summary(
    rows: list[dict[str, JsonValue]], arms: list[Arm], model_keys: list[ModelKey]
) -> dict[str, JsonValue]:
    payload: dict[str, JsonValue] = {}
    for model_key in model_keys:
        by_arm: dict[str, JsonValue] = {}
        for arm in arms:
            failures = [
                {
                    "sample_idx": row.get("sample_idx"),
                    "error_type": row.get("target_error_type"),
                    "error_message": row.get("target_error_message"),
                }
                for row in rows
                if row["model_key"] == model_key
                and row["arm"] == arm
                and not row_target_succeeded(row)
            ]
            by_arm[arm] = {"target_failures": len(failures), "failures": failures}
        payload[model_key] = by_arm
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--samples", nargs="+", required=True)
    parser.add_argument("--models", nargs="+", default=["gpt_5_5", "opus_4_8"])
    parser.add_argument("--arms", nargs="+", default=["plain_pdf"])
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-rubrics-per-call", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    samples = load_samples(parse_sample_indices(args.samples), output_dir)
    arms = parse_arms(args.arms)
    model_keys = parse_model_keys(args.models)
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(
                run_grade,
                sample=sample,
                arm=arm,
                model_key=model_key,
                output_dir=output_dir,
                force=args.force,
                max_rubrics_per_call=args.max_rubrics_per_call,
            )
            for sample in samples
            for arm in arms
            for model_key in model_keys
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except (requests.RequestException, OSError, ValueError, json.JSONDecodeError) as exc:
                print(f"judge task failed error={type(exc).__name__}: {exc}", flush=True)
                raise
    build_summary(output_dir, samples, arms, model_keys)


if __name__ == "__main__":
    main()
