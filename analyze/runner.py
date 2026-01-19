"""High-level helpers to load runs and assemble tabular metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from analyze.loaders import (
    DEFAULT_DATASET_FILES,
    iter_record_files,
    load_atomic_facts,
    load_summary_results,
)
from analyze.metrics import compute_case_metrics, compute_turn_time, merge_curves
from analyze.models import CaseMetrics, GroundTruthFacts, RunResult, RunSpec


def _resolve_dataset_path(spec: RunSpec) -> Path:
    if spec.dataset_path:
        return spec.dataset_path
    key = spec.dataset.lower()
    if key in DEFAULT_DATASET_FILES:
        return DEFAULT_DATASET_FILES[key]
    raise FileNotFoundError(f"No dataset path provided for {spec.dataset}")


def load_run_results(spec: RunSpec) -> RunResult:
    """Load one run and compute aggregated metrics."""
    dataset_path = _resolve_dataset_path(spec)
    ground_truth = load_atomic_facts(dataset_path)
    summary, summary_meta = load_summary_results(spec.summary_path)
    top1_accuracy = summary_meta.get("top1_accuracy")
    total_items_reported = summary_meta.get("total_items")

    summary_case_ids = set(summary.keys())
    case_metrics: list[CaseMetrics] = []
    seen_ids: set[str] = set()
    for case_id, payload in iter_record_files(spec.record_dir):
        trace = payload.get("trace") or []
        gt = ground_truth.get(case_id) or GroundTruthFacts(set(), set())
        cm = compute_case_metrics(
            trace=trace,
            ground_truth=gt,
            success=summary.get(case_id),
            case_id=case_id,
        )
        case_metrics.append(cm)
        seen_ids.add(case_id)

    # Fill in cases that appear in the summary but have no trace file
    missing_ids = summary_case_ids - seen_ids
    for case_id in sorted(missing_ids):
        gt = ground_truth.get(case_id) or GroundTruthFacts(set(), set())
        case_metrics.append(
            compute_case_metrics(
                trace=[],
                ground_truth=gt,
                success=summary.get(case_id),
                case_id=case_id,
            )
        )

    total_cases = len(case_metrics)
    success_cases = [cm for cm in case_metrics if cm.success is not None]
    success_count = sum(1 for cm in success_cases if cm.success)
    # Prefer summary's reported accuracy to match result.json exactly when provided
    if top1_accuracy is not None and total_items_reported:
        success_rate = top1_accuracy
        success_count = int(round(top1_accuracy * total_items_reported))
    else:
        success_rate = success_count / len(success_cases) if success_cases else None

    avg_patient_cov = (
        sum(cm.patient_coverage for cm in case_metrics) / total_cases
        if total_cases
        else 0.0
    )
    avg_exam_cov = (
        sum(cm.exam_coverage for cm in case_metrics) / total_cases if total_cases else 0.0
    )
    avg_turns = (
        sum(cm.turns for cm in case_metrics) / total_cases if total_cases else 0.0
    )
    max_turns = spec.max_turns or max((cm.turns for cm in case_metrics), default=0)
    patient_curve, exam_curve = (
        ([], []) if max_turns == 0 else merge_curves(case_metrics, max_turns)
    )

    return RunResult(
        spec=spec,
        case_metrics=case_metrics,
        success_rate=success_rate,
        avg_patient_coverage=avg_patient_cov,
        avg_exam_coverage=avg_exam_cov,
        avg_turns=avg_turns,
        coverage_curve_patient=patient_curve,
        coverage_curve_exam=exam_curve,
        success_count=success_count,
        total_cases=total_cases,
    )


def load_runs(specs: Iterable[RunSpec]) -> list[RunResult]:
    """Load multiple runs."""
    return [load_run_results(spec) for spec in specs]


def build_summary_rows(run_results: Iterable[RunResult]) -> list[dict]:
    """Flatten run metrics for table output."""
    rows: list[dict] = []
    for result in run_results:
        rows.append(
            {
                "dataset": result.spec.dataset,
                "mode": result.spec.mode,
                "method": result.spec.method_label,
                "model": result.spec.model,
                "max_turns": result.spec.max_turns,
                "success_rate": result.success_rate,
                "patient_coverage": result.avg_patient_coverage,
                "exam_coverage": result.avg_exam_coverage,
                "info_coverage": (result.avg_patient_coverage + result.avg_exam_coverage)
                / 2,
                "avg_turns": result.avg_turns,
                "cases": result.total_cases,
            }
        )
    return rows


def build_turn_rows(run_results: Iterable[RunResult]) -> list[dict]:
    """Return average interaction turns per dataset/mode/model."""
    rows: list[dict] = []
    for result in run_results:
        rows.append(
            {
                "dataset": result.spec.dataset,
                "mode": result.spec.mode,
                "method": result.spec.method_label,
                "model": result.spec.model,
                "max_turns": result.spec.max_turns,
                "avg_turns": result.avg_turns,
                "cases": result.total_cases,
            }
        )
    return rows


def build_turn_time_rows(specs: Iterable[RunSpec]) -> list[dict]:
    """Return average per-turn duration per dataset/mode/model."""
    rows: list[dict] = []
    for spec in specs:
        case_count = 0
        total_turns = 0
        total_duration = 0.0
        total_avg = 0.0
        for _, payload in iter_record_files(spec.record_dir):
            trace = payload.get("trace") or []
            turns, duration, avg_duration = compute_turn_time(trace)
            case_count += 1
            total_turns += turns
            total_duration += duration
            total_avg += avg_duration
        avg_turn_time = total_avg / case_count if case_count else 0.0
        rows.append(
            {
                "dataset": spec.dataset,
                "mode": spec.mode,
                "method": spec.method_label,
                "model": spec.model,
                "max_turns": spec.max_turns,
                "avg_turn_time": avg_turn_time,
                "cases": case_count,
                "total_turns": total_turns,
                "total_duration": total_duration,
            }
        )
    return rows


def build_turn_time_mode_rows(rows: Iterable[dict]) -> list[dict]:
    """Aggregate per-turn duration across datasets for each mode/model."""
    grouped: dict[tuple[str, str, str, int | None], dict] = {}
    for row in rows:
        key = (row.get("mode"), row.get("method"), row.get("model"), row.get("max_turns"))
        entry = grouped.setdefault(
            key,
            {
                "mode": row.get("mode"),
                "method": row.get("method"),
                "model": row.get("model"),
                "max_turns": row.get("max_turns"),
                "avg_turn_time": 0.0,
                "cases": 0,
                "total_turns": 0,
                "total_duration": 0.0,
                "datasets": set(),
            },
        )
        cases = int(row.get("cases") or 0)
        entry["avg_turn_time"] += (row.get("avg_turn_time") or 0.0) * cases
        entry["cases"] += cases
        entry["total_turns"] += int(row.get("total_turns") or 0)
        entry["total_duration"] += float(row.get("total_duration") or 0.0)
        entry["datasets"].add(row.get("dataset"))
    results: list[dict] = []
    for entry in grouped.values():
        cases = entry["cases"]
        avg_turn_time = entry["avg_turn_time"] / cases if cases else 0.0
        results.append(
            {
                "mode": entry["mode"],
                "method": entry["method"],
                "model": entry["model"],
                "max_turns": entry["max_turns"],
                "avg_turn_time": avg_turn_time,
                "cases": cases,
                "total_turns": entry["total_turns"],
                "total_duration": entry["total_duration"],
                "datasets": len(entry["datasets"]),
            }
        )
    return results


def coverage_by_outcome(run: RunResult) -> dict[str, float]:
    """Average coverage split by success/failure."""
    success_cov = [cm for cm in run.case_metrics if cm.success is True]
    failure_cov = [cm for cm in run.case_metrics if cm.success is False]
    return {
        "success_patient": sum(cm.patient_coverage for cm in success_cov) / len(success_cov)
        if success_cov
        else 0.0,
        "success_exam": sum(cm.exam_coverage for cm in success_cov) / len(success_cov)
        if success_cov
        else 0.0,
        "failure_patient": sum(cm.patient_coverage for cm in failure_cov) / len(failure_cov)
        if failure_cov
        else 0.0,
        "failure_exam": sum(cm.exam_coverage for cm in failure_cov) / len(failure_cov)
        if failure_cov
        else 0.0,
    }


def build_ablation_rows(
    run_results: Iterable[RunResult],
    baseline_method: str,
) -> list[dict]:
    """Compute delta vs baseline method for each dataset/model pair.

    Delta values are converted to percentage points (0-100 scale).
    """
    baseline_map: dict[tuple[str, str], RunResult] = {}
    rows: list[dict] = []
    # First pass: collect baselines (match by mode, not method_label)
    for run in run_results:
        key = (run.spec.dataset, run.spec.model)
        if run.spec.mode == baseline_method:
            baseline_map[key] = run
    # Second pass: compute deltas for non-baseline methods
    for run in run_results:
        key = (run.spec.dataset, run.spec.model)
        if run.spec.mode == baseline_method:
            continue
        baseline = baseline_map.get(key)
        if not baseline:
            continue
        # Convert deltas to percentage points (0-100 scale)
        rows.append(
            {
                "dataset": run.spec.dataset,
                "model": run.spec.model,
                "method": run.spec.method_label,
                "baseline": baseline_method,
                "delta_patient_cov": (run.avg_patient_coverage - baseline.avg_patient_coverage)
                * 100,
                "delta_exam_cov": (run.avg_exam_coverage - baseline.avg_exam_coverage) * 100,
                "delta_info_cov": (
                    (run.avg_patient_coverage + run.avg_exam_coverage) / 2
                    - (baseline.avg_patient_coverage + baseline.avg_exam_coverage) / 2
                )
                * 100,
                "delta_success": ((run.success_rate or 0) - (baseline.success_rate or 0))
                * 100,
            }
        )
    return rows
