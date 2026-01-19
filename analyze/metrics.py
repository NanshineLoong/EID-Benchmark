"""Atomic metric computations."""

from __future__ import annotations

import re
from typing import Iterable, Optional

from analyze.models import CaseMetrics, CoveragePoint, GroundTruthFacts


def _find_fact_indices(content: str, role: str | None = None) -> set[str]:
    """Extract numbered fact indices like '1.' or '12.' from text.

    For patient role: only extract indices from [REFERENCE] section.
    For measurement role: extract all indices from the entire content.
    """
    if role == "patient":
        # Extract only from [REFERENCE] section
        ref_match = re.search(
            r"\[REFERENCE\]\s*(.+?)(?=\n\[RESPONSE\]|\n\n|$)", content, re.DOTALL
        )
        if ref_match:
            ref_content = ref_match.group(1).strip()
            if ref_content.upper() == "N/A":
                return set()
            return set(re.findall(r"(\d+)\.", ref_content))
        return set()
    # For measurement or unknown role, extract from entire content
    return set(re.findall(r"(\d+)\.", content))


def count_turns(trace: Iterable[dict]) -> int:
    """Count interaction turns (patient + measurement responses)."""
    return sum(1 for entry in trace if entry.get("role_id") in {"patient", "measurement"})


def compute_turn_time(trace: Iterable[dict]) -> tuple[int, float, float]:
    """Return (turns, total_duration, avg_duration_per_turn)."""
    turns = count_turns(trace)
    total_duration = 0.0
    for entry in trace:
        if entry.get("role_id") in {"patient", "measurement"}:
            continue
        duration = entry.get("duration")
        if duration is None:
            continue
        try:
            total_duration += float(duration)
        except (TypeError, ValueError):
            continue
    avg_duration = total_duration / turns if turns else 0.0
    return turns, total_duration, avg_duration


def compute_case_metrics(
    trace: list[dict],
    ground_truth: GroundTruthFacts,
    success: Optional[bool] = None,
    case_id: str | None = None,
) -> CaseMetrics:
    """Compute coverage and turn stats for a single case."""
    collected_patient: set[str] = set()
    collected_exam: set[str] = set()
    coverage_points: list[CoveragePoint] = []
    turn = 0

    patient_total = ground_truth.patient_total
    exam_total = ground_truth.exam_total

    for entry in trace:
        role = entry.get("role_id")
        if role not in {"patient", "measurement"}:
            continue
        turn += 1
        indices = _find_fact_indices(entry.get("content", ""), role=role)
        # Match indices based on role to avoid conflicts when indices overlap
        if role == "patient":
            collected_patient.update(
                idx for idx in indices if idx in ground_truth.patient_fact_ids
            )
        elif role == "measurement":
            collected_exam.update(
                idx for idx in indices if idx in ground_truth.exam_fact_ids
            )

        patient_cov = len(collected_patient) / patient_total if patient_total else 1.0
        exam_cov = len(collected_exam) / exam_total if exam_total else 1.0
        coverage_points.append(
            CoveragePoint(
                turn=turn,
                patient_coverage=patient_cov,
                exam_coverage=exam_cov,
            )
        )

    final_patient_cov = len(collected_patient) / patient_total if patient_total else 1.0
    final_exam_cov = len(collected_exam) / exam_total if exam_total else 1.0

    return CaseMetrics(
        case_id=case_id or "",
        turns=turn,
        patient_coverage=final_patient_cov,
        exam_coverage=final_exam_cov,
        coverage_by_turn=coverage_points,
        success=success,
    )


def average_curve(
    points: list[CoveragePoint], max_turns: int
) -> list[tuple[int, float, float]]:
    """Carry-forward coverage curve up to max_turns for a single case."""
    curve = []
    last_patient = 0.0
    last_exam = 0.0
    by_turn = {p.turn: p for p in points}
    for turn in range(1, max_turns + 1):
        point = by_turn.get(turn)
        if point:
            last_patient = point.patient_coverage
            last_exam = point.exam_coverage
        curve.append((turn, last_patient, last_exam))
    return curve


def merge_curves(
    cases: list[CaseMetrics], max_turns: int
) -> tuple[list[float], list[float]]:
    """Average per-turn coverage curves using carry-forward."""
    patient_curve = []
    exam_curve = []
    case_curves = [average_curve(case.coverage_by_turn, max_turns) for case in cases]
    for turn in range(1, max_turns + 1):
        patient_vals = []
        exam_vals = []
        for curve in case_curves:
            _, p, e = curve[turn - 1]
            patient_vals.append(p)
            exam_vals.append(e)
        patient_curve.append(
            sum(patient_vals) / len(patient_vals) if patient_vals else 0.0
        )
        exam_curve.append(sum(exam_vals) / len(exam_vals) if exam_vals else 0.0)
    return patient_curve, exam_curve
