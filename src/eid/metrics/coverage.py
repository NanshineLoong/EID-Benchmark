"""Information coverage metrics for diagnostic conversations."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GroundTruthFacts:
    """Ground truth fact indices for a case."""

    patient_fact_ids: set[str] = field(default_factory=set)
    exam_fact_ids: set[str] = field(default_factory=set)

    @property
    def patient_total(self) -> int:
        return len(self.patient_fact_ids)

    @property
    def exam_total(self) -> int:
        return len(self.exam_fact_ids)

    @property
    def total(self) -> int:
        return self.patient_total + self.exam_total


@dataclass
class CoverageResult:
    """Coverage computation result for a single case."""

    patient_coverage: float
    exam_coverage: float
    information_coverage_rate: float
    collected_patient_facts: list[str]
    collected_exam_facts: list[str]
    ground_truth_patient_facts: list[str]
    ground_truth_exam_facts: list[str]


def extract_fact_ids(facts: list[str]) -> set[str]:
    """Extract fact indices from a list of fact strings.

    Args:
        facts: List of facts like ["1. Fact text", "2. Another fact"]

    Returns:
        Set of indices as strings: {"1", "2"}
    """
    ids: set[str] = set()
    for fact in facts:
        match = re.match(r"\s*(\d+)\.", fact)
        if match:
            ids.add(match.group(1))
    return ids


def find_fact_indices_in_content(content: str, role: str | None = None) -> set[str]:
    """Extract numbered fact indices from text content.

    For patient role: only extract indices from [REFERENCE] section.
    For reporter/measurement role: extract all indices from the entire content.

    Args:
        content: Text content to search
        role: Role identifier (patient, reporter, measurement)

    Returns:
        Set of fact indices found
    """
    if role == "patient":
        ref_match = re.search(
            r"\[REFERENCE\]\s*(.+?)(?=\n\[RESPONSE\]|\n\n|$)", content, re.DOTALL
        )
        if ref_match:
            ref_content = ref_match.group(1).strip()
            if ref_content.upper() == "N/A":
                return set()
            return set(re.findall(r"(\d+)\.", ref_content))
        return set()
    return set(re.findall(r"(\d+)\.", content))


def compute_coverage(
    trace: list[dict[str, Any]],
    patient_facts: list[str],
    exam_facts: list[str],
) -> CoverageResult:
    """Compute information coverage from a conversation trace.

    Args:
        trace: List of conversation turns with role_id and content
        patient_facts: Ground truth patient facts
        exam_facts: Ground truth exam/measurement facts

    Returns:
        CoverageResult with coverage metrics and collected facts
    """
    ground_truth = GroundTruthFacts(
        patient_fact_ids=extract_fact_ids(patient_facts),
        exam_fact_ids=extract_fact_ids(exam_facts),
    )

    collected_patient_ids: set[str] = set()
    collected_exam_ids: set[str] = set()

    for entry in trace:
        role = entry.get("role_id", "")
        content = entry.get("content", "")

        if role == "patient":
            indices = find_fact_indices_in_content(content, role="patient")
            collected_patient_ids.update(
                idx for idx in indices if idx in ground_truth.patient_fact_ids
            )
        elif role in {"reporter", "measurement"}:
            indices = find_fact_indices_in_content(content, role=role)
            collected_exam_ids.update(
                idx for idx in indices if idx in ground_truth.exam_fact_ids
            )

    patient_total = ground_truth.patient_total
    exam_total = ground_truth.exam_total

    patient_coverage = (
        len(collected_patient_ids) / patient_total if patient_total else 1.0
    )
    exam_coverage = len(collected_exam_ids) / exam_total if exam_total else 1.0

    total_collected = len(collected_patient_ids) + len(collected_exam_ids)
    total_facts = ground_truth.total
    information_coverage_rate = total_collected / total_facts if total_facts else 1.0

    collected_patient_facts = [
        f for f in patient_facts
        if any(re.match(rf"\s*{idx}\.", f) for idx in collected_patient_ids)
    ]
    collected_exam_facts = [
        f for f in exam_facts
        if any(re.match(rf"\s*{idx}\.", f) for idx in collected_exam_ids)
    ]

    return CoverageResult(
        patient_coverage=patient_coverage,
        exam_coverage=exam_coverage,
        information_coverage_rate=information_coverage_rate,
        collected_patient_facts=collected_patient_facts,
        collected_exam_facts=collected_exam_facts,
        ground_truth_patient_facts=patient_facts,
        ground_truth_exam_facts=exam_facts,
    )
