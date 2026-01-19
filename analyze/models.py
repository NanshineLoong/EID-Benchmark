"""Shared data models for analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

MODE_DIR_ALIASES = {
    # Backward compatibility with stored results folder names
    "role_play": "roleplay",
}


@dataclass
class CoveragePoint:
    """Coverage snapshot after a specific turn."""

    turn: int
    patient_coverage: float
    exam_coverage: float


@dataclass
class CaseMetrics:
    """Per-case metrics derived from a trace and ground truth facts."""

    case_id: str
    turns: int
    patient_coverage: float
    exam_coverage: float
    coverage_by_turn: list[CoveragePoint]
    success: Optional[bool] = None


@dataclass
class RunSpec:
    """Identifies a single run to analyze."""

    dataset: str
    mode: str
    model: str
    max_turns: Optional[int] = None
    method: Optional[str] = None
    label: Optional[str] = None
    results_root: Path = Path("results")
    dataset_path: Optional[Path] = None

    @classmethod
    def from_mapping(cls, mapping: dict) -> "RunSpec":
        """Build a spec from a dictionary (e.g., parsed CLI input)."""
        max_turns = mapping.get("max_turns")
        return cls(
            dataset=mapping["dataset"],
            mode=mapping["mode"],
            model=mapping["model"],
            max_turns=int(max_turns) if max_turns is not None else None,
            method=mapping.get("method"),
            label=mapping.get("label"),
            results_root=Path(mapping.get("results_root", "results")),
            dataset_path=Path(mapping["dataset_path"])
            if mapping.get("dataset_path")
            else None,
        )

    @property
    def run_dir(self) -> Path:
        """Resolve the directory containing summary/record files."""
        mode_dir = MODE_DIR_ALIASES.get(self.mode, self.mode)
        base = self.results_root / self.dataset / mode_dir / self.model
        if self.max_turns:
            return base / f"{self.max_turns}_turns"
        return base

    @property
    def summary_path(self) -> Path:
        """Path to the summary JSON file."""
        return self.run_dir / "result.json"

    @property
    def record_dir(self) -> Path:
        """Directory with per-case JSON traces."""
        return self.run_dir / "record"

    @property
    def method_label(self) -> str:
        """Human readable label for plotting/table grouping."""
        return self.method or self.mode

    @property
    def display_label(self) -> str:
        """Label used in scatter/line plots."""
        return self.label or f"{self.method_label}/{self.model}"


@dataclass
class RunResult:
    """Aggregated metrics for a run."""

    spec: RunSpec
    case_metrics: list[CaseMetrics]
    success_rate: Optional[float]
    avg_patient_coverage: float
    avg_exam_coverage: float
    avg_turns: float
    coverage_curve_patient: list[float]
    coverage_curve_exam: list[float]
    success_count: int
    total_cases: int


@dataclass
class GroundTruthFacts:
    """Ground truth fact indices for a case."""

    patient_fact_ids: set[str]
    exam_fact_ids: set[str]

    @property
    def patient_total(self) -> int:
        return len(self.patient_fact_ids)

    @property
    def exam_total(self) -> int:
        return len(self.exam_fact_ids)


GroundTruthStore = dict[str, GroundTruthFacts]


def iter_case_ids(run_results: Iterable[RunResult]) -> set[str]:
    """Return all case ids present in the provided runs."""
    ids: set[str] = set()
    for run in run_results:
        ids.update(cm.case_id for cm in run.case_metrics)
    return ids
