"""Metrics module for evaluation."""

from eid.metrics.evaluators import (
    Metric,
    ExactMatchMetric,
    LLMJudgeMetric,
    get_metric,
)
from eid.metrics.coverage import (
    GroundTruthFacts,
    CoverageResult,
    extract_fact_ids,
    find_fact_indices_in_content,
    compute_coverage,
)

__all__ = [
    "Metric",
    "ExactMatchMetric",
    "LLMJudgeMetric",
    "get_metric",
    "GroundTruthFacts",
    "CoverageResult",
    "extract_fact_ids",
    "find_fact_indices_in_content",
    "compute_coverage",
]
