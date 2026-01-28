"""Evaluation metrics for diagnosis accuracy."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

from eid.agents import create_agent

if TYPE_CHECKING:
    from eid.config import ModelConfig

logger = logging.getLogger(__name__)


class Metric(ABC):
    """Abstract base class for evaluation metrics."""

    @abstractmethod
    def compare(self, prediction: str, ground_truth: str) -> tuple[dict[str, Any], bool]:
        """Compare prediction with ground truth.

        Args:
            prediction: Model's prediction/diagnosis
            ground_truth: Ground truth answer

        Returns:
            Tuple of (result_dict, is_correct)
        """
        pass

    def summarize(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Summarize evaluation results.

        Args:
            results: List of result dictionaries from compare()

        Returns:
            Summary statistics dictionary
        """
        if not results:
            return {
                "total_items": 0,
                "top1_accuracy": 0.0,
                "avg_information_coverage_rate": 0.0,
                "avg_patient_coverage": 0.0,
                "avg_exam_coverage": 0.0,
            }

        correct_count = sum(1 for r in results if r.get("is_correct", False))
        total = len(results)

        avg_info_coverage = sum(
            r.get("information_coverage_rate", 0.0) for r in results
        ) / total
        avg_patient_coverage = sum(
            r.get("patient_coverage", 0.0) for r in results
        ) / total
        avg_exam_coverage = sum(
            r.get("exam_coverage", 0.0) for r in results
        ) / total

        return {
            "total_items": total,
            "correct_count": correct_count,
            "top1_accuracy": correct_count / total if total > 0 else 0.0,
            "avg_information_coverage_rate": avg_info_coverage,
            "avg_patient_coverage": avg_patient_coverage,
            "avg_exam_coverage": avg_exam_coverage,
            "results": results,
        }


class ExactMatchMetric(Metric):
    """Simple exact match metric (case-insensitive)."""

    def compare(self, prediction: str, ground_truth: str) -> tuple[dict[str, Any], bool]:
        """Compare using exact string match (case-insensitive)."""
        pred_clean = prediction.strip().lower()
        gt_clean = ground_truth.strip().lower()

        is_correct = pred_clean == gt_clean

        return {
            "prediction": prediction,
            "ground_truth": ground_truth,
            "is_correct": is_correct,
        }, is_correct


class LLMJudgeMetric(Metric):
    """LLM-based evaluation metric using a judge model.

    Uses an LLM to compare the prediction against ground truth,
    useful for semantic matching of diagnoses.
    """

    def __init__(
        self,
        judge_config: "ModelConfig",
        task_type: str = "diagnosis",
    ) -> None:
        """Initialize LLM judge metric.

        Args:
            judge_config: Model configuration for the judge
            task_type: Type of task (diagnosis or differential_diagnosis)
        """
        self.judge_config = judge_config
        self.task_type = task_type
        self._judge: Any = None

    def _get_judge(self) -> Any:
        """Lazy initialization of judge agent."""
        if self._judge is None:
            system_prompt = self._get_judge_system_prompt()
            self._judge = create_agent(
                role_id="judge",
                system_prompt=system_prompt,
                config=self.judge_config,
            )
        return self._judge

    def _get_judge_system_prompt(self) -> str:
        """Get system prompt for judge based on task type."""
        if self.task_type == "differential_diagnosis":
            return (
                "You are an expert clinician evaluating differential diagnoses. "
                "You will compare a predicted diagnosis list against a reference diagnosis.\n\n"
                "For each predicted diagnosis, score it as follows:\n"
                "- Score 2: Exact match or clinically equivalent\n"
                "- Score 1: Related condition (parent/child relationship, similar presentation)\n"
                "- Score 0: Unrelated or incorrect\n\n"
                "Output format:\n"
                "Diagnosis 1: [prediction] -> Score [0/1/2]\n"
                "...\n"
                "TOP1_CORRECT: [YES/NO]\n"
                "TOP5_CORRECT: [YES/NO]"
            )
        else:
            return (
                "You are an expert clinician evaluating a medical diagnosis. "
                "Compare the predicted diagnosis against the reference diagnosis.\n\n"
                "Determine if the prediction is:\n"
                "- CORRECT: Exact match or clinically equivalent diagnosis\n"
                "- INCORRECT: Different or wrong diagnosis\n\n"
                "Output format:\n"
                "ANALYSIS: [Brief explanation]\n"
                "VERDICT: [CORRECT/INCORRECT]"
            )

    def compare(self, prediction: str, ground_truth: str) -> tuple[dict[str, Any], bool]:
        """Compare using LLM judge."""
        judge = self._get_judge()

        prompt = (
            f"Reference Diagnosis:\n{ground_truth}\n\n"
            f"Predicted Diagnosis:\n{prediction}\n\n"
            "Please evaluate the prediction."
        )

        try:
            response = judge.step(prompt)

            if self.task_type == "differential_diagnosis":
                is_correct = self._parse_differential_response(response)
            else:
                is_correct = self._parse_diagnosis_response(response)

            return {
                "prediction": prediction,
                "ground_truth": ground_truth,
                "judge_response": response,
                "is_correct": is_correct,
            }, is_correct

        except Exception as e:
            logger.warning(f"LLM judge failed: {e}")
            # Fallback to simple string matching
            is_correct = ground_truth.lower() in prediction.lower()
            return {
                "prediction": prediction,
                "ground_truth": ground_truth,
                "judge_error": str(e),
                "is_correct": is_correct,
            }, is_correct

    def _parse_diagnosis_response(self, response: str) -> bool:
        """Parse judge response for diagnosis task."""
        response_upper = response.upper()
        if "VERDICT:" in response_upper:
            if "CORRECT" in response_upper.split("VERDICT:")[-1]:
                return "INCORRECT" not in response_upper.split("VERDICT:")[-1].split("\n")[0]
        return "CORRECT" in response_upper and "INCORRECT" not in response_upper

    def _parse_differential_response(self, response: str) -> bool:
        """Parse judge response for differential diagnosis task."""
        response_upper = response.upper()
        # Check TOP1_CORRECT
        if "TOP1_CORRECT:" in response_upper:
            top1_line = response_upper.split("TOP1_CORRECT:")[-1].split("\n")[0]
            return "YES" in top1_line
        # Fallback: check for any Score 2 in first diagnosis
        score_match = re.search(r"DIAGNOSIS\s*1.*SCORE\s*2", response_upper)
        return score_match is not None


def get_metric(
    dataset_name: str,
    judge_config: "ModelConfig | None" = None,
) -> Metric:
    """Get appropriate metric for a dataset.

    Args:
        dataset_name: Name of the dataset
        judge_config: Optional model config for LLM judge

    Returns:
        Configured Metric instance
    """
    # Datasets that need LLM judge for semantic matching
    llm_judge_datasets = {
        "diagnosisarena",
        "clinicalbench",
        "rarearena",
        "medqa",
        "derm",
    }

    if dataset_name.lower() in llm_judge_datasets and judge_config is not None:
        task_type = (
            "differential_diagnosis"
            if dataset_name.lower() in {"diagnosisarena", "clinicalbench", "rarearena"}
            else "diagnosis"
        )
        return LLMJudgeMetric(judge_config=judge_config, task_type=task_type)

    return ExactMatchMetric()
