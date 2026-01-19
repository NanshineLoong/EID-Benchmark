"""Main Benchmark class for running evaluations."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

from eid.config import ModelConfig
from eid.datasets import Dataset, DataItem, load_dataset
from eid.metrics import Metric, get_metric
from eid.scenarios import get_scenario
from eid.scenarios.base import BaseScenario, CaseInput, ScenarioResult

logger = logging.getLogger(__name__)


class Benchmark:
    """Main benchmark runner for evaluating LLM diagnostic capabilities.

    Coordinates dataset loading, scenario execution, and result evaluation.
    """

    def __init__(
        self,
        dataset: Dataset,
        metric: Metric,
        output_dir: Path | str = "results",
        save_traces: bool = True,
    ) -> None:
        """Initialize benchmark.

        Args:
            dataset: Dataset to evaluate on
            metric: Metric for evaluation
            output_dir: Directory for output files
            save_traces: Whether to save individual case traces
        """
        self.dataset = dataset
        self.metric = metric
        self.output_dir = Path(output_dir)
        self.save_traces = save_traces

    def evaluate(
        self,
        scenario: BaseScenario,
        max_workers: int = 10,
        summary_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Run evaluation on the dataset.

        Args:
            scenario: Scenario to execute
            max_workers: Maximum parallel workers
            summary_path: Path to save summary JSON

        Returns:
            Summary statistics dictionary
        """
        logger.info(
            "Starting evaluation: %d items, %d workers",
            len(self.dataset),
            max_workers,
        )

        evaluation_results: list[dict[str, Any]] = []

        # Create trace output directory
        trace_dir = self.output_dir / "record"
        if self.save_traces:
            trace_dir.mkdir(parents=True, exist_ok=True)

        # Run evaluation
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._run_single_item, scenario, item): item
                for item in self.dataset
            }

            for future in tqdm(
                as_completed(futures),
                total=len(self.dataset),
                desc="Evaluating",
                ncols=100,
            ):
                item = futures[future]
                try:
                    full_result = future.result()
                    # Extract evaluation result (without trace and metadata)
                    eval_result = self._extract_evaluation_result(full_result)
                    evaluation_results.append(eval_result)

                    # Save trace if enabled
                    if self.save_traces:
                        trace_path = trace_dir / f"{item.case_id}.json"
                        self._save_trace(trace_path, full_result)

                except Exception as e:
                    logger.error("Failed to evaluate %s: %s", item.case_id, e)
                    evaluation_results.append({
                        "case_id": item.case_id,
                        "error": str(e),
                        "is_correct": False,
                    })

        # Summarize results
        summary = self.metric.summarize(evaluation_results)
        summary["dataset"] = self.dataset.name
        summary["scenario"] = scenario.__class__.__name__

        # Save summary
        if summary_path:
            summary_path = Path(summary_path)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            logger.info("Summary saved to %s", summary_path)

        return summary

    def _run_single_item(
        self,
        scenario: BaseScenario,
        item: DataItem,
    ) -> dict[str, Any]:
        """Run scenario on a single item and evaluate.

        Args:
            scenario: Scenario to execute
            item: Data item to evaluate

        Returns:
            Full result dictionary including trace and metadata
        """
        # Build case input
        case_input = CaseInput(
            case_id=item.case_id,
            task=item.task,
            patient_facts=item.patient_facts,
            exam_facts=item.exam_facts,
            ground_truth=item.answer,
        )

        # Run scenario
        result = scenario.run(case_input)

        # Evaluate result
        eval_result, is_correct = self.metric.compare(result.answer, item.answer)

        return {
            "case_id": item.case_id,
            "prediction": result.answer,
            "ground_truth": item.answer,
            "is_correct": is_correct,
            "trace": result.trace,
            "metadata": result.metadata,
            **eval_result,
        }

    def _extract_evaluation_result(self, full_result: dict[str, Any]) -> dict[str, Any]:
        """Extract evaluation result without trace and metadata.

        Args:
            full_result: Full result dictionary with trace and metadata

        Returns:
            Evaluation result dictionary without trace and metadata
        """
        return {
            k: v
            for k, v in full_result.items()
            if k not in ("trace", "metadata")
        }

    def _save_trace(self, path: Path, result: dict[str, Any]) -> None:
        """Save trace to JSON file."""
        with path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)


def run_evaluation(
    dataset_name: str,
    mode: str,
    doctor_model: str,
    patient_model: str | None = None,
    measurement_model: str | None = None,
    annotator_model: str | None = None,
    summarizer_model: str | None = None,
    diagnostician_model: str | None = None,
    verifier_model: str | None = None,
    max_items: int | None = None,
    max_turns: int = 16,
    max_workers: int = 10,
    output_dir: str = "results",
    dataset_path: str | None = None,
) -> dict[str, Any]:
    """Convenience function to run a complete evaluation.

    Args:
        dataset_name: Name of the dataset
        mode: Evaluation mode (cot, roleplay, react, sc, refine)
        doctor_model: Model name for doctor
        patient_model: Model name for patient (optional)
        measurement_model: Model name for measurement (optional)
        annotator_model: Model name for annotator (optional)
        summarizer_model: Model name for summarizer (optional, SC/REFINE)
        diagnostician_model: Model name for diagnostician (optional, SC/REFINE)
        verifier_model: Model name for verifier (optional, REFINE)
        max_items: Maximum items to evaluate
        max_turns: Maximum interaction turns
        max_workers: Maximum parallel workers
        output_dir: Output directory
        dataset_path: Custom dataset path (optional)

    Returns:
        Summary statistics dictionary
    """
    from eid.config import load_config, ModelConfig

    # Load environment configuration
    load_config()

    # Load dataset
    dataset = load_dataset(dataset_name, path=dataset_path, max_items=max_items)

    # Create model configs
    doctor_config = ModelConfig.from_string(doctor_model)

    patient_config = (
        ModelConfig.from_string(patient_model) if patient_model else doctor_config
    )
    measurement_config = (
        ModelConfig.from_string(measurement_model) if measurement_model else doctor_config
    )
    annotator_config = (
        ModelConfig.from_string(annotator_model) if annotator_model else doctor_config
    )
    summarizer_config = (
        ModelConfig.from_string(summarizer_model) if summarizer_model else None
    )
    diagnostician_config = (
        ModelConfig.from_string(diagnostician_model) if diagnostician_model else None
    )
    verifier_config_obj = (
        ModelConfig.from_string(verifier_model) if verifier_model else None
    )

    # Create scenario
    scenario = get_scenario(
        mode=mode,
        dataset_name=dataset_name,
        doctor_config=doctor_config,
        patient_config=patient_config,
        measurement_config=measurement_config,
        max_turns=max_turns,
        summarizer_config=summarizer_config,
        diagnostician_config=diagnostician_config,
        verifier_config=verifier_config_obj,
    )

    # Create metric
    metric = get_metric(dataset_name, judge_config=annotator_config)

    # Build output path
    model_name = doctor_model.replace("/", "_")
    output_path = Path(output_dir) / dataset_name / mode / model_name
    if mode != "cot":
        output_path = output_path / f"{max_turns}_turns"

    # Run benchmark
    benchmark = Benchmark(
        dataset=dataset,
        metric=metric,
        output_dir=output_path,
        save_traces=True,
    )

    summary = benchmark.evaluate(
        scenario=scenario,
        max_workers=max_workers,
        summary_path=output_path / "result.json",
    )

    return summary
