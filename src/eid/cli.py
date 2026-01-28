"""Command-line interface for EID-Benchmark evaluations."""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

AVAILABLE_MODES = ["cot", "roleplay", "react", "sc", "refine"]


def main(argv: list[str] | None = None) -> None:
    """Main entry point for CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load environment
    load_dotenv()

    try:
        run_evaluations(args)
    except Exception as exc:
        logger.error("Evaluation failed: %s", exc, exc_info=True)
        sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        description="EID-Benchmark: Evaluate LLM diagnostic capabilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic CoT evaluation
  eid-eval --datasets medqa --modes cot --doctor-model gpt-5-mini

  # Multi-turn roleplay with different models for different roles
  eid-eval --datasets diagnosisarena clinicalbench \\
           --modes react refine \\
           --doctor-model gpt-5-mini \\
           --patient-model gpt-5-mini \\
           --max-turns 16

  # Full evaluation with parallel workers
  eid-eval --datasets medqa diagnosisarena \\
           --modes cot roleplay react sc refine \\
           --doctor-model gpt-5-mini \\
           --max-turns 8 12 16 \\
           --max-workers 50

  # SC/REFINE with separate models for summarizer/diagnostician/verifier
  eid-eval --datasets diagnosisarena \\
           --modes sc refine \\
           --doctor-model gpt-5-mini \\
           --summarizer-model gpt-5-mini \\
           --diagnostician-model gpt-5-mini \\
           --verifier-model gpt-5-mini \\
           --max-turns 16
        """,
    )

    # Dataset arguments
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Datasets to evaluate (medqa, diagnosisarena, clinicalbench, rarearena, derm)",
    )
    parser.add_argument(
        "--dataset-path",
        help="Custom path to dataset file (overrides default)",
    )

    # Mode arguments
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=AVAILABLE_MODES,
        default=["cot"],
        help="Evaluation modes (default: cot)",
    )

    # Model arguments - main roles
    parser.add_argument(
        "--doctor-model",
        nargs="+",
        required=True,
        help="Model(s) for doctor role (e.g., gpt-5-mini)",
    )
    parser.add_argument(
        "--patient-model",
        help="Model for patient simulator (default: same as doctor)",
    )
    parser.add_argument(
        "--reporter-model",
        help="Model for reporter simulator (default: same as doctor)",
    )
    parser.add_argument(
        "--annotator-model",
        help="Model for evaluation judge (default: same as doctor)",
    )

    # SC/REFINE specific roles
    parser.add_argument(
        "--summarizer-model",
        help="Model for summarizer role in SC/REFINE modes (default: same as doctor)",
    )
    parser.add_argument(
        "--diagnostician-model",
        help="Model for diagnostician role in SC/REFINE modes (default: same as doctor)",
    )
    parser.add_argument(
        "--verifier-model",
        help="Model for verifier role in REFINE mode (default: same as doctor)",
    )

    # Execution arguments
    parser.add_argument(
        "--max-items",
        type=int,
        default=200,
        help="Maximum items per dataset (default: 200)",
    )
    parser.add_argument(
        "--max-turns",
        nargs="+",
        type=int,
        default=[16],
        help="Maximum turns for roleplay modes (default: 16)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum parallel workers for concurrent evaluation ",
    )
    parser.add_argument(
        "--task-workers",
        type=int,
        default=1,
        help="Parallel evaluation tasks across different configs (default: 1)",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Output directory (default: results)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip if result.json already exists",
    )

    return parser


def run_evaluations(args: argparse.Namespace) -> None:
    """Run evaluations based on parsed arguments."""
    from eid.benchmark import run_evaluation

    # Build task list
    tasks: list[dict[str, Any]] = []

    for dataset in args.datasets:
        for doctor_model in args.doctor_model:
            for mode in args.modes:
                if mode == "cot":
                    # CoT doesn't vary by max_turns
                    tasks.append({
                        "dataset": dataset,
                        "mode": mode,
                        "doctor_model": doctor_model,
                        "max_turns": None,
                    })
                else:
                    # Roleplay modes vary by max_turns
                    for max_turns in args.max_turns:
                        tasks.append({
                            "dataset": dataset,
                            "mode": mode,
                            "doctor_model": doctor_model,
                            "max_turns": max_turns,
                        })

    logger.info("Total tasks: %d", len(tasks))

    def run_task(task: dict[str, Any]) -> None:
        """Run a single evaluation task."""
        dataset = task["dataset"]
        mode = task["mode"]
        doctor_model = task["doctor_model"]
        max_turns = task["max_turns"]

        # Build output path and check for existing
        model_name = doctor_model.replace("/", "_")
        output_path = Path(args.output_dir) / dataset / mode / model_name
        if max_turns is not None:
            output_path = output_path / f"{max_turns}_turns"
        result_path = output_path / "result.json"

        if args.skip_existing and result_path.exists():
            logger.info("Skipping (exists): %s", result_path)
            return

        turns_msg = f", turns={max_turns}" if max_turns else ""
        logger.info("Running: %s / %s / %s%s", dataset, mode, doctor_model, turns_msg)

        try:
            summary = run_evaluation(
                dataset_name=dataset,
                mode=mode,
                doctor_model=doctor_model,
                patient_model=args.patient_model,
                reporter_model=args.reporter_model,
                annotator_model=args.annotator_model,
                summarizer_model=args.summarizer_model,
                diagnostician_model=args.diagnostician_model,
                verifier_model=args.verifier_model,
                max_items=args.max_items,
                max_turns=max_turns or 16,
                max_workers=args.max_workers,
                output_dir=args.output_dir,
                dataset_path=args.dataset_path,
            )
            logger.info(
                "Completed: %s / %s / %s - Accuracy: %.2f%%",
                dataset,
                mode,
                doctor_model,
                summary.get("top1_accuracy", 0) * 100,
            )
        except Exception as e:
            logger.error(
                "Failed: %s / %s / %s - %s",
                dataset,
                mode,
                doctor_model,
                e,
                exc_info=True,
            )

    # Run tasks
    if args.task_workers > 1:
        with ThreadPoolExecutor(max_workers=args.task_workers) as executor:
            futures = [executor.submit(run_task, task) for task in tasks]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error("Task failed: %s", e)
    else:
        for task in tasks:
            run_task(task)

    logger.info("All evaluations completed")


if __name__ == "__main__":
    main()
