"""CLI entry point for analysis experiments (E1-E4)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from analyze.models import RunSpec
from analyze.plots import (
    plot_coverage_curve,
    plot_outcome_bars,
    plot_turn_limit_trend,
    scatter_success_vs_coverage,
)
from analyze.runner import (
    build_ablation_rows,
    build_summary_rows,
    build_turn_rows,
    build_turn_time_mode_rows,
    build_turn_time_rows,
    load_runs,
)
from analyze.excel import rows_to_wide, export_excel


def _print_table(rows: List[dict]) -> None:
    if not rows:
        print("No rows to display.")
        return
    headers = list(rows[0].keys())
    col_widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            val = row.get(h, "")
            if isinstance(val, float):
                col_widths[h] = max(col_widths[h], len(f"{val:.4f}"))
            else:
                col_widths[h] = max(col_widths[h], len(str(val)))
    header_line = " | ".join(f"{h:<{col_widths[h]}}" for h in headers)
    sep_line = "-+-".join("-" * col_widths[h] for h in headers)
    print(header_line)
    print(sep_line)
    for row in rows:
        parts = []
        for h in headers:
            val = row.get(h, "")
            if isinstance(val, float):
                parts.append(f"{val:<{col_widths[h]}.4f}")
            else:
                parts.append(f"{str(val):<{col_widths[h]}}")
        print(" | ".join(parts))


def _expand_specs(args: argparse.Namespace) -> List[RunSpec]:
    """Build all run specs from list inputs (cartesian product)."""
    datasets = args.datasets
    modes = args.modes
    models = args.models
    max_turns_list = args.max_turns or [None]
    specs: List[RunSpec] = []
    for dataset in datasets:
        for mode in modes:
            for model in models:
                turns_iter = [None] if mode == "cot" else max_turns_list
                for mt in turns_iter:
                    specs.append(
                        RunSpec(
                            dataset=dataset,
                            mode=mode,
                            model=model,
                            max_turns=mt,
                            results_root=Path(args.results_root),
                            dataset_path=Path(args.dataset_path) if args.dataset_path else None,
                        )
                    )
    return specs


def _save_markdown(rows: List[dict], path: Path) -> None:
    if not rows:
        print("No rows to save.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        cells = []
        for h in headers:
            val = row.get(h, "")
            if isinstance(val, float):
                cells.append(f"{val:.1f}")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved markdown table to {path}")


def cmd_e1(args: argparse.Namespace) -> None:
    runs = load_runs(_expand_specs(args))
    rows = build_summary_rows(runs)
    _print_table(rows)
    if args.excel:
        wide = rows_to_wide(rows)
        export_excel(wide, Path(args.excel))
    fig_dir = Path(args.fig_dir) if args.fig_dir else None
    if fig_dir:
        scatter_success_vs_coverage(runs, fig_dir / "e1_scatter_all.png")
        for dataset in sorted({r.spec.dataset for r in runs}):
            scatter_success_vs_coverage(
                runs, fig_dir / f"e1_scatter_{dataset}.png", dataset=dataset
            )


def cmd_e2(args: argparse.Namespace) -> None:
    runs = load_runs(_expand_specs(args))
    metric_map = {
        "success": "success_rate",
        "patient": "patient_coverage",
        "exam": "exam_coverage",
        "info": "info_coverage",
    }
    metric = metric_map.get(args.metric, "success_rate")
    fig_dir = Path(args.fig_dir) if args.fig_dir else Path("figures")
    plot_turn_limit_trend(
        runs,
        output_path=fig_dir / f"e2_turn_limit_{metric}.png",
        metric=metric,
        dataset=args.dataset,
        model=args.model,
    )
    plot_coverage_curve(
        runs,
        output_path=fig_dir / f"e2_coverage_curve_{args.curve_metric}.png",
        dataset=args.dataset,
        model=args.model,
        metric=args.curve_metric,
    )


def cmd_e3(args: argparse.Namespace) -> None:
    runs = load_runs(_expand_specs(args))
    fig_dir = Path(args.fig_dir) if args.fig_dir else Path("figures")
    plot_outcome_bars(
        runs,
        output_path=fig_dir / "e3_outcome_coverage.png",
        dataset=args.dataset,
        model=args.model,
    )


def cmd_e4(args: argparse.Namespace) -> None:
    runs = load_runs(_expand_specs(args))
    rows = build_ablation_rows(runs, baseline_method=args.baseline)
    _print_table(rows)
    if args.md:
        _save_markdown(rows, Path(args.md))


def cmd_turns(args: argparse.Namespace) -> None:
    """Average interaction turns per dataset/mode/model."""
    runs = load_runs(_expand_specs(args))
    rows = build_turn_rows(runs)
    _print_table(rows)
    if args.md:
        _save_markdown(rows, Path(args.md))


def cmd_turn_time(args: argparse.Namespace) -> None:
    """Average per-turn duration by mode/model."""
    specs = _expand_specs(args)
    rows = build_turn_time_rows(specs)
    if not args.by_dataset:
        rows = build_turn_time_mode_rows(rows)
    _print_table(rows)
    if args.md:
        _save_markdown(rows, Path(args.md))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified analysis entry for experiment E1-E4")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--datasets", nargs="+", required=True, help="Datasets to include")
    common.add_argument("--modes", nargs="+", required=True, help="Modes to include")
    common.add_argument("--models", nargs="+", required=True, help="Models to include")
    common.add_argument(
        "--max-turns",
        nargs="+",
        type=int,
        help="Optional turn limits",
    )
    common.add_argument(
        "--results-root",
        default="results",
        help="Base directory containing result.json",
    )
    common.add_argument(
        "--dataset-path",
        help="Override dataset path",
    )

    p1 = sub.add_parser("e1", parents=[common], help="E1 overall evaluation + scatter plot")
    p1.add_argument("--fig-dir", default="figures", help="Figure output directory")
    p1.add_argument("--excel", help="Save summary to Excel")
    p1.set_defaults(func=cmd_e1)

    p2 = sub.add_parser("e2", parents=[common], help="E2 max-turn limit experiments")
    p2.add_argument("--metric", choices=["success", "patient", "exam", "info"], default="success")
    p2.add_argument("--curve-metric", choices=["patient", "exam"], default="patient")
    p2.add_argument("--dataset", help="Filter by dataset")
    p2.add_argument("--model", help="Filter by model")
    p2.add_argument("--fig-dir", default="figures", help="Figure output directory")
    p2.set_defaults(func=cmd_e2)

    p3 = sub.add_parser("e3", parents=[common], help="E3 coverage vs diagnosis outcome")
    p3.add_argument("--dataset", help="Filter by dataset")
    p3.add_argument("--model", help="Filter by model")
    p3.add_argument("--fig-dir", default="figures", help="Figure output directory")
    p3.set_defaults(func=cmd_e3)

    p4 = sub.add_parser("e4", parents=[common], help="E4 ablation comparison")
    p4.add_argument("--baseline", required=True, help="Baseline method")
    p4.add_argument("--md", help="Save to markdown")
    p4.set_defaults(func=cmd_e4)

    p_turns = sub.add_parser("turns", parents=[common], help="Average interaction turns")
    p_turns.add_argument("--md", help="Save to markdown")
    p_turns.set_defaults(func=cmd_turns)

    p_turn_time = sub.add_parser("turn_time", parents=[common], help="Average per-turn duration")
    p_turn_time.add_argument("--by-dataset", action="store_true", help="Show per-dataset rows")
    p_turn_time.add_argument("--md", help="Save to markdown")
    p_turn_time.set_defaults(func=cmd_turn_time)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
