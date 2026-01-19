"""Plotting utilities for analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from analyze.models import RunResult

MODEL_LABEL_MAP = {
    "openai_gpt-4o": "GPT-4o",
    "openai_gpt-4o-mini": "GPT-4o-mini",
    "openrouter_anthropic_claude-3-opus": "Claude-3-Opus",
    "openrouter_anthropic_claude-3-sonnet": "Claude-3-Sonnet",
    "openrouter_deepseek_deepseek-v3.2": "DeepSeek-v3.2",
    "openrouter_qwen_qwen-2.5-72b-instruct": "Qwen2.5-72B",
}

COLOR_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]

MODE_TO_METHOD = {
    "roleplay": "Baseline",
    "react": "ReAct",
    "sc": "SC",
    "refine": "REFINE",
}

METHOD_ORDER = ["Baseline", "ReAct", "SC", "REFINE"]
SUCCESS_COLOR = "#2ca02c"
FAILURE_COLOR = "#d62728"
METHOD_MARKERS = {
    "Baseline": "o",
    "ReAct": "s",
    "SC": "D",
    "REFINE": "^",
}
HIGH_RES_DPI = 300


def _filter_runs(
    run_results: Iterable[RunResult],
    dataset: Optional[str] = None,
    model: Optional[str] = None,
) -> list[RunResult]:
    runs: list[RunResult] = []
    for run in run_results:
        if dataset and run.spec.dataset != dataset:
            continue
        if model and run.spec.model != model:
            continue
        runs.append(run)
    return runs


def _style_boxplot(bp, color: str) -> None:
    for box in bp["boxes"]:
        box.set_facecolor(color)
        box.set_edgecolor(color)
        box.set_alpha(0.7)
    for whisker in bp["whiskers"]:
        whisker.set_color(color)
    for cap in bp["caps"]:
        cap.set_color(color)
    for median in bp["medians"]:
        median.set_color("#222222")
    for flier in bp.get("fliers", []):
        flier.set(markerfacecolor=color, markeredgecolor=color, alpha=0.4)


def scatter_success_vs_coverage(
    run_results: Iterable[RunResult],
    output_path: Path,
    dataset: Optional[str] = None,
    annotate: bool = True,
) -> None:
    """Scatter plot: average info coverage vs success rate."""
    runs = [r for r in _filter_runs(run_results, dataset=dataset) if r.total_cases > 0]
    if not runs:
        return

    def _model_label(model: str) -> str:
        return MODEL_LABEL_MAP.get(model) or model

    def _method_label(run: RunResult) -> str:
        return MODE_TO_METHOD.get(run.spec.mode, run.spec.method_label)

    def _annotation_for_run(run: RunResult) -> str:
        model_name = _model_label(run.spec.model)
        method_name = _method_label(run)
        if run.spec.mode == "roleplay":
            return model_name
        return f"{method_name}({model_name})"

    def _marker_for_method(method_name: str) -> str:
        return METHOD_MARKERS.get(method_name, "X")

    model_order: list[str] = []
    for r in runs:
        if r.spec.model not in model_order:
            model_order.append(r.spec.model)
    model_colors = {
        model: COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        for idx, model in enumerate(model_order)
    }

    with plt.rc_context(
        {
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    ):
        plt.figure(figsize=(6.5, 5.5))
        xs: list[float] = []
        ys: list[float] = []

        for run in runs:
            x = (run.avg_patient_coverage + run.avg_exam_coverage) / 2 * 100
            y = (run.success_rate or 0) * 100
            marker = _marker_for_method(_method_label(run))
            color = model_colors.get(run.spec.model, COLOR_PALETTE[0])
            plt.scatter(x, y, color=color, marker=marker)
            if annotate:
                plt.annotate(
                    _annotation_for_run(run),
                    (x, y),
                    fontsize=7,
                    textcoords="offset points",
                    xytext=(0, 6),
                    ha="center",
                )
            xs.append(x)
            ys.append(y)

        if xs and ys:
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            x_range = x_max - x_min
            y_range = y_max - y_min
            x_pad = max(1, x_range * 0.05) if x_range else 2
            y_pad = max(1, y_range * 0.05) if y_range else 2
            plt.xlim(x_min - x_pad, x_max + x_pad)
            plt.ylim(y_min - y_pad, y_max + y_pad)

        plt.xlabel("Information Collection Rate (ICR) (%)")
        plt.ylabel("Success Rate (SR) (%)")
        plt.grid(True, alpha=0.3)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout(pad=1.5)
        plt.savefig(output_path, bbox_inches="tight", dpi=HIGH_RES_DPI)
        plt.close()


def plot_turn_limit_trend(
    run_results: Iterable[RunResult],
    output_path: Path,
    metric: str = "success_rate",
    dataset: Optional[str] = None,
    model: Optional[str] = None,
) -> None:
    """Line plot: success rate (left) with avg ICR (right) vs max turns."""
    runs = [
        run
        for run in _filter_runs(run_results, dataset=dataset, model=model)
        if run.spec.max_turns is not None
    ]
    if not runs:
        return

    grouped: dict[str, list[RunResult]] = {}
    for run in runs:
        method_label = MODE_TO_METHOD.get(run.spec.mode, run.spec.method_label)
        grouped.setdefault(method_label, []).append(run)

    def _auto_limits(values: list[float], pad_ratio: float = 0.05) -> tuple[float, float]:
        vals = [v for v in values if v is not None]
        if not vals:
            return (0, 1)
        vmin, vmax = min(vals), max(vals)
        if vmin == vmax:
            delta = max(abs(vmax) * 0.1, 1)
            return (vmin - delta, vmax + delta)
        pad = (vmax - vmin) * pad_ratio
        return (vmin - pad, vmax + pad)

    with plt.rc_context(
        {
            "font.size": 13,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
        }
    ):
        fig, ax_left = plt.subplots(figsize=(8, 5.5))
        ax_right = ax_left.twinx()

        method_keys = [m for m in METHOD_ORDER if m in grouped]
        method_keys.extend(sorted(k for k in grouped.keys() if k not in METHOD_ORDER))

        handles: list = []
        labels: list = []
        all_success_vals: list[float] = []
        all_icr_vals: list[float] = []

        for idx, method in enumerate(method_keys):
            items = grouped[method]
            items = sorted(items, key=lambda r: r.spec.max_turns or 0)
            x = [r.spec.max_turns for r in items]
            success_y = [(r.success_rate or 0) * 100 for r in items]
            avg_icr_y = [
                ((r.avg_patient_coverage + r.avg_exam_coverage) / 2) * 100 for r in items
            ]

            color = COLOR_PALETTE[(idx + 2) % len(COLOR_PALETTE)]
            success_marker = METHOD_MARKERS.get(method, MARKERS[idx % len(MARKERS)])

            lh = ax_left.plot(
                x,
                success_y,
                label=f"{method} - SR",
                color=color,
                linestyle="-",
                marker=success_marker,
                linewidth=2.4,
                markersize=7,
            )[0]
            ih = ax_right.plot(
                x,
                avg_icr_y,
                label=f"{method} - ICR",
                color=color,
                linestyle="--",
                marker="D",
                linewidth=2.2,
                markersize=7,
                alpha=0.9,
            )[0]
            handles.extend([lh, ih])
            labels.extend([f"{method} - SR", f"{method} - ICR"])
            all_success_vals.extend(success_y)
            all_icr_vals.extend(avg_icr_y)

        left_min, left_max = _auto_limits(all_success_vals)
        right_min, right_max = _auto_limits(all_icr_vals)
        ax_left.set_ylim(left_min, left_max)
        ax_right.set_ylim(right_min, right_max)

        ax_left.set_xlabel("Max turns")
        ax_left.set_ylabel("SR (%)")
        ax_right.set_ylabel("ICR (%)")
        ax_left.grid(True, alpha=0.3, linestyle="--")

        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=2,
            frameon=False,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(pad=0.6)
        fig.subplots_adjust(top=0.86)
        fig.savefig(output_path, bbox_inches="tight", dpi=HIGH_RES_DPI)
        plt.close(fig)


def plot_coverage_curve(
    run_results: Iterable[RunResult],
    output_path: Path,
    dataset: Optional[str] = None,
    model: Optional[str] = None,
    metric: str = "patient",
) -> None:
    """Plot coverage growth across dialogue turns for each method."""
    runs = _filter_runs(run_results, dataset=dataset, model=model)
    if not runs:
        return

    plt.figure(figsize=(7, 5))
    for run in runs:
        x = list(range(1, len(run.coverage_curve_patient) + 1))
        if metric == "exam":
            y = [v * 100 for v in run.coverage_curve_exam]
            ylabel = "Exam info coverage (%)"
        else:
            y = [v * 100 for v in run.coverage_curve_patient]
            ylabel = "Patient info coverage (%)"
        plt.plot(x, y, marker="o", label=run.spec.method_label)
    plt.xlabel("Turns")
    plt.ylabel(ylabel)
    plt.title(f"Coverage vs turns ({dataset or 'all'})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=1.5)
    plt.savefig(output_path, dpi=HIGH_RES_DPI)
    plt.close()


def plot_outcome_bars(
    run_results: Iterable[RunResult],
    output_path: Path,
    dataset: Optional[str] = None,
    model: Optional[str] = None,
) -> None:
    """Success vs failure ICR distributions per dataset/method."""
    runs = _filter_runs(run_results, dataset=dataset, model=model)
    if not runs:
        return

    datasets = sorted({r.spec.dataset for r in runs})
    with plt.rc_context(
        {
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    ):
        fig, axes = plt.subplots(
            1,
            len(datasets),
            figsize=(max(8, len(datasets) * 4), 5),
            sharey=True,
        )
        if len(datasets) == 1:
            axes = [axes]

        legend_handles = [
            Patch(facecolor=SUCCESS_COLOR, edgecolor=SUCCESS_COLOR, alpha=0.7, label="Success"),
            Patch(facecolor=FAILURE_COLOR, edgecolor=FAILURE_COLOR, alpha=0.7, label="Failure"),
        ]

        for ax, ds in zip(axes, datasets):
            dataset_runs = [r for r in runs if r.spec.dataset == ds]
            method_values: dict[str, dict[str, list[float]]] = {}
            for run in dataset_runs:
                method_label = MODE_TO_METHOD.get(run.spec.mode, run.spec.method_label)
                buckets = method_values.setdefault(method_label, {"success": [], "failure": []})
                for cm in run.case_metrics:
                    icr = ((cm.patient_coverage + cm.exam_coverage) / 2) * 100
                    if cm.success is True:
                        buckets["success"].append(icr)
                    elif cm.success is False:
                        buckets["failure"].append(icr)

            method_labels = [
                label
                for label in METHOD_ORDER
                if label in method_values
                and (method_values[label]["success"] or method_values[label]["failure"])
            ]

            if not method_labels:
                ax.set_title(f"{ds} (no data)")
                ax.axis("off")
                continue

            centers = []
            last_pos = -1.0
            for idx, label in enumerate(method_labels):
                base = idx * 3
                centers.append(base + 0.5)
                success_vals = method_values[label]["success"]
                failure_vals = method_values[label]["failure"]
                if success_vals:
                    bp = ax.boxplot(
                        success_vals,
                        positions=[base],
                        widths=0.8,
                        patch_artist=True,
                        showmeans=False,
                    )
                    _style_boxplot(bp, SUCCESS_COLOR)
                if failure_vals:
                    bp = ax.boxplot(
                        failure_vals,
                        positions=[base + 1],
                        widths=0.8,
                        patch_artist=True,
                        showmeans=False,
                    )
                    _style_boxplot(bp, FAILURE_COLOR)
                last_pos = base + 1

            ax.set_xticks(centers)
            ax.set_xticklabels(method_labels, rotation=20)
            ax.set_title(ds)
            ax.set_xlim(-1, last_pos + 1.5)
            ax.set_ylim(-2, 102)
            ax.grid(True, axis="y", alpha=0.3, linestyle="--")

        axes[0].set_ylabel("ICR (%)")
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=2,
            frameon=False,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(pad=0.6)
        fig.subplots_adjust(bottom=0.16)
        fig.savefig(output_path, bbox_inches="tight", dpi=HIGH_RES_DPI)
        plt.close(fig)
