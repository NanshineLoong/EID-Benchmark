"""Excel export utilities for analysis results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def rows_to_wide(rows: list[dict]) -> pd.DataFrame:
    """Convert long-format rows to wide format for Excel export.

    Pivots the data so that each metric becomes a separate column,
    with models as rows and grouped by dataset/mode.
    """
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Create a combined index for pivoting
    if "max_turns" in df.columns and df["max_turns"].notna().any():
        df["mode_turns"] = df.apply(
            lambda r: f"{r['mode']}_{r['max_turns']}t"
            if pd.notna(r["max_turns"])
            else r["mode"],
            axis=1,
        )
    else:
        df["mode_turns"] = df["mode"]

    # Select metrics columns
    metric_cols = [
        "success_rate",
        "patient_coverage",
        "exam_coverage",
        "info_coverage",
        "avg_turns",
    ]
    existing_metrics = [c for c in metric_cols if c in df.columns]

    # Melt to long format
    id_vars = ["dataset", "model", "mode_turns"]
    melted = df.melt(
        id_vars=id_vars,
        value_vars=existing_metrics,
        var_name="metric",
        value_name="value",
    )

    # Pivot to wide format
    wide = melted.pivot_table(
        index=["dataset", "model"],
        columns=["mode_turns", "metric"],
        values="value",
        aggfunc="first",
    )

    # Flatten column names
    wide.columns = [f"{mode}_{metric}" for mode, metric in wide.columns]
    wide = wide.reset_index()

    return wide


def export_excel(
    df: pd.DataFrame,
    output_path: Path,
    sheet_name: str = "Results",
) -> None:
    """Export DataFrame to Excel with formatting.

    Args:
        df: DataFrame to export
        output_path: Output file path
        sheet_name: Name of the sheet
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert percentage values
    for col in df.columns:
        if any(
            keyword in col.lower()
            for keyword in ["rate", "coverage", "accuracy"]
        ):
            if df[col].dtype in ["float64", "float32"]:
                # Check if values are in 0-1 range and convert to percentage
                if df[col].max() <= 1.0:
                    df[col] = df[col] * 100

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)

        # Get workbook and worksheet
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]

        # Auto-adjust column widths
        for idx, col in enumerate(df.columns):
            max_length = max(
                df[col].astype(str).map(len).max(),
                len(col),
            )
            worksheet.column_dimensions[chr(65 + idx)].width = min(max_length + 2, 30)

    print(f"Saved Excel file to {output_path}")
