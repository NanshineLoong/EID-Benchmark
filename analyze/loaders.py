"""IO helpers: load datasets, summaries, and record traces."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

from analyze.models import GroundTruthFacts, GroundTruthStore

# Fallback mapping for common dataset names
DEFAULT_DATASET_FILES: Dict[str, Path] = {
    "derm": Path("data/derm_segmented.jsonl"),
    "medqa": Path("data/agentclinic_medqa_extended_segmented.jsonl"),
    "agentclinic_medqa": Path("data/agentclinic_medqa_extended_segmented.jsonl"),
    "diagnosisarena": Path("data/DiagnosisArena_segmented.jsonl"),
    "clinicalbench": Path("data/ClinicalBench/data_en_sampled_segmented.jsonl"),
    "rarearena": Path("data/RareArena/RDC_sampled_segmented.jsonl"),
}


def _extract_fact_ids(facts: Iterable[str]) -> set[str]:
    ids: set[str] = set()
    for fact in facts:
        match = re.match(r"\s*(\d+)\.", fact)
        if match:
            ids.add(match.group(1))
    return ids


def load_atomic_facts(dataset_path: Path) -> GroundTruthStore:
    """Load ground-truth facts from a JSONL dataset."""
    ground_truth: GroundTruthStore = {}
    with dataset_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            # Prefer the stable UID used by result/record files (e.g., clinical_case_uid)
            case_id = str(
                data.get("case_id")
                or data.get("clinical_case_uid")
                or data.get("id")
                or data.get("_id")
            )
            atomic = data.get("atomic_facts") or {}
            patient_facts = atomic.get("patient_facts") or atomic.get("patient_fact") or []
            exam_facts = atomic.get("exam_facts") or atomic.get("measurement_facts") or []
            ground_truth[case_id] = GroundTruthFacts(
                patient_fact_ids=_extract_fact_ids(patient_facts),
                exam_fact_ids=_extract_fact_ids(exam_facts),
            )
    return ground_truth


def load_summary_results(summary_path: Path) -> tuple[dict[str, bool], dict]:
    """Return (case_id -> success flag, metadata) from a summary.json."""
    if not summary_path.exists():
        return {}, {}
    with summary_path.open(encoding="utf-8") as f:
        data = json.load(f)
    results = {}
    for item in data.get("results", []):
        case_id = str(item.get("case_id") or item.get("id") or item.get("_id"))
        if not case_id:
            continue
        success = (
            item.get("result")
            or item.get("top1_correct")
            or item.get("correct")
            or item.get("is_correct")
        )
        results[case_id] = bool(success)
    meta = {
        "top1_accuracy": data.get("top1_accuracy"),
        "total_items": data.get("total_items") or len(data.get("results", [])),
    }
    return results, meta


def iter_record_files(record_dir: Path) -> Iterable[Tuple[str, dict]]:
    """Yield (case_id, payload) from all record JSON files."""
    if not record_dir.exists():
        return []
    for path in sorted(record_dir.glob("*.json")):
        try:
            with path.open(encoding="utf-8") as f:
                yield path.stem, json.load(f)
        except Exception as exc:
            print(f"Warn: failed to load {path}: {exc}", file=sys.stderr)
