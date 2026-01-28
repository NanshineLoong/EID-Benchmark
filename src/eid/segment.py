#!/usr/bin/env python3
"""Data segmentation tool for extracting atomic facts from medical case data.

This module provides functionality to split complex medical case information into
independent atomic facts, categorized as patient_facts (demographics, history,
symptoms) and exam_facts (tests, lab results, imaging studies).

Usage:
    eid-segment --dataset data/derm.jsonl --fields case_vignette --model gpt-5-mini
"""

import argparse
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from tqdm import tqdm

from camel.agents import ChatAgent
from camel.configs.openai_config import ChatGPTConfig
from camel.models import ModelFactory
from camel.types import ModelPlatformType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class AtomicFactsResponse(BaseModel):
    """Response model for atomic facts extraction."""

    patient_facts: List[str] = Field(
        description=(
            "Independent, non-overlapping atomic facts about patient information. "
            "Each string must start with an index like '1. '. Return [] if none."
        )
    )
    exam_facts: List[str] = Field(
        description=(
            "Independent, non-overlapping atomic facts about examinations/tests/results. "
            "Each string must start with an index like '1. '. Return [] if none."
        )
    )


def resolve_dataset_path(dataset: str) -> Path:
    """Resolve dataset path, checking both absolute and relative paths.

    Args:
        dataset: Path to the dataset file.

    Returns:
        Resolved Path object.
    """
    p = Path(dataset)
    if p.exists():
        return p

    # Try data/ subdirectory
    candidate = Path.cwd() / "data" / dataset
    if candidate.exists():
        return candidate
    return p


def iter_jsonl(path: Path, max_items: Optional[int] = None):
    """Iterate over JSONL file records.

    Args:
        path: Path to JSONL file.
        max_items: Maximum number of items to yield.

    Yields:
        Parsed JSON records.
    """
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            n += 1
            if max_items is not None and n >= max_items:
                break


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write records to JSONL file.

    Args:
        path: Output file path.
        rows: List of records to write.
    """
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _to_text(v: Any) -> str:
    """Convert value to text representation.

    Args:
        v: Value to convert.

    Returns:
        String representation.
    """
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    return json.dumps(v, ensure_ascii=False)


class Segmenter:
    """LLM-based segmenter for extracting atomic facts from medical cases."""

    def __init__(self, model_name: str = "gpt-5-mini") -> None:
        """Initialize the segmenter with specified model.

        Args:
            model_name: Name of the model to use for segmentation.
        """
        sys_msg = (
            "You are a careful information extraction assistant. "
            "You never add or infer information. You only rewrite/split what is explicitly provided."
        )
        model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=model_name,
            model_config_dict=ChatGPTConfig(temperature=0.0, max_tokens=40960).as_dict(),
        )
        self.agent = ChatAgent(system_message=sys_msg, model=model)

    def segment(self, payload: Dict[str, Any]) -> AtomicFactsResponse:
        """Segment case information into atomic facts.

        Args:
            payload: Dictionary containing case information fields.

        Returns:
            AtomicFactsResponse with patient_facts and exam_facts.

        Raises:
            Exception: If segmentation fails.
        """
        self.agent.reset()  # Keep calls independent

        prompt = (
            "Break the following information into independent atomic facts.\n"
            "Rules:\n"
            "- One piece of information per statement.\n"
            "- Facts must be self-contained and non-overlapping.\n"
            "- Do NOT add, infer, or normalize beyond the given text.\n"
            "- Keep the original language of the input.\n"
            "- Each fact string must start with an index like '1. ', '2. ', etc.\n"
            "- Classify each fact into either patient_facts or exam_facts:\n"
            "  * patient_facts: Information about the patient's demographics, history, "
            "symptoms, complaints, or clinical presentation and so on.\n"
            "  * exam_facts: Information about examinations, tests, laboratory results, "
            "imaging studies and so on.\n"
            "- Do NOT duplicate facts across patient_facts and exam_facts. "
            "  If a fact could belong to both, choose the best list and omit it from the other.\n"
            "- If there is no content for a list, return an empty list.\n\n"
            "Case Information (JSON):\n"
            f"{json.dumps(payload, ensure_ascii=False)}\n"
        )

        try:
            resp = self.agent.step(prompt, response_format=AtomicFactsResponse)
            result = None
            if not resp.msg.parsed:
                try:
                    result_dict = json.loads(resp.msg.content)
                    result = AtomicFactsResponse(**result_dict)
                except Exception as e:
                    logger.warning(f"Error parsing response: {e}")
                    logger.debug(f"Response content: {resp.msg.content}")
                    return AtomicFactsResponse(patient_facts=[], exam_facts=[])
            else:
                result = resp.msg.parsed
            return result
        except Exception as e:
            # Catch all possible errors (including ContentFilterFinishReasonError, etc.)
            logger.error(f"Error in segment method: {type(e).__name__}: {e}")
            raise


_thread_local = threading.local()


def _get_segmenter(model: str) -> Segmenter:
    """Get or create thread-local Segmenter instance.

    Args:
        model: Model name for segmentation.

    Returns:
        Segmenter instance.
    """
    seg = getattr(_thread_local, "segmenter", None)
    if seg is None or getattr(_thread_local, "model_name", None) != model:
        _thread_local.segmenter = Segmenter(model_name=model)
        _thread_local.model_name = model
    return _thread_local.segmenter


def _process_one(
    idx: int,
    rec: Dict[str, Any],
    fields: List[str],
    model: str,
) -> Tuple[int, Optional[Dict[str, Any]], Optional[str]]:
    """Process a single record.

    Args:
        idx: Record index.
        rec: Record dictionary.
        fields: Fields to extract facts from.
        model: Model name for segmentation.

    Returns:
        Tuple of (index, updated_record, error_message).
        - On success: (idx, rec, None)
        - On failure: (idx, None, error_msg)
    """
    # Get record ID for error logging
    record_id = rec.get("_id") or rec.get("id") or f"index-{idx}"

    try:
        payload = {k: _to_text(rec.get(k)) for k in fields}

        seg = _get_segmenter(model)
        parsed = seg.segment(payload)

        # parsed is always AtomicFactsResponse object
        rec["atomic_facts"] = {
            "patient_facts": list(parsed.patient_facts) if parsed.patient_facts else [],
            "exam_facts": list(parsed.exam_facts) if parsed.exam_facts else [],
        }
        return idx, rec, None
    except Exception as e:
        error_msg = f"ID={record_id}, Error={type(e).__name__}: {str(e)}"
        logger.error(f"Failed to process record - {error_msg}")
        return idx, None, error_msg


def main() -> None:
    """Main entry point for the segmentation CLI."""
    ap = argparse.ArgumentParser(
        description="Segment medical case data into atomic facts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic segmentation
  eid-segment --dataset data/derm.jsonl --fields case_vignette

  # Multiple fields with custom output
  eid-segment --dataset data/cases.jsonl --fields patient_history,examination \\
              --out data/cases_segmented.jsonl

  # Parallel processing with more workers
  eid-segment --dataset data/large_dataset.jsonl --fields case_vignette \\
              --workers 50 --max-items 1000
        """,
    )
    ap.add_argument(
        "--dataset",
        default="data/derm_private_only.jsonl",
        help="Path to .jsonl file (supports data/ prefix resolution)",
    )
    ap.add_argument(
        "--fields",
        default="case_vignette",
        help="Comma-separated fields to extract facts from",
    )
    ap.add_argument(
        "--max-items",
        type=int,
        default=100,
        help="Maximum number of items to process (default: 100)",
    )
    ap.add_argument(
        "--model",
        default="gpt-5-mini",
        help="Model to use for segmentation (default: gpt-5-mini)",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=100,
        help="Number of parallel workers (default: 100)",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Optional output path (default: <input>_segmented.jsonl)",
    )
    args = ap.parse_args()

    in_path = resolve_dataset_path(args.dataset)
    if not in_path.exists():
        raise FileNotFoundError(f"Dataset not found: {in_path}")

    fields = [x.strip() for x in args.fields.split(",") if x.strip()]
    if not fields:
        raise ValueError("fields must be non-empty.")

    out_path = Path(args.out) if args.out else in_path.with_name(in_path.stem + "_segmented.jsonl")
    error_log_path = out_path.with_name(out_path.stem + "_errors.jsonl")

    records = list(iter_jsonl(in_path, max_items=args.max_items))
    out_rows: List[Optional[Dict[str, Any]]] = [None] * len(records)
    error_records: List[Dict[str, Any]] = []

    logger.info(f"Starting to process {len(records)} records...")

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [
            ex.submit(_process_one, i, records[i], fields, args.model)
            for i in range(len(records))
        ]
        success_count = 0
        with tqdm(total=len(records), desc="Progress", unit="records") as pbar:
            for fut in as_completed(futs):
                i, updated, error_msg = fut.result()
                if updated is not None:
                    out_rows[i] = updated
                    success_count += 1
                else:
                    # Record error information
                    error_rec = {
                        "index": i,
                        "record_id": records[i].get("_id") or records[i].get("id") or f"index-{i}",
                        "error": error_msg,
                        "original_record": records[i] if error_msg else None,
                    }
                    error_records.append(error_rec)
                    logger.warning(
                        f"Skipping record {i} (ID: {error_rec['record_id']}): {error_msg}"
                    )
                pbar.set_postfix({"success": success_count, "failed": len(error_records)})
                pbar.update(1)

    # Write successfully processed records
    successful_records = [r for r in out_rows if r is not None]
    write_jsonl(out_path, successful_records)
    logger.info(f"Successfully processed {len(successful_records)} records, saved to: {out_path}")

    # Write error records
    if error_records:
        write_jsonl(error_log_path, error_records)
        logger.warning(
            f"Skipped {len(error_records)} records, errors saved to: {error_log_path}"
        )
    else:
        logger.info("All records processed successfully, no errors.")


if __name__ == "__main__":
    main()
