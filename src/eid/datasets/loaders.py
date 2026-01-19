"""Dataset loaders for various medical diagnosis benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eid.datasets.base import Dataset, DataItem

# Supported dataset configurations
SUPPORTED_DATASETS = {
    "medqa": {
        "default_path": "data/agentclinic_medqa_segmented.jsonl",
        "task_type": "diagnosis",
    },
    "diagnosisarena": {
        "default_path": "data/DiagnosisArena_segmented.jsonl",
        "task_type": "differential_diagnosis",
    },
    "clinicalbench": {
        "default_path": "data/clinicalbench_segmented.jsonl",
        "task_type": "differential_diagnosis",
    },
    "rarearena": {
        "default_path": "data/RDC_segmented.jsonl",
        "task_type": "differential_diagnosis",
    },
    "derm": {
        "default_path": "data/derm_segmented.jsonl",
        "task_type": "diagnosis",
    },
}


def load_dataset(
    name: str,
    path: str | Path | None = None,
    max_items: int | None = None,
) -> Dataset:
    """Load a dataset by name.

    Args:
        name: Dataset name (medqa, diagnosisarena, clinicalbench, rarearena, derm)
        path: Custom path to dataset file (optional)
        max_items: Maximum number of items to load

    Returns:
        Dataset instance

    Raises:
        ValueError: If dataset name is not recognized
        FileNotFoundError: If dataset file not found
    """
    name_lower = name.lower()

    if name_lower not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unknown dataset: {name}. "
            f"Supported datasets: {list(SUPPORTED_DATASETS.keys())}"
        )

    config = SUPPORTED_DATASETS[name_lower]

    if path is None:
        path = Path(config["default_path"])
    else:
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    # Load based on dataset type
    if name_lower == "medqa":
        items = _load_medqa(path)
    elif name_lower == "diagnosisarena":
        items = _load_diagnosisarena(path)
    elif name_lower == "clinicalbench":
        items = _load_clinicalbench(path)
    elif name_lower == "rarearena":
        items = _load_rarearena(path)
    elif name_lower == "derm":
        items = _load_derm(path)
    else:
        items = _load_generic(path)

    return Dataset(items=items, name=name_lower, max_items=max_items)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL file."""
    items = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def _extract_atomic_facts(data: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Extract patient and exam facts from atomic_facts field."""
    atomic = data.get("atomic_facts") or {}
    patient_facts = atomic.get("patient_facts") or atomic.get("patient_fact") or []
    exam_facts = atomic.get("exam_facts") or atomic.get("measurement_facts") or []
    return patient_facts, exam_facts


def _build_task_from_osce(osce: dict[str, Any]) -> str:
    """Build task description from OSCE_Examination structure.
    
    This creates a comprehensive case presentation for CoT evaluation.
    """
    parts = []
    
    # Objective
    if osce.get("Objective_for_Doctor"):
        parts.append(f"Objective: {osce['Objective_for_Doctor']}")
    
    # Patient information
    patient = osce.get("Patient_Actor", {})
    if patient.get("Demographics"):
        parts.append(f"\nPatient: {patient['Demographics']}")
    
    if patient.get("History"):
        parts.append(f"\nHistory: {patient['History']}")
    
    # Symptoms
    symptoms = patient.get("Symptoms", {})
    if symptoms:
        symptom_parts = []
        if symptoms.get("Primary_Symptom"):
            symptom_parts.append(f"Primary: {symptoms['Primary_Symptom']}")
        if symptoms.get("Secondary_Symptoms"):
            secondary = symptoms["Secondary_Symptoms"]
            if isinstance(secondary, list):
                symptom_parts.append(f"Secondary: {', '.join(secondary)}")
            else:
                symptom_parts.append(f"Secondary: {secondary}")
        if symptom_parts:
            parts.append(f"\nSymptoms: {'; '.join(symptom_parts)}")
    
    # Past medical history
    if patient.get("Past_Medical_History"):
        parts.append(f"\nPast Medical History: {patient['Past_Medical_History']}")
    
    # Social history
    if patient.get("Social_History"):
        parts.append(f"\nSocial History: {patient['Social_History']}")
    
    # Review of systems
    if patient.get("Review_of_Systems"):
        parts.append(f"\nReview of Systems: {patient['Review_of_Systems']}")
    
    # Physical examination findings
    exam_findings = osce.get("Physical_Examination_Findings", {})
    if exam_findings:
        exam_parts = []
        
        # Vital signs
        vitals = exam_findings.get("Vital_Signs", {})
        if vitals:
            vital_strs = [f"{k}: {v}" for k, v in vitals.items() if v]
            if vital_strs:
                exam_parts.append(f"Vital Signs: {', '.join(vital_strs)}")
        
        # Other examination findings
        for key, value in exam_findings.items():
            if key == "Vital_Signs":
                continue
            if isinstance(value, dict):
                sub_parts = [f"{k}: {v}" for k, v in value.items() if v]
                if sub_parts:
                    exam_parts.append(f"{key}: {'; '.join(sub_parts)}")
            elif value:
                exam_parts.append(f"{key}: {value}")
        
        if exam_parts:
            parts.append(f"\nPhysical Examination:\n- " + "\n- ".join(exam_parts))
    
    # Test results
    test_results = osce.get("Test_Results", {})
    if test_results:
        test_parts = []
        for category, results in test_results.items():
            if isinstance(results, dict):
                if "Findings" in results:
                    test_parts.append(f"{category}: {results['Findings']}")
                else:
                    sub_parts = [f"{k}: {v}" for k, v in results.items() if v]
                    if sub_parts:
                        test_parts.append(f"{category}: {'; '.join(sub_parts)}")
            elif results:
                test_parts.append(f"{category}: {results}")
        
        if test_parts:
            parts.append(f"\nTest Results:\n- " + "\n- ".join(test_parts))
    
    return "".join(parts)


def _load_medqa(path: Path) -> list[DataItem]:
    """Load AgentClinic MedQA dataset."""
    raw_items = _load_jsonl(path)
    items = []

    for data in raw_items:
        case_id = str(data.get("id") or data.get("case_id") or data.get("_id"))
        
        # Build task from OSCE_Examination if available
        osce = data.get("OSCE_Examination", {})
        if osce:
            task = _build_task_from_osce(osce)
            answer = osce.get("Correct_Diagnosis") or ""
        else:
            task = data.get("task") or data.get("question") or data.get("input") or ""
            answer = data.get("answer") or data.get("ground_truth") or ""

        patient_facts, exam_facts = _extract_atomic_facts(data)

        items.append(DataItem(
            case_id=case_id,
            task=task,
            answer=answer,
            patient_facts=patient_facts,
            exam_facts=exam_facts,
            raw=data,
        ))

    return items


def _load_diagnosisarena(path: Path) -> list[DataItem]:
    """Load DiagnosisArena dataset."""
    raw_items = _load_jsonl(path)
    items = []

    for data in raw_items:
        case_id = str(data.get("clinical_case_uid") or data.get("case_id") or data.get("id"))
        task = data.get("task") or data.get("clinical_case") or ""
        answer = data.get("answer") or data.get("differential_diagnosis") or ""

        if isinstance(answer, list):
            answer = ", ".join(answer)

        patient_facts, exam_facts = _extract_atomic_facts(data)

        items.append(DataItem(
            case_id=case_id,
            task=task,
            answer=answer,
            patient_facts=patient_facts,
            exam_facts=exam_facts,
            raw=data,
        ))

    return items


def _load_clinicalbench(path: Path) -> list[DataItem]:
    """Load ClinicalBench dataset."""
    raw_items = _load_jsonl(path)
    items = []

    for data in raw_items:
        case_id = str(data.get("case_id") or data.get("id") or data.get("_id"))
        task = data.get("task") or data.get("case_description") or ""
        answer = data.get("answer") or data.get("differential_diagnosis") or ""

        if isinstance(answer, list):
            answer = ", ".join(answer)

        patient_facts, exam_facts = _extract_atomic_facts(data)

        items.append(DataItem(
            case_id=case_id,
            task=task,
            answer=answer,
            patient_facts=patient_facts,
            exam_facts=exam_facts,
            raw=data,
        ))

    return items


def _load_rarearena(path: Path) -> list[DataItem]:
    """Load RareArena dataset."""
    raw_items = _load_jsonl(path)
    items = []

    for data in raw_items:
        case_id = str(data.get("case_id") or data.get("id") or data.get("_id"))
        task = data.get("task") or data.get("case_presentation") or ""
        answer = data.get("answer") or data.get("diagnosis") or ""

        if isinstance(answer, list):
            answer = ", ".join(answer)

        patient_facts, exam_facts = _extract_atomic_facts(data)

        items.append(DataItem(
            case_id=case_id,
            task=task,
            answer=answer,
            patient_facts=patient_facts,
            exam_facts=exam_facts,
            raw=data,
        ))

    return items


def _load_derm(path: Path) -> list[DataItem]:
    """Load Derm dataset."""
    raw_items = _load_jsonl(path)
    items = []

    for data in raw_items:
        case_id = str(data.get("case_id") or data.get("id") or data.get("_id"))
        task = data.get("task") or data.get("case_description") or ""
        answer = data.get("answer") or data.get("diagnosis") or ""

        patient_facts, exam_facts = _extract_atomic_facts(data)

        items.append(DataItem(
            case_id=case_id,
            task=task,
            answer=answer,
            patient_facts=patient_facts,
            exam_facts=exam_facts,
            raw=data,
        ))

    return items


def _load_generic(path: Path) -> list[DataItem]:
    """Load generic JSONL dataset with standard fields."""
    raw_items = _load_jsonl(path)
    items = []

    for data in raw_items:
        case_id = str(
            data.get("case_id")
            or data.get("clinical_case_uid")
            or data.get("id")
            or data.get("_id")
        )
        task = data.get("task") or data.get("input") or data.get("question") or ""
        answer = data.get("answer") or data.get("ground_truth") or data.get("output") or ""

        if isinstance(answer, list):
            answer = ", ".join(str(a) for a in answer)

        patient_facts, exam_facts = _extract_atomic_facts(data)

        items.append(DataItem(
            case_id=case_id,
            task=task,
            answer=answer,
            patient_facts=patient_facts,
            exam_facts=exam_facts,
            raw=data,
        ))

    return items
