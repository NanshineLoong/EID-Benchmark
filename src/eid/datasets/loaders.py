"""Dataset loaders for various medical diagnosis benchmarks."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from eid.datasets.base import Dataset, DataItem

logger = logging.getLogger(__name__)

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
        raise ValueError(f"No loader implemented for dataset: {name}")

    logger.info(f"Loaded {len(items)} items from {name_lower} dataset")
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
    """Extract patient and exam facts from atomic_facts field.
    
    Returns lists of facts as-is from the data structure.
    """
    atomic = data.get("atomic_facts", {})
    if not isinstance(atomic, dict):
        return [], []
    
    patient_facts = atomic.get("patient_facts", [])
    exam_facts = atomic.get("exam_facts", [])
    
    # Ensure we return lists
    if not isinstance(patient_facts, list):
        patient_facts = []
    if not isinstance(exam_facts, list):
        exam_facts = []
    
    return patient_facts, exam_facts


def _build_task_from_osce(osce: dict[str, Any]) -> str:
    """Build task description from OSCE_Examination structure.
    
    Creates a comprehensive case presentation following clinical structure:
    Objective -> Patient Info -> Physical Exam -> Test Results
    """
    parts = []
    
    # Objective
    objective = osce.get("Objective_for_Doctor")
    if objective:
        parts.append(f"Objective: {objective}")
    
    # Patient Actor information
    patient = osce.get("Patient_Actor", {})
    if not isinstance(patient, dict):
        patient = {}
    
    # Demographics
    demographics = patient.get("Demographics")
    if demographics:
        parts.append(f"\nPatient: {demographics}")
    
    # History
    history = patient.get("History")
    if history:
        parts.append(f"\nHistory: {history}")
    
    # Symptoms
    symptoms = patient.get("Symptoms", {})
    if isinstance(symptoms, dict) and symptoms:
        symptom_lines = []
        primary = symptoms.get("Primary_Symptom")
        if primary:
            symptom_lines.append(f"Primary: {primary}")
        
        secondary = symptoms.get("Secondary_Symptoms")
        if secondary:
            if isinstance(secondary, list):
                symptom_lines.append(f"Secondary: {', '.join(str(s) for s in secondary)}")
            else:
                symptom_lines.append(f"Secondary: {secondary}")
        
        if symptom_lines:
            parts.append(f"\nSymptoms: {'; '.join(symptom_lines)}")
    
    # Past Medical History
    pmh = patient.get("Past_Medical_History")
    if pmh:
        parts.append(f"\nPast Medical History: {pmh}")
    
    # Social History
    social = patient.get("Social_History")
    if social:
        parts.append(f"\nSocial History: {social}")
    
    # Review of Systems
    ros = patient.get("Review_of_Systems")
    if ros:
        parts.append(f"\nReview of Systems: {ros}")
    
    # Physical Examination Findings
    exam_findings = osce.get("Physical_Examination_Findings", {})
    if isinstance(exam_findings, dict) and exam_findings:
        exam_lines = []
        
        # Vital Signs first
        vitals = exam_findings.get("Vital_Signs", {})
        if isinstance(vitals, dict) and vitals:
            vital_strs = [f"{k}: {v}" for k, v in vitals.items() if v]
            if vital_strs:
                exam_lines.append(f"Vital Signs: {', '.join(vital_strs)}")
        
        # Other examination findings
        for key, value in exam_findings.items():
            if key == "Vital_Signs":
                continue
            
            if isinstance(value, dict):
                sub_strs = [f"{k}: {v}" for k, v in value.items() if v]
                if sub_strs:
                    exam_lines.append(f"{key}: {'; '.join(sub_strs)}")
            elif value:
                exam_lines.append(f"{key}: {value}")
        
        if exam_lines:
            parts.append(f"\nPhysical Examination:\n- " + "\n- ".join(exam_lines))
    
    # Test Results
    test_results = osce.get("Test_Results", {})
    if isinstance(test_results, dict) and test_results:
        test_lines = []
        
        for category, results in test_results.items():
            if isinstance(results, dict):
                # Look for Findings field first
                findings = results.get("Findings")
                if findings:
                    test_lines.append(f"{category}: {findings}")
                else:
                    # Otherwise format all sub-fields
                    sub_strs = [f"{k}: {v}" for k, v in results.items() if v]
                    if sub_strs:
                        test_lines.append(f"{category}: {'; '.join(sub_strs)}")
            elif results:
                test_lines.append(f"{category}: {results}")
        
        if test_lines:
            parts.append(f"\nTest Results:\n- " + "\n- ".join(test_lines))
    
    return "".join(parts)


def _load_medqa(path: Path) -> list[DataItem]:
    """Load AgentClinic MedQA dataset.
    
    Expects JSONL with OSCE_Examination structure containing:
    - Objective_for_Doctor
    - Patient_Actor
    - Physical_Examination_Findings  
    - Test_Results
    - Correct_Diagnosis
    """
    raw_items = _load_jsonl(path)
    items = []

    for idx, data in enumerate(raw_items):
        try:
            case_id = str(data.get("id", f"medqa_{idx}"))
            
            # Extract OSCE_Examination structure
            osce = data.get("OSCE_Examination")
            if not osce:
                logger.warning(f"Missing OSCE_Examination in item {idx}, skipping")
                continue
            
            # Build comprehensive task description from OSCE structure
            task = _build_task_from_osce(osce)
            answer = osce.get("Correct_Diagnosis", "")
            
            if not answer:
                logger.warning(f"Missing Correct_Diagnosis in item {case_id}")

            patient_facts, exam_facts = _extract_atomic_facts(data)

            items.append(DataItem(
                case_id=case_id,
                task=task,
                answer=answer,
                patient_facts=patient_facts,
                exam_facts=exam_facts,
                raw=data,
            ))
        except Exception as e:
            logger.error(f"Failed to parse MedQA item {idx}: {e}")
            continue

    return items


def _load_diagnosisarena(path: Path) -> list[DataItem]:
    """Load DiagnosisArena dataset.
    
    Expects JSONL with fields:
    - id: case identifier
    - Case Information: patient demographics and history
    - Physical Examination: physical exam findings
    - Diagnostic Tests: lab/imaging results
    - Final Diagnosis: ground truth diagnosis
    - Options: multiple choice options (optional)
    - Right Option: correct option (optional)
    """
    raw_items = _load_jsonl(path)
    items = []

    for idx, data in enumerate(raw_items):
        try:
            case_id = str(data.get("id", f"diagnosisarena_{idx}"))
            
            # Build task from case components
            case_info = data.get("Case Information", "")
            physical_exam = data.get("Physical Examination", "")
            diagnostic_tests = data.get("Diagnostic Tests", "")
            
            task = f"""Make a diagnosis for the patient's disease based on the case information, physical examination, and diagnostic tests. Enumerate the top 5 most likely diagnoses for the following patient in order, with the most likely disease listed first.

Case Information:
{case_info}

Physical Examination:
{physical_exam}

Diagnostic Tests:
{diagnostic_tests}"""

            answer = data.get("Final Diagnosis", "")
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
        except Exception as e:
            logger.error(f"Failed to parse DiagnosisArena item {idx}: {e}")
            continue

    return items


def _load_clinicalbench(path: Path) -> list[DataItem]:
    """Load ClinicalBench dataset.
    
    Expects JSONL with fields:
    - clinical_case_uid or id: case identifier
    - clinical_case_summary: case description
    - principal_diagnosis or differential_diagnosis: ground truth
    """
    raw_items = _load_jsonl(path)
    items = []

    for idx, data in enumerate(raw_items):
        try:
            case_id = str(data.get("clinical_case_uid") or data.get("id", f"clinicalbench_{idx}"))
            
            # Try different field names for case description
            case_summary = data.get("clinical_case_summary") or data.get("case_description") or ""
            
            # Build task
            task = f"""Make a differential diagnosis for the patient based on the case report. Enumerate the top 5 most likely diagnoses for the following patient in order, with the most likely diagnosis listed first.

Case Report:
{case_summary}"""

            answer = (
                data.get("principal_diagnosis") 
                or data.get("differential_diagnosis")
                or data.get("answer")
                or ""
            )
            
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
        except Exception as e:
            logger.error(f"Failed to parse ClinicalBench item {idx}: {e}")
            continue

    return items


def _load_rarearena(path: Path) -> list[DataItem]:
    """Load RareArena dataset (RDC or RDS task).
    
    Expects JSONL with fields:
    - _id: case identifier
    - case_report: patient case description
    - test_results: diagnostic test results (RDC only)
    - diagnosis or Orpha_name: ground truth diagnosis
    - Orpha_id: ORPHA disease identifier
    """
    raw_items = _load_jsonl(path)
    items = []

    for idx, data in enumerate(raw_items):
        try:
            case_id = str(data.get("_id") or data.get("id", f"rarearena_{idx}"))
            
            case_report = data.get("case_report", "")
            test_results = data.get("test_results", "")
            
            # Build task - check if test_results available (RDC) or not (RDS)
            if test_results:
                task = f"""Make a diagnosis for the patient's rare disease based on the case report and diagnostic test results. Enumerate the top 5 most likely rare disease diagnoses for the following patient in order, with the most likely disease listed first.

Case Report:
{case_report}

Diagnostic Tests:
{test_results}"""
            else:
                task = f"""Make a diagnosis for the patient's rare disease based on the case report. Enumerate the top 5 most likely rare disease diagnoses for the following patient in order, with the most likely disease listed first.

Case Report:
{case_report}"""

            # Extract diagnosis from various field names
            answer = data.get("diagnosis") or data.get("Orpha_name") or ""
            
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
        except Exception as e:
            logger.error(f"Failed to parse RareArena item {idx}: {e}")
            continue

    return items


def _load_derm(path: Path) -> list[DataItem]:
    """Load Dermatology dataset.
    
    Expects JSONL with fields:
    - case_id: case identifier
    - case_vignette: clinical case description
    - answer: ground truth diagnosis
    - choice_1 to choice_4: multiple choice options (optional)
    """
    raw_items = _load_jsonl(path)
    items = []

    for idx, data in enumerate(raw_items):
        try:
            case_id = str(data.get("Unnamed: 0", f"derm_{idx}"))
            
            # case_vignette contains the full case description
            case_vignette = data.get("case_vignette", "")
            
            # Build task prompt
            task = f"""Please diagnose the patient based on the following information:

{case_vignette}"""

            answer = data.get("answer", "")

            patient_facts, exam_facts = _extract_atomic_facts(data)

            items.append(DataItem(
                case_id=case_id,
                task=task,
                answer=answer,
                patient_facts=patient_facts,
                exam_facts=exam_facts,
                raw=data,
            ))
        except Exception as e:
            logger.error(f"Failed to parse Derm item {idx}: {e}")
            continue

    return items
