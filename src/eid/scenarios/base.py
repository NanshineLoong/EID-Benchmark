"""Base scenario class and result container."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScenarioResult:
    """Container for scenario execution results.

    Attributes:
        answer: Final answer/diagnosis from the scenario
        trace: List of interaction records
        metadata: Additional metadata
    """

    answer: str
    trace: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CaseInput:
    """Input data for a single evaluation case.

    Attributes:
        case_id: Unique identifier for the case
        task: Case description/task prompt
        patient_facts: List of patient facts for simulator
        exam_facts: List of examination facts for simulator
        ground_truth: Ground truth answer/diagnosis
    """

    case_id: str
    task: str
    patient_facts: list[str] = field(default_factory=list)
    exam_facts: list[str] = field(default_factory=list)
    ground_truth: str = ""


class BaseScenario(ABC):
    """Abstract base class for evaluation scenarios.

    Each scenario implements a specific evaluation mode with its own
    interaction pattern between agents.
    """

    def __init__(self, dataset_name: str) -> None:
        """Initialize the scenario.

        Args:
            dataset_name: Name of the dataset being evaluated
        """
        self.dataset_name = dataset_name

    @abstractmethod
    def run(self, case_input: CaseInput) -> ScenarioResult:
        """Execute the scenario on a single case.

        Args:
            case_input: Input data for the case

        Returns:
            ScenarioResult with answer and trace
        """
        pass

    @staticmethod
    def extract_action(text: str) -> tuple[str, str]:
        """Extract action type and content from agent response.

        Args:
            text: Agent response text

        Returns:
            Tuple of (action_type, content)
            action_type is one of: 'query', 'test', 'diagnosis', 'finish', 'unknown'
        """
        # Check for diagnosis first
        diagnosis_match = re.search(r"\[DIAGNOSIS\]\s*([\s\S]*)", text)
        if diagnosis_match:
            return "diagnosis", diagnosis_match.group(1).strip()

        # Check for test
        test_match = re.search(r"\[TEST\]\s*([\s\S]*)", text)
        if test_match:
            return "test", test_match.group(1).strip()

        # Check for query
        query_match = re.search(r"\[QUERY\]\s*(.*)", text)
        if query_match:
            return "query", query_match.group(1).strip()

        # Check for finish
        if "[FINISH]" in text:
            return "finish", ""

        return "unknown", text

    @staticmethod
    def extract_patient_response(text: str) -> str:
        """Extract patient response from [RESPONSE] tag.

        Args:
            text: Patient agent response

        Returns:
            Extracted response content
        """
        match = re.search(r"\[RESPONSE\]\s*([\s\S]*)", text)
        if match:
            return match.group(1).strip()
        return text

    @staticmethod
    def extract_summary(text: str) -> str:
        """Extract summary from [SUMMARY] tag.

        Args:
            text: Summarizer agent response

        Returns:
            Extracted summary content
        """
        match = re.search(r"\[SUMMARY\]\s*([\s\S]*)", text)
        if match:
            return match.group(1).strip()
        return text

    @staticmethod
    def extract_verifier_decision(text: str) -> tuple[str, str]:
        """Extract decision and feedback from verifier response.

        Args:
            text: Verifier agent response

        Returns:
            Tuple of (decision, feedback)
        """
        decision_match = re.search(r"\[DECISION\]\s*(\w+)", text)
        feedback_match = re.search(r"\[FEEDBACK\]\s*([\s\S]*)", text)

        decision = decision_match.group(1).upper() if decision_match else "PASS"
        feedback = feedback_match.group(1).strip() if feedback_match else ""

        return decision, feedback
