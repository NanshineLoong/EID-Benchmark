"""Chain-of-Thought (CoT) scenario.

Single-pass diagnosis from case description without multi-turn interaction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from eid.agents import create_agent
from eid.prompts import PromptManager
from eid.scenarios.base import BaseScenario, CaseInput, ScenarioResult

if TYPE_CHECKING:
    from eid.config import ModelConfig


class CoTScenario(BaseScenario):
    """Chain-of-Thought evaluation scenario.

    The doctor agent receives the full case description and produces
    a diagnosis in a single pass with explicit reasoning.
    """

    def __init__(
        self,
        dataset_name: str,
        doctor_config: "ModelConfig",
    ) -> None:
        """Initialize CoT scenario.

        Args:
            dataset_name: Name of the dataset
            doctor_config: Model configuration for doctor agent
        """
        super().__init__(dataset_name)
        self.doctor_config = doctor_config
        self.prompts = PromptManager(dataset_name)

    def run(self, case_input: CaseInput) -> ScenarioResult:
        """Execute CoT scenario on a single case.

        Args:
            case_input: Input data for the case

        Returns:
            ScenarioResult with diagnosis and trace
        """
        # Create doctor agent with CoT system prompt
        doctor = create_agent(
            role_id="doctor",
            system_prompt="You are a board-certified clinician.",
            config=self.doctor_config,
        )

        # Build instruction with case information
        instruction = self.prompts.get_cot_instruction(case_input.task)

        # Execute
        response = doctor.step(instruction)
        duration = doctor.get_last_duration()

        # Extract answer
        _, answer = self.extract_action(response)
        if not answer:
            # If no [DIAGNOSIS] tag, use full response
            answer = response

        # Build trace
        trace = [
            {
                "role_id": "doctor",
                "content": response,
                "duration": duration,
            }
        ]

        # Collect role records for detailed logging
        role_records = [{
            "doctor": doctor.get_history(),
            "token_usage": doctor.get_usage_stats(),
        }]

        return ScenarioResult(
            answer=answer,
            trace=trace,
            metadata={
                "mode": "cot",
                "role_records": role_records,
            },
        )
