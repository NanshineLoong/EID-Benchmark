"""ReAct scenario.

Multi-turn interaction with explicit reasoning (THOUGHT) before each action.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from eid.agents import create_agent
from eid.prompts import PromptManager
from eid.scenarios.base import BaseScenario, CaseInput, ScenarioResult

if TYPE_CHECKING:
    from eid.config import ModelConfig


class ReactScenario(BaseScenario):
    """ReAct evaluation scenario.

    The doctor produces explicit [THOUGHT] reasoning before each
    [QUERY], [TEST], or [DIAGNOSIS] action.
    """

    def __init__(
        self,
        dataset_name: str,
        doctor_config: "ModelConfig",
        patient_config: "ModelConfig",
        measurement_config: "ModelConfig",
        max_turns: int = 16,
    ) -> None:
        """Initialize ReAct scenario.

        Args:
            dataset_name: Name of the dataset
            doctor_config: Model configuration for doctor agent
            patient_config: Model configuration for patient simulator
            measurement_config: Model configuration for measurement simulator
            max_turns: Maximum interaction turns
        """
        super().__init__(dataset_name)
        self.doctor_config = doctor_config
        self.patient_config = patient_config
        self.measurement_config = measurement_config
        self.max_turns = max_turns
        self.prompts = PromptManager(dataset_name)

    def run(self, case_input: CaseInput) -> ScenarioResult:
        """Execute ReAct scenario on a single case.

        Args:
            case_input: Input data for the case

        Returns:
            ScenarioResult with diagnosis and trace
        """
        # Format facts for simulators
        patient_facts_str = "\n".join(case_input.patient_facts)
        exam_facts_str = "\n".join(case_input.exam_facts)

        # Create agents
        doctor = create_agent(
            role_id="doctor",
            system_prompt=self.prompts.get_react_doctor_system_prompt().format(
                max_turns=self.max_turns
            ),
            config=self.doctor_config,
            message_window_size=24,
            summarize_threshold=80,
        )

        patient = create_agent(
            role_id="patient",
            system_prompt=self.prompts.get_patient_system_prompt().format(
                patient_facts=patient_facts_str
            ),
            config=self.patient_config,
            message_window_size=1,
            summarize_threshold=95,
        )

        measurement = create_agent(
            role_id="measurement",
            system_prompt=self.prompts.get_measurement_system_prompt().format(
                exam_facts=exam_facts_str
            ),
            config=self.measurement_config,
            message_window_size=1,
            summarize_threshold=80,
        )

        # Run interaction loop
        trace: list[dict] = []
        last_reply = ""
        answer = ""
        current_turn = 0

        while current_turn < self.max_turns:
            # Doctor turn
            instruction = self.prompts.get_doctor_turn_instruction(
                current_turns=current_turn,
                max_turns=self.max_turns,
                last_reply=last_reply,
            )

            doctor_response = doctor.step(instruction)
            duration = doctor.get_last_duration()

            trace.append({
                "role_id": "doctor",
                "content": doctor_response,
                "duration": duration,
            })

            # Parse doctor action
            action_type, action_content = self.extract_action(doctor_response)

            if action_type == "diagnosis":
                answer = action_content
                break

            current_turn += 1

            # Route to appropriate simulator
            if action_type == "test":
                # Measurement turn
                m_instruction = self.prompts.get_measurement_turn_instruction(action_content)
                m_response = measurement.step(m_instruction)
                m_duration = measurement.get_last_duration()

                trace.append({
                    "role_id": "measurement",
                    "content": m_response,
                    "duration": m_duration,
                })
                last_reply = m_response

            else:
                # Patient turn (query or unknown)
                p_instruction = self.prompts.get_patient_turn_instruction(action_content)
                p_response = patient.step(p_instruction)
                p_duration = patient.get_last_duration()

                trace.append({
                    "role_id": "patient",
                    "content": p_response,
                    "duration": p_duration,
                })
                last_reply = self.extract_patient_response(p_response)

        # Collect role records for detailed logging
        role_records = self._collect_role_records([
            ("doctor", doctor),
            ("patient", patient),
            ("measurement", measurement),
        ])

        return ScenarioResult(
            answer=answer,
            trace=trace,
            metadata={
                "mode": "react",
                "turns": current_turn,
                "max_turns": self.max_turns,
                "role_records": role_records,
            },
        )

    def _collect_role_records(
        self, agents: list[tuple[str, "AgentWrapper"]]  # noqa: F821
    ) -> list[dict]:
        """Collect conversation history from all agents."""
        records = []
        for role_id, agent in agents:
            history = agent.get_history()
            token_usage = agent.get_usage_stats()
            records.append({
                role_id: history,
                "token_usage": token_usage,
            })
        return records
