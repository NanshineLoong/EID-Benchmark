"""SC (Summarized-Conversation) scenario.

Multi-turn interaction with a separate summarizer and diagnostician pipeline.
Doctor gathers evidence, summarizer creates clinical note, diagnostician makes diagnosis.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from eid.agents import create_agent
from eid.prompts import PromptManager
from eid.scenarios.base import BaseScenario, CaseInput, ScenarioResult

if TYPE_CHECKING:
    from eid.config import ModelConfig


class SCScenario(BaseScenario):
    """SC (Summarized-Conversation) evaluation scenario.

    The doctor gathers evidence using [QUERY], [TEST], and [FINISH] commands.
    When finished, a summarizer creates a clinical note, and a diagnostician
    produces the final diagnosis based on the summary.
    """

    def __init__(
        self,
        dataset_name: str,
        doctor_config: "ModelConfig",
        patient_config: "ModelConfig",
        reporter_config: "ModelConfig",
        max_turns: int = 16,
        summarizer_config: "ModelConfig | None" = None,
        diagnostician_config: "ModelConfig | None" = None,
    ) -> None:
        """Initialize SC scenario.

        Args:
            dataset_name: Name of the dataset
            doctor_config: Model configuration for doctor role
            patient_config: Model configuration for patient simulator
            reporter_config: Model configuration for reporter simulator
            max_turns: Maximum interaction turns
            summarizer_config: Model configuration for summarizer role (default: doctor_config)
            diagnostician_config: Model configuration for diagnostician role (default: doctor_config)
        """
        super().__init__(dataset_name)
        self.doctor_config = doctor_config
        self.patient_config = patient_config
        self.reporter_config = reporter_config
        self.max_turns = max_turns
        self.summarizer_config = summarizer_config or doctor_config
        self.diagnostician_config = diagnostician_config or doctor_config
        self.prompts = PromptManager(dataset_name)

    def run(self, case_input: CaseInput) -> ScenarioResult:
        """Execute SC scenario on a single case.

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
            system_prompt=self.prompts.get_sc_doctor_system_prompt().format(
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

        reporter = create_agent(
            role_id="reporter",
            system_prompt=self.prompts.get_reporter_system_prompt().format(
                exam_facts=exam_facts_str
            ),
            config=self.reporter_config,
            message_window_size=1,
            summarize_threshold=80,
        )

        summarizer = create_agent(
            role_id="summarizer",
            system_prompt=self.prompts.get_summarizer_system_prompt(),
            config=self.summarizer_config,
        )

        diagnostician = create_agent(
            role_id="diagnostician",
            system_prompt=self.prompts.get_diagnostician_system_prompt(),
            config=self.diagnostician_config,
        )

        # Run interaction loop
        trace: list[dict] = []
        dialogue_history = ""
        last_reply = ""
        answer = ""
        current_turn = 0
        finished = False

        while current_turn <= self.max_turns and not finished:
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

            # Update dialogue history
            dialogue_history += f"\nDoctor: {action_content}\n"

            if action_type == "finish" or current_turn >= self.max_turns:
                finished = True
                break

            current_turn += 1

            # Route to appropriate simulator
            if action_type == "test":
                # Reporter turn
                m_instruction = self.prompts.get_reporter_turn_instruction(action_content)
                m_response = reporter.step(m_instruction)
                m_duration = reporter.get_last_duration()

                trace.append({
                    "role_id": "reporter",
                    "content": m_response,
                    "duration": m_duration,
                })
                last_reply = m_response
                dialogue_history += f"\nMeasurement: {m_response}\n"

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
                dialogue_history += f"\nPatient: {last_reply}\n"

        # Summarizer phase
        s_instruction = self.prompts.get_summarizer_instruction(dialogue_history)
        s_response = summarizer.step(s_instruction)
        s_duration = summarizer.get_last_duration()

        trace.append({
            "role_id": "summarizer",
            "content": s_response,
            "duration": s_duration,
        })

        summary = self.extract_summary(s_response)

        # Diagnostician phase
        d_instruction = self.prompts.get_diagnostician_instruction(summary)
        d_response = diagnostician.step(d_instruction)
        d_duration = diagnostician.get_last_duration()

        trace.append({
            "role_id": "diagnostician",
            "content": d_response,
            "duration": d_duration,
        })

        # Extract diagnosis
        _, answer = self.extract_action(d_response)
        if not answer:
            answer = d_response

        # Collect role records for detailed logging
        role_records = self._collect_role_records([
            ("doctor", doctor),
            ("patient", patient),
            ("reporter", reporter),
            ("summarizer", summarizer),
            ("diagnostician", diagnostician),
        ])

        return ScenarioResult(
            answer=answer,
            trace=trace,
            metadata={
                "mode": "sc",
                "turns": current_turn,
                "max_turns": self.max_turns,
                "summary": summary,
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
