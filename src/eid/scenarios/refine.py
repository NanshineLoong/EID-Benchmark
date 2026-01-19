"""REFINE scenario.

SC pipeline with a verification loop that can request more evidence
if the diagnosis is deemed incomplete.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from eid.agents import create_agent
from eid.prompts import PromptManager
from eid.scenarios.base import BaseScenario, CaseInput, ScenarioResult

if TYPE_CHECKING:
    from eid.config import ModelConfig
    from eid.agents.chat_agent import AgentWrapper


class RefineScenario(BaseScenario):
    """REFINE evaluation scenario.

    Extends SC with a diagnostician verifier that can reject incomplete
    diagnoses and request the doctor to gather more evidence.
    """

    def __init__(
        self,
        dataset_name: str,
        doctor_config: "ModelConfig",
        patient_config: "ModelConfig",
        measurement_config: "ModelConfig",
        max_turns: int = 16,
        summarizer_config: "ModelConfig | None" = None,
        diagnostician_config: "ModelConfig | None" = None,
        verifier_config: "ModelConfig | None" = None,
    ) -> None:
        """Initialize REFINE scenario.

        Args:
            dataset_name: Name of the dataset
            doctor_config: Model configuration for doctor role
            patient_config: Model configuration for patient simulator
            measurement_config: Model configuration for measurement simulator
            max_turns: Maximum interaction turns
            summarizer_config: Model configuration for summarizer role (default: doctor_config)
            diagnostician_config: Model configuration for diagnostician role (default: doctor_config)
            verifier_config: Model configuration for verifier role (default: doctor_config)
        """
        super().__init__(dataset_name)
        self.doctor_config = doctor_config
        self.patient_config = patient_config
        self.measurement_config = measurement_config
        self.max_turns = max_turns
        self.summarizer_config = summarizer_config or doctor_config
        self.diagnostician_config = diagnostician_config or doctor_config
        self.verifier_config = verifier_config or doctor_config
        self.prompts = PromptManager(dataset_name)

    def run(self, case_input: CaseInput) -> ScenarioResult:
        """Execute REFINE scenario on a single case.

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

        measurement = create_agent(
            role_id="measurement",
            system_prompt=self.prompts.get_measurement_system_prompt().format(
                exam_facts=exam_facts_str
            ),
            config=self.measurement_config,
            message_window_size=1,
            summarize_threshold=80,
        )

        summarizer = create_agent(
            role_id="summarizer",
            system_prompt=self.prompts.get_summarizer_system_prompt(),
            config=self.summarizer_config,
            message_window_size=3,
            summarize_threshold=80,
        )

        diagnostician = create_agent(
            role_id="diagnostician",
            system_prompt=self.prompts.get_diagnostician_system_prompt(),
            config=self.diagnostician_config,
            message_window_size=3,
            summarize_threshold=90,
        )

        verifier = create_agent(
            role_id="diagnostician_verifier",
            system_prompt=self.prompts.get_diagnostician_verifier_system_prompt(),
            config=self.verifier_config,
            message_window_size=3,
            summarize_threshold=80,
        )

        # Run interaction loop
        trace: list[dict] = []
        dialogue_history = ""
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

            # Update dialogue history
            dialogue_history += f"\nDoctor: {action_content}\n"

            # Check for finish or turn limit
            if action_type == "finish" or current_turn >= self.max_turns - 1:
                # Run verification pipeline
                answer, verified = self._run_verification_pipeline(
                    trace=trace,
                    dialogue_history=dialogue_history,
                    summarizer=summarizer,
                    diagnostician=diagnostician,
                    verifier=verifier,
                    doctor=doctor,
                    patient=patient,
                    measurement=measurement,
                    current_turn=current_turn,
                )

                if verified or current_turn >= self.max_turns - 1:
                    break

                # Verifier requested more evidence - continue the loop
                current_turn += 1
                continue

            current_turn += 1

            # Route to appropriate simulator
            if action_type == "test":
                m_instruction = self.prompts.get_measurement_turn_instruction(action_content)
                m_response = measurement.step(m_instruction)
                m_duration = measurement.get_last_duration()

                trace.append({
                    "role_id": "measurement",
                    "content": m_response,
                    "duration": m_duration,
                })
                last_reply = m_response
                dialogue_history += f"\nMeasurement: {m_response}\n"

            else:
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

        # Collect role records
        role_records = self._collect_role_records([
            ("doctor", doctor),
            ("patient", patient),
            ("measurement", measurement),
            ("summarizer", summarizer),
            ("diagnostician", diagnostician),
            ("verifier", verifier),
        ])

        return ScenarioResult(
            answer=answer,
            trace=trace,
            metadata={
                "mode": "refine",
                "turns": current_turn,
                "max_turns": self.max_turns,
                "role_records": role_records,
            },
        )

    def _run_verification_pipeline(
        self,
        trace: list[dict],
        dialogue_history: str,
        summarizer: "AgentWrapper",
        diagnostician: "AgentWrapper",
        verifier: "AgentWrapper",
        doctor: "AgentWrapper",
        patient: "AgentWrapper",
        measurement: "AgentWrapper",
        current_turn: int,
    ) -> tuple[str, bool]:
        """Run the summarizer -> diagnostician -> verifier pipeline.

        Args:
            trace: Interaction trace list
            dialogue_history: Accumulated dialogue
            summarizer: Summarizer agent
            diagnostician: Diagnostician agent
            verifier: Verifier agent
            doctor: Doctor agent (for feedback loop)
            patient: Patient agent (for feedback loop)
            measurement: Measurement agent (for feedback loop)
            current_turn: Current turn count

        Returns:
            Tuple of (answer, verified) where verified indicates if diagnosis was accepted
        """
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

        _, answer = self.extract_action(d_response)
        if not answer:
            answer = d_response

        # Verifier phase
        v_instruction = self.prompts.get_verifier_instruction(
            current_turns=current_turn,
            max_turns=self.max_turns,
            summary=summary,
            diagnosis=answer,
        )
        v_response = verifier.step(v_instruction)
        v_duration = verifier.get_last_duration()

        trace.append({
            "role_id": "diagnostician_verifier",
            "content": v_response,
            "duration": v_duration,
        })

        decision, feedback = self.extract_verifier_decision(v_response)

        # If incomplete and not at turn limit, request more evidence
        if decision == "INCOMPLETE" and current_turn < self.max_turns - 1:
            # Doctor receives feedback and gathers more evidence
            fb_instruction = self.prompts.get_doctor_feedback_instruction(feedback)
            fb_response = doctor.step(fb_instruction)
            fb_duration = doctor.get_last_duration()

            trace.append({
                "role_id": "doctor",
                "content": fb_response,
                "duration": fb_duration,
            })

            # Parse doctor's response to feedback
            action_type, action_content = self.extract_action(fb_response)

            # Route to appropriate simulator
            if action_type == "test":
                m_instruction = self.prompts.get_measurement_turn_instruction(action_content)
                m_response = measurement.step(m_instruction)
                m_duration = measurement.get_last_duration()

                trace.append({
                    "role_id": "measurement",
                    "content": m_response,
                    "duration": m_duration,
                })
            else:
                p_instruction = self.prompts.get_patient_turn_instruction(action_content)
                p_response = patient.step(p_instruction)
                p_duration = patient.get_last_duration()

                trace.append({
                    "role_id": "patient",
                    "content": p_response,
                    "duration": p_duration,
                })

            return answer, False  # Not verified, continue gathering

        return answer, True  # Verified (PASS or at turn limit)

    def _collect_role_records(
        self, agents: list[tuple[str, "AgentWrapper"]]
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
