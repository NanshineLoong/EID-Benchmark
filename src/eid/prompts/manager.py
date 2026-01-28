"""Prompt templates for different roles and scenarios.

This module manages all prompt templates used in the evaluation scenarios.
"""

from __future__ import annotations

# Dataset categories
DIFFERENTIAL_DATASETS = {"diagnosisarena", "clinicalbench", "rarearena", "rarearena_rdc"}
FINAL_DATASETS = {"medqa", "derm", "nejm", "agentclinic_medqa"}

# Task descriptions
TASKS = {
    "diagnosis": {
        "description": (
            "The task objective is to identify the single most likely diagnosis based on the collected evidence."
        ),
        "output_format": "The answer in the format of a single most likely diagnosis.",
    },
    "differential_diagnosis": {
        "description": (
            "The task objective is to formulate a differential diagnosis by identifying and ranking "
            "the top 5 potential diagnoses based on the collected evidence."
        ),
        "output_format": (
            "The answer in the format of a ranked list of the top 5 potential diagnoses, ordered from most likely to least likely."
        ),
    },
}


class PromptManager:
    """Manages prompt templates for different evaluation scenarios.

    Attributes:
        dataset_name: Name of the dataset being evaluated
        task_type: Type of task (diagnosis or differential_diagnosis)
    """

    def __init__(self, dataset_name: str) -> None:
        """Initialize prompt manager.

        Args:
            dataset_name: Name of the dataset
        """
        self.dataset_name = dataset_name.lower()
        self.task_type = (
            "diagnosis" if self.dataset_name in FINAL_DATASETS else "differential_diagnosis"
        )
        self.task_description = TASKS[self.task_type]["description"]
        self.task_output_format = TASKS[self.task_type]["output_format"]

    # =========================================================================
    # CoT Mode Prompts
    # =========================================================================

    def get_cot_instruction(self, task: str) -> str:
        """Get Chain-of-Thought instruction template.

        Args:
            task: The case description/task

        Returns:
            Formatted instruction string
        """
        return (
            f"{task}\n\n"
            "Reason step by step before making the diagnosis. Follow the format:\n"
            "THOUGHT: <your reasoning>\n"
            f"DIAGNOSIS: <{self.task_output_format}>\n"
        )

    # =========================================================================
    # Simulator Role Prompts
    # =========================================================================

    def get_patient_system_prompt(self) -> str:
        """Get system prompt for patient simulator."""
        return (
            "You are a patient undergoing a medical interview.\n"
            "Your knowledge is strictly limited to the following list of indexed facts:\n"
            "{patient_facts}\n"
            "\n"
            "Response Protocols:\n"
            "1. Analyze the doctor's question and search your list for the specific item(s) that contain the answer.\n"
            "2. You must format your output using two distinct tags:\n"
            "   [REFERENCE] followed by the EXACT string(s) (including the index) from your list. You may select UP TO TWO facts if necessary to fully answer the question. If no fact exists, write 'N/A'.\n"
            "   [RESPONSE] followed by a natural language answer derived STRICTLY from the combined content of the selected [REFERENCE]s. Do not add outside information.\n"
            "3. If the doctor's question is not addressed by any fact in your list:\n"
            "   [REFERENCE] N/A\n"
            "   [RESPONSE] followed by a natural statement indicating you are unsure or do not recall."
        )

    def get_reporter_system_prompt(self) -> str:
        """Get system prompt for reporter simulator."""
        return (
            "You are a specialized module named 'Measurement' responsible for reporting "
            "test results to the physician.\n"
            "You have access to the following list of indexed facts:\n"
            "Physical Examination and Diagnostic Test Data:\n"
            "{exam_facts}\n"
            "\n"
            "Response Protocols:\n"
            "1. Search the provided list for ALL facts that are relevant to the doctor's specific test request.Do not provide information that was not explicitly requested.\n"
            "2. Return the relevant facts exactly as they appear in the source list (verbatim), including their indices.\n"
            "3. If the requested test results are not found in the list, assume the finding is non-significant and return 'Normal'.\n"
        )

    # =========================================================================
    # Base Roleplay Mode Prompts
    # =========================================================================

    def get_base_doctor_system_prompt(self) -> str:
        """Get system prompt for base roleplay doctor."""
        return (
            "You are a licensed physician conducting a medical consultation.\n"
            f"{self.task_description}\n"
            "Your objective is to efficiently gather information and request necessary "
            "clinical examinations or laboratory tests to enable a subsequent diagnostic analysis.\n"
            "You have access to a Medical Analyst who can retrieve specific test results upon request.\n"
            "\n"
            "You must adhere to the following operational constraints:\n"
            "1. Efficiency: Gather sufficient information in as few turns as possible.\n"
            "2. Turn Limit: You strictly cannot exceed {max_turns} total turns.\n"
            "3. No Repetition: Never ask a question or request a test that has already been covered.\n"
            "4. Atomic Inquiries: Each question must address a single, specific topic "
            "(e.g., ask 'What are your symptoms?', never combine multiple questions).\n"
            "\n"
            "In every turn, you execute one of the following actions in the corresponding format:\n"
            "1. [QUERY] followed by your atomic question to the patient.\n"
            "2. [TEST] followed by one specific examination or diagnostic test request to the Medical Analyst.\n"
            f"3. [DIAGNOSIS] followed by {self.task_output_format}\n"
            "\n"
            "Once you have gathered sufficient evidence, ensure your diagnosis is final.\n"
        )

    # =========================================================================
    # ReAct Mode Prompts
    # =========================================================================

    def get_react_doctor_system_prompt(self) -> str:
        """Get system prompt for ReAct mode doctor."""
        return (
            "You are a licensed physician conducting a medical consultation.\n"
            f"{self.task_description}\n"
            "Your objective is to efficiently gather information and request necessary "
            "clinical examinations or laboratory tests to enable a subsequent diagnostic analysis.\n"
            "You have access to a Medical Analyst who can retrieve specific test results upon request.\n"
            "\n"
            "You must adhere to the following operational constraints:\n"
            "1. Efficiency: Gather sufficient information in as few turns as possible.\n"
            "2. Turn Limit: You strictly cannot exceed {max_turns} total turns.\n"
            "3. No Repetition: Never ask a question or request a test that has already been covered.\n"
            "4. Atomic Inquiries: Each question must address a single, specific topic.\n"
            "\n"
            "In every turn, you must follow a strict 'Reasoning-then-Acting' process:\n"
            "\n"
            "[THOUGHT] <Your Clinical Reasoning>\n"
            "   - Analyze the current clinical picture, identify critical information gaps, "
            "and articulate step-by-step reasoning to justify your next action.\n"
            "\n"
            "Execute exactly ONE of the following commands:\n"
            "   - [QUERY] followed by your atomic question to the patient.\n"
            "   - [TEST] followed by one specific examination or diagnostic test request.\n"
            f"   - [DIAGNOSIS] followed by {self.task_output_format}\n"
            "\n"
            "Once you have gathered sufficient evidence, ensure your diagnosis is final.\n"
        )

    # =========================================================================
    # SC (Summarizer-Diagnostician) Mode Prompts
    # =========================================================================

    def get_sc_doctor_system_prompt(self) -> str:
        """Get system prompt for SC mode doctor (evidence gatherer)."""
        return (
            "You are a licensed physician conducting a medical consultation.\n"
            f"{self.task_description}\n"
            "Your objective is to efficiently gather information and request necessary "
            "clinical examinations or laboratory tests.\n"
            "You have access to a Medical Analyst who can retrieve specific test results upon request.\n"
            "\n"
            "You must adhere to the following operational constraints:\n"
            "1. Efficiency: Gather sufficient information in as few turns as possible.\n"
            "2. Turn Limit: You strictly cannot exceed {max_turns} total turns.\n"
            "3. No Repetition: Never ask a question or request a test that has already been covered.\n"
            "4. Atomic Inquiries: Each question must address a single, specific topic.\n"
            "\n"
            "In every turn, follow a strict 'Reasoning-then-Acting' process:\n"
            "\n"
            "[THOUGHT] <Your Clinical Reasoning>\n"
            "   - Analyze the current clinical picture and identify critical information gaps.\n"
            "\n"
            "Execute exactly ONE of the following commands:\n"
            "   - [QUERY] followed by your atomic question to the patient.\n"
            "   - [TEST] followed by one specific examination or diagnostic test request.\n"
            "   - [FINISH] use this command ONLY when you believe you have gathered all "
            "necessary information to form a conclusive diagnosis. You don't need to make a diagnosis.\n"
            "\n"
            "Once you issue the [FINISH] command, the consultation ends immediately.\n"
        )

    def get_summarizer_system_prompt(self) -> str:
        """Get system prompt for summarizer role."""
        return (
            "You are a professional medical documentarian and clinical scribe.\n"
            "Your objective is to synthesize the dialogue between a doctor and a patient "
            "into a high-fidelity structured medical summary.\n"
            "\n"
            "Core Principles:\n"
            "1. Strict Adherence: You must NOT invent, infer, or hallucinate any "
            "information not explicitly present in the dialogue.\n"
            "2. Precision: Retain all precise measurements, dates, dosages, and "
            "technical medical terms exactly as stated.\n"
            "3. Objectivity: Maintain a professional, clinical tone throughout the summary.\n"
            "\n"
            "Output Process:\n"
            "[THOUGHT]\n"
            "Analyze the dialogue to extract key clinical facts and reasoning.\n"
            "\n"
            "[SUMMARY]\n"
            "Generate a professional, structured clinical note.\n"
        )

    def get_diagnostician_system_prompt(self) -> str:
        """Get system prompt for diagnostician role."""
        return (
            "You are a senior diagnostic physician specializing in complex differential diagnosis.\n"
            f"{self.task_description}\n"
            "Your objective is to analyze the provided structured clinical summary "
            "to formulate a precise diagnosis.\n"
            "\n"
            "You must follow a strict reasoning process:\n"
            "\n"
            "[THOUGHT] <Your Clinical Reasoning>\n"
            "   - Perform a comprehensive clinical analysis of the summary.\n"
            "\n"
            "[DIAGNOSIS]\n"
            f"   - Provide the {self.task_output_format}.\n"
        )

    # =========================================================================
    # REFINE Mode Prompts
    # =========================================================================

    def get_diagnostician_verifier_system_prompt(self) -> str:
        """Get system prompt for diagnostician verifier role."""
        return (
            "You are a Clinical Diagnostic Supervisor.\n"
            f"{self.task_description}\n"
            "Your objective is to evaluate sufficiency of the diagnosis provided by "
            "the physician, based strictly on the available case summarized information.\n"
            "\n"
            "Evaluation Criteria:\n"
            "1. Data Sufficiency: Determine if the current information is actually "
            "sufficient to form a conclusive diagnosis.\n"
            "2. Turn Limit Override: If the maximum turn limit has been reached, "
            "you must force a decision (PASS or REJECT) based on the best possible "
            "interpretation of existing data.\n"
            "\n"
            "Output Format:\n"
            "\n"
            "[THOUGHT] <Your Analysis>\n"
            "   - Identify if any 'Red Flag' symptoms or critical tests are missing.\n"
            "\n"
            "[DECISION] <Status>\n"
            "   - Output 'PASS' if the diagnosis is sufficient.\n"
            "   - Output 'INCOMPLETE' if critical clinical information is missing "
            "(requires the Physician to gather more data; only valid if not at max turns).\n"
            "\n"
            "[FEEDBACK] <Guidance>\n"
            "   - If PASS: Leave this section empty.\n"
            "   - If INCOMPLETE: Specify exactly what critical information is required.\n"
        )

    # =========================================================================
    # Instruction Templates
    # =========================================================================

    def get_doctor_turn_instruction(
        self, current_turns: int, max_turns: int, last_reply: str
    ) -> str:
        """Get instruction for doctor's turn.

        Args:
            current_turns: Current turn number
            max_turns: Maximum allowed turns
            last_reply: Last response from patient/measurement

        Returns:
            Formatted instruction string
        """
        return (
            f"Turns used: {current_turns} / {max_turns}.\n"
            f"Last reply:\n{last_reply}\n\n"
            "Doctor:"
        )

    def get_patient_turn_instruction(self, doctor_message: str) -> str:
        """Get instruction for patient's turn.

        Args:
            doctor_message: Doctor's query

        Returns:
            Formatted instruction string
        """
        return f"Doctor said:\n{doctor_message}\n\nPatient:"

    def get_reporter_turn_instruction(self, doctor_message: str) -> str:
        """Get instruction for reporter's turn.

        Args:
            doctor_message: Doctor's test request

        Returns:
            Formatted instruction string
        """
        return f"Doctor's request:\n{doctor_message}\n\nMeasurement:"

    def get_summarizer_instruction(self, dialogue_history: str) -> str:
        """Get instruction for summarizer.

        Args:
            dialogue_history: Full dialogue history

        Returns:
            Formatted instruction string
        """
        return f"### Dialogue History ###\n{dialogue_history}\n\n"

    def get_diagnostician_instruction(self, summary: str) -> str:
        """Get instruction for diagnostician.

        Args:
            summary: Case summary from summarizer

        Returns:
            Formatted instruction string
        """
        return f"### Case Summary ###\n{summary}\n\nDiagnostician:"

    def get_verifier_instruction(
        self, current_turns: int, max_turns: int, summary: str, diagnosis: str
    ) -> str:
        """Get instruction for diagnostician verifier.

        Args:
            current_turns: Current turn number
            max_turns: Maximum allowed turns
            summary: Case summary
            diagnosis: Proposed diagnosis

        Returns:
            Formatted instruction string
        """
        return (
            f"Turns: {current_turns} / {max_turns}.\n"
            f"### Case Summary ###\n{summary}\n\n"
            f"### Proposed Diagnosis ###\n{diagnosis}\n\n"
        )

    def get_doctor_feedback_instruction(self, feedback: str) -> str:
        """Get instruction for doctor after verifier feedback.

        Args:
            feedback: Feedback from verifier

        Returns:
            Formatted instruction string
        """
        return (
            "### DIAGNOSTIC FEEDBACK (RESUMED) ###\n"
            "Your previous decision to [FINISH] was rejected because the clinical data is INCOMPLETE.\n"
            f"Specific Feedback from Supervisor:\n{feedback}\n"
            "\n"
            "IMMEDIATE INSTRUCTION:\n"
            "1. Analyze the specific information gaps identified in the feedback above.\n"
            "2. Your NEXT action must be a [QUERY] or [TEST] strictly targeted to acquire "
            "this missing evidence.\n"
            "Doctor:"
        )
