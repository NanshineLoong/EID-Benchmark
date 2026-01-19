"""Evaluation scenarios module.

Each scenario defines a specific evaluation mode with its own
interaction pattern between agents.
"""

from eid.scenarios.base import BaseScenario, ScenarioResult
from eid.scenarios.cot import CoTScenario
from eid.scenarios.roleplay import RoleplayScenario
from eid.scenarios.react import ReactScenario
from eid.scenarios.sc import SCScenario
from eid.scenarios.refine import RefineScenario

__all__ = [
    "BaseScenario",
    "ScenarioResult",
    "CoTScenario",
    "RoleplayScenario",
    "ReactScenario",
    "SCScenario",
    "RefineScenario",
    "get_scenario",
]


def get_scenario(
    mode: str,
    dataset_name: str,
    doctor_config: "ModelConfig",  # noqa: F821
    patient_config: "ModelConfig | None" = None,  # noqa: F821
    measurement_config: "ModelConfig | None" = None,  # noqa: F821
    max_turns: int = 16,
    summarizer_config: "ModelConfig | None" = None,  # noqa: F821
    diagnostician_config: "ModelConfig | None" = None,  # noqa: F821
    verifier_config: "ModelConfig | None" = None,  # noqa: F821
) -> BaseScenario:
    """Factory function to create a scenario by mode name.

    Args:
        mode: Scenario mode (cot, roleplay, react, sc, refine)
        dataset_name: Name of the dataset
        doctor_config: Model config for doctor role
        patient_config: Model config for patient simulator
        measurement_config: Model config for measurement simulator
        max_turns: Maximum interaction turns
        summarizer_config: Model config for summarizer role (SC/REFINE)
        diagnostician_config: Model config for diagnostician role (SC/REFINE)
        verifier_config: Model config for verifier role (REFINE)

    Returns:
        Configured scenario instance

    Raises:
        ValueError: If mode is not recognized
    """
    scenarios = {
        "cot": CoTScenario,
        "roleplay": RoleplayScenario,
        "react": ReactScenario,
        "sc": SCScenario,
        "refine": RefineScenario,
    }

    if mode not in scenarios:
        raise ValueError(f"Unknown scenario mode: {mode}. Available: {list(scenarios.keys())}")

    scenario_class = scenarios[mode]

    if mode == "cot":
        return scenario_class(dataset_name=dataset_name, doctor_config=doctor_config)

    # For roleplay modes, ensure simulator configs are provided
    if patient_config is None:
        patient_config = doctor_config
    if measurement_config is None:
        measurement_config = doctor_config

    # SC and REFINE modes need additional configs
    if mode == "sc":
        return scenario_class(
            dataset_name=dataset_name,
            doctor_config=doctor_config,
            patient_config=patient_config,
            measurement_config=measurement_config,
            max_turns=max_turns,
            summarizer_config=summarizer_config,
            diagnostician_config=diagnostician_config,
        )
    elif mode == "refine":
        return scenario_class(
            dataset_name=dataset_name,
            doctor_config=doctor_config,
            patient_config=patient_config,
            measurement_config=measurement_config,
            max_turns=max_turns,
            summarizer_config=summarizer_config,
            diagnostician_config=diagnostician_config,
            verifier_config=verifier_config,
        )

    return scenario_class(
        dataset_name=dataset_name,
        doctor_config=doctor_config,
        patient_config=patient_config,
        measurement_config=measurement_config,
        max_turns=max_turns,
    )
