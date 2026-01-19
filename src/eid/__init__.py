"""
EID-Benchmark: Evidence Elicitation in Interactive Diagnosis Benchmark

This package provides evaluation tools for assessing LLM diagnostic capabilities
through multi-turn patient-doctor interactions.
"""

from eid.config import ModelConfig, get_model_config, load_config
from eid.benchmark import Benchmark

__version__ = "0.1.0"

__all__ = [
    "Benchmark",
    "ModelConfig",
    "get_model_config",
    "load_config",
]
