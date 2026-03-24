"""
PromptLab - AI Prompt Evolution Framework

A framework for automatically evolving and optimizing AI prompts
using genetic algorithms and LLM-based mutation.
"""

from promptlab.core.evolution import EvolutionEngine, evolve_prompt
from promptlab.core.evaluator import Evaluator
from promptlab.core.mutator import Mutator
from promptlab.core.types import (
    Prompt,
    MutationStrategy,
    EvolutionConfig,
    EvaluationResult,
    EvolutionResult,
)

__version__ = "0.1.0"

__all__ = [
    "EvolutionEngine",
    "evolve_prompt",
    "Evaluator",
    "Mutator",
    "Prompt",
    "MutationStrategy",
    "EvolutionConfig",
    "EvaluationResult",
    "EvolutionResult",
]
