"""
Core type definitions for PromptLab.

Defines the data structures used throughout the framework.
"""

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable


class MutationStrategy(Enum):
    """Strategies for mutating prompts."""

    REWRITE = "rewrite"  # Complete rewrite while maintaining semantics
    EXPAND = "expand"  # Add details and context
    SIMPLIFY = "simplify"  # Remove redundancy, simplify expression
    STRUCTURE = "structure"  # Change structure (lists, sections, markdown)
    STYLE = "style"  # Change style (formal, casual, technical)
    EXAMPLE = "example"  # Add or modify examples
    COT = "cot"  # Chain of Thought - add step-by-step reasoning
    FEW_SHOT = "few_shot"  # Few-shot learning - add multiple examples
    ROLE_PLAY = "role_play"  # Role-playing - add persona/role context


@dataclass
class Prompt:
    """Represents a single prompt in the evolution process."""

    content: str
    score: float = 0.0
    generation: int = 0
    parent_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        """Generate a unique ID for this prompt."""
        return str(hash(self.content))

    def to_dict(self) -> dict[str, Any]:
        """Convert Prompt to a dictionary for serialization."""
        return {
            "content": self.content,
            "score": self.score,
            "generation": self.generation,
            "parent_id": self.parent_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Prompt":
        """Create a Prompt from a dictionary."""
        return cls(
            content=data["content"],
            score=data.get("score", 0.0),
            generation=data.get("generation", 0),
            parent_id=data.get("parent_id"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class EvaluationResult:
    """Result of evaluating a prompt."""

    prompt: Prompt
    score: float
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class EvolutionConfig:
    """Configuration for the evolution process."""

    generations: int = 10
    population_size: int = 8
    mutation_strategies: list[MutationStrategy] = field(
        default_factory=lambda: [
            MutationStrategy.REWRITE,
            MutationStrategy.EXPAND,
            MutationStrategy.STRUCTURE,
        ]
    )
    keep_top: int = 3
    random_seed: int | None = None

    # LLM settings
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 2000

    def __post_init__(self):
        """Validate configuration."""
        if self.generations < 1:
            raise ValueError("generations must be at least 1")
        if self.population_size < 1:
            raise ValueError("population_size must be at least 1")
        if self.keep_top < 1:
            raise ValueError("keep_top must be at least 1")
        if self.keep_top > self.population_size:
            raise ValueError("keep_top cannot exceed population_size")


EvaluatorFn = Callable[[str], float]
EvaluatorFnDetailed = Callable[[str], EvaluationResult]


@dataclass
class EvolutionResult:
    """Result of the evolution process."""

    best_prompt: Prompt
    all_prompts: list[Prompt] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    generations_completed: int = 0
    converged: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert EvolutionResult to a dictionary for serialization."""
        return {
            "best_prompt": self.best_prompt.to_dict(),
            "all_prompts": [p.to_dict() for p in self.all_prompts],
            "history": self.history,
            "generations_completed": self.generations_completed,
            "converged": self.converged,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvolutionResult":
        """Create an EvolutionResult from a dictionary."""
        return cls(
            best_prompt=Prompt.from_dict(data["best_prompt"]),
            all_prompts=[Prompt.from_dict(p) for p in data.get("all_prompts", [])],
            history=data.get("history", []),
            generations_completed=data.get("generations_completed", 0),
            converged=data.get("converged", False),
        )

    def save(self, filepath: str) -> None:
        """Save the evolution result to a JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> "EvolutionResult":
        """Load an evolution result from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
