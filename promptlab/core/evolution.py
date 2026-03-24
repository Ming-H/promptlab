"""
Evolution engine for prompt optimization.

Coordinates the mutation, evaluation, and selection process.
"""

import random
from typing import Any

from promptlab.core.types import (
    EvolutionConfig,
    EvolutionResult,
    MutationStrategy,
    Prompt,
)
from promptlab.core.evaluator import Evaluator
from promptlab.core.mutator import Mutator
from promptlab.utils.llm import LLMClient, create_llm_client


class EvolutionEngine:
    """
    Manages the evolution of prompts over multiple generations.

    The engine uses a genetic algorithm approach:
    1. Start with an initial prompt
    2. Generate mutations
    3. Evaluate each mutation
    4. Select the top performers
    5. Repeat for N generations
    """

    def __init__(
        self,
        config: EvolutionConfig | None = None,
        llm_client: LLMClient | None = None,
    ):
        """
        Initialize the EvolutionEngine.

        Args:
            config: Configuration for the evolution process
            llm_client: LLM client to use for mutations
        """
        self.config = config or EvolutionConfig()
        self.llm = llm_client or create_llm_client(
            provider="mock",
            model=self.config.model,
        )

        self.mutator = Mutator(
            llm_client=self.llm,
            strategies=self.config.mutation_strategies,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)

    def evolve(
        self,
        initial_prompt: str | Prompt,
        evaluator: Any,  # Callable[[str], float] or Evaluator
        generations: int | None = None,
    ) -> EvolutionResult:
        """
        Run the evolution process.

        Args:
            initial_prompt: The starting prompt
            evaluator: Function or Evaluator to score prompts
            generations: Number of generations (overrides config)

        Returns:
            EvolutionResult containing the best prompt and history
        """
        generations = generations or self.config.generations

        # Normalize inputs
        if isinstance(initial_prompt, str):
            current_prompt = Prompt(content=initial_prompt)
        else:
            current_prompt = initial_prompt

        if not isinstance(evaluator, Evaluator):
            evaluator = Evaluator(evaluator)

        # Track all prompts and history
        all_prompts: list[Prompt] = [current_prompt]
        history: list[dict[str, Any]] = []
        best_prompt = current_prompt
        best_score = -float("inf")

        # Evaluate initial prompt
        initial_result = evaluator.evaluate(current_prompt)
        current_prompt.score = initial_result.score
        if current_prompt.score > best_score:
            best_score = current_prompt.score
            best_prompt = current_prompt

        for gen in range(generations):
            # Generate mutations
            mutations = self.mutator.mutate_many(
                current_prompt,
                n=self.config.population_size,
                allow_duplicate_strategies=True,
            )

            # Evaluate all mutations
            results = evaluator.evaluate_many(mutations)

            # Update scores and track best
            gen_best_score = -float("inf")
            gen_best_prompt = current_prompt

            for result in results:
                result.prompt.score = result.score
                all_prompts.append(result.prompt)

                if result.score > gen_best_score:
                    gen_best_score = result.score
                    gen_best_prompt = result.prompt

                if result.score > best_score:
                    best_score = result.score
                    best_prompt = result.prompt

            # Record generation stats
            gen_history = {
                "generation": gen + 1,
                "best_score": gen_best_score,
                "avg_score": sum(r.score for r in results) / len(results),
                "population_size": len(results),
            }
            history.append(gen_history)

            # Check for convergence (no improvement)
            if gen > 0 and history[-2]["best_score"] == gen_best_score:
                # No improvement, but continue for potential future improvement
                pass

            # Select best for next generation
            current_prompt = gen_best_prompt

        return EvolutionResult(
            best_prompt=best_prompt,
            all_prompts=all_prompts,
            history=history,
            generations_completed=generations,
            converged=history[-1]["best_score"] == history[0]["best_score"] if history else False,
        )


def evolve_prompt(
    initial_prompt: str,
    evaluator: Any,  # Callable[[str], float]
    generations: int = 10,
    population_size: int = 8,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
) -> str:
    """
    Convenience function to evolve a prompt.

    Args:
        initial_prompt: The starting prompt
        evaluator: Function that takes a prompt string and returns a score (0-1)
        generations: Number of generations to evolve
        population_size: Number of mutations per generation
        model: LLM model to use
        temperature: Temperature for mutations

    Returns:
        The evolved prompt string
    """
    # Calculate keep_top to be at most population_size
    keep_top = min(3, population_size)

    config = EvolutionConfig(
        generations=generations,
        population_size=population_size,
        keep_top=keep_top,
        model=model,
        temperature=temperature,
    )

    engine = EvolutionEngine(config)
    result = engine.evolve(initial_prompt, evaluator)

    return result.best_prompt.content
