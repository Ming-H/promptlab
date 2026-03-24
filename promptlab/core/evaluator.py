"""
Prompt evaluation framework.

Evaluates prompts using user-provided evaluation functions.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from promptlab.core.types import EvaluationResult, Prompt, EvaluatorFn, EvaluatorFnDetailed


class Evaluator:
    """
    Evaluates prompts using custom evaluation functions.

    The evaluator is responsible for running a prompt through
    an evaluation function and collecting the results.
    """

    def __init__(
        self,
        evaluator_fn: EvaluatorFn | EvaluatorFnDetailed,
    ):
        """
        Initialize the Evaluator.

        Args:
            evaluator_fn: A function that takes a prompt string and returns
                         either a float (score) or an EvaluationResult
        """
        self.evaluator_fn = evaluator_fn

    def evaluate(self, prompt: Prompt | str) -> EvaluationResult:
        """
        Evaluate a single prompt.

        Args:
            prompt: The prompt to evaluate

        Returns:
            An EvaluationResult containing the score and any additional metrics
        """
        if isinstance(prompt, str):
            prompt = Prompt(content=prompt)

        try:
            result = self.evaluator_fn(prompt.content)

            if isinstance(result, float):
                return EvaluationResult(
                    prompt=prompt,
                    score=result,
                    metrics={},
                )
            elif isinstance(result, EvaluationResult):
                # Update the prompt reference
                result.prompt = prompt
                return result
            else:
                raise TypeError(
                    f"Evaluator function must return float or EvaluationResult, got {type(result)}"
                )
        except Exception as e:
            return EvaluationResult(
                prompt=prompt,
                score=0.0,
                error=str(e),
            )

    def evaluate_many(
        self,
        prompts: list[Prompt | str],
        parallel: bool = False,
        max_workers: int = 4,
    ) -> list[EvaluationResult]:
        """
        Evaluate multiple prompts.

        Args:
            prompts: List of prompts to evaluate
            parallel: Whether to use parallel execution
            max_workers: Maximum number of parallel workers (only used if parallel=True)

        Returns:
            List of EvaluationResult objects in the same order as input
        """
        if parallel and len(prompts) > 1:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self.evaluate, prompt): i
                    for i, prompt in enumerate(prompts)
                }
                results = [None] * len(prompts)
                for future in as_completed(futures):
                    idx = futures[future]
                    results[idx] = future.result()
                return results
        else:
            # Sequential execution
            return [self.evaluate(p) for p in prompts]
