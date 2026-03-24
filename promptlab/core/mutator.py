"""
Prompt mutation engine.

Generates variations of prompts using different strategies.
"""

import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from promptlab.core.types import MutationStrategy, Prompt
from promptlab.utils.llm import LLMClient, create_llm_client


# Strategy-specific mutation prompts
_MUTATION_TEMPLATES = {
    MutationStrategy.REWRITE: """Rewrite the following prompt to be more effective while maintaining its core meaning and intent. Output only the rewritten prompt, no explanation.

Original prompt: {prompt}

Rewritten prompt:""",

    MutationStrategy.EXPAND: """Expand the following prompt by adding relevant details, context, and clarifications that will improve its effectiveness. Output only the expanded prompt, no explanation.

Original prompt: {prompt}

Expanded prompt:""",

    MutationStrategy.SIMPLIFY: """Simplify the following prompt by removing redundancy, wordiness, and unnecessary complexity while keeping its core meaning. Output only the simplified prompt, no explanation.

Original prompt: {prompt}

Simplified prompt:""",

    MutationStrategy.STRUCTURE: """Reformat the following prompt to improve its structure (using markdown formatting, bullet points, sections, etc.) to make it clearer and more effective. Output only the reformatted prompt, no explanation.

Original prompt: {prompt}

Reformatted prompt:""",

    MutationStrategy.STYLE: """Change the style of the following prompt to make it more direct and actionable. Use clear, precise language. Output only the restyled prompt, no explanation.

Original prompt: {prompt}

Restyled prompt:""",

    MutationStrategy.EXAMPLE: """Enhance the following prompt by adding a concrete example that clarifies what kind of output is expected. Output only the enhanced prompt with the example, no explanation.

Original prompt: {prompt}

Enhanced prompt with example:""",

    MutationStrategy.COT: """Enhance the following prompt by adding chain-of-thought reasoning instructions. Add explicit steps that guide the AI to think through the problem systematically. Output only the enhanced prompt, no explanation.

Original prompt: {prompt}

Enhanced prompt with chain-of-thought:""",

    MutationStrategy.FEW_SHOT: """Enhance the following prompt by adding 2-3 diverse examples that demonstrate the expected input-output patterns. Output only the enhanced prompt with examples, no explanation.

Original prompt: {prompt}

Enhanced prompt with few-shot examples:""",

    MutationStrategy.ROLE_PLAY: """Enhance the following prompt by adding a role or persona that the AI should adopt when responding. Define the expertise, perspective, and tone appropriate for the task. Output only the enhanced prompt, no explanation.

Original prompt: {prompt}

Enhanced prompt with role-play:""",
}


class Mutator:
    """
    Generates mutated versions of prompts using LLM-based strategies.

    Each mutation strategy uses a different prompt template to guide
    the LLM in creating a variation of the original prompt.
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        strategies: list[MutationStrategy] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ):
        """
        Initialize the Mutator.

        Args:
            llm_client: LLM client to use for mutations
            strategies: List of mutation strategies to use
            temperature: Temperature for LLM generation
            max_tokens: Maximum tokens for LLM generation
        """
        self.llm = llm_client or create_llm_client("mock")
        self.strategies = strategies or [
            MutationStrategy.REWRITE,
            MutationStrategy.EXPAND,
            MutationStrategy.STRUCTURE,
        ]
        self.temperature = temperature
        self.max_tokens = max_tokens

    def mutate(
        self,
        prompt: Prompt | str,
        strategy: MutationStrategy | None = None,
    ) -> Prompt:
        """
        Generate a single mutation of the given prompt.

        Args:
            prompt: The prompt to mutate
            strategy: The mutation strategy to use (random if None)

        Returns:
            A new Prompt object with the mutated content
        """
        if isinstance(prompt, str):
            prompt = Prompt(content=prompt)

        # Select strategy if not provided
        if strategy is None:
            strategy = random.choice(self.strategies)

        # Generate mutation
        template = _MUTATION_TEMPLATES.get(strategy)
        if not template:
            raise ValueError(f"Unknown strategy: {strategy}")

        mutation_prompt = template.format(prompt=prompt.content)
        new_content = self.llm.complete(
            mutation_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Clean up the result (remove common prefixes)
        new_content = new_content.strip()
        for prefix in ["Here's the rewritten prompt:", "Rewritten prompt:", "Result:"]:
            if new_content.startswith(prefix):
                new_content = new_content[len(prefix):].strip()

        return Prompt(
            content=new_content,
            generation=prompt.generation + 1,
            parent_id=prompt.id,
            metadata={"mutation_strategy": strategy.value},
        )

    def mutate_many(
        self,
        prompt: Prompt | str,
        n: int,
        allow_duplicate_strategies: bool = False,
        parallel: bool = False,
        max_workers: int = 4,
    ) -> list[Prompt]:
        """
        Generate multiple mutations of the given prompt.

        Args:
            prompt: The prompt to mutate
            n: Number of mutations to generate
            allow_duplicate_strategies: Whether to use the same strategy multiple times
            parallel: Whether to use parallel execution for mutations
            max_workers: Maximum number of parallel workers (only used if parallel=True)

        Returns:
            A list of mutated Prompt objects
        """
        if isinstance(prompt, str):
            prompt = Prompt(content=prompt)

        mutations: list[Prompt] = []

        # Prepare strategies to use
        if allow_duplicate_strategies:
            strategies_to_use = [None] * n  # None means random selection
        else:
            # Use each strategy at most once, then cycle
            available_strategies = self.strategies.copy()
            strategies_to_use = []
            for i in range(n):
                if not available_strategies:
                    available_strategies = self.strategies.copy()
                strategy = available_strategies.pop(random.randint(0, len(available_strategies) - 1))
                strategies_to_use.append(strategy)

        if parallel and len(strategies_to_use) > 1:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self.mutate, prompt, strategy): i
                    for i, strategy in enumerate(strategies_to_use)
                }
                results = [None] * len(strategies_to_use)
                for future in as_completed(futures):
                    idx = futures[future]
                    results[idx] = future.result()
                mutations = [r for r in results if r is not None]
        else:
            # Sequential execution
            for strategy in strategies_to_use:
                mutations.append(self.mutate(prompt, strategy))

        return mutations
