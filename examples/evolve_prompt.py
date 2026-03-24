"""
Example: Evolving a prompt for code generation.

This example demonstrates how to use PromptLab to optimize a prompt
for generating Python code.
"""

from promptlab import evolve_prompt, EvolutionEngine, EvolutionConfig
from promptlab.core.types import MutationStrategy


def code_quality_evaluator(prompt: str) -> float:
    """
    Evaluate a code generation prompt.

    This is a simple example evaluator. In practice, you might:
    - Actually run the prompt through an LLM
    - Execute the generated code
    - Test the code against test cases
    - Measure performance metrics

    For this example, we use keyword analysis.
    """
    prompt_lower = prompt.lower()

    # Check for important elements in code generation prompts
    criteria = {
        "specific_function": any(word in prompt_lower for word in ["function", "def", "method"]),
        "has_context": any(word in prompt_lower for word in ["context", "background", "given"]),
        "has_requirements": any(word in prompt_lower for word in ["should", "must", "requirement", "spec"]),
        "has_example": "example" in prompt_lower,
        "has_output_spec": any(word in prompt_lower for word in ["output", "return", "result"]),
        "is_detailed": len(prompt.split()) > 20,
    }

    return sum(criteria.values()) / len(criteria)


def main():
    """Run the evolution example."""
    print("=" * 60)
    print("PromptLab - Code Generation Prompt Evolution")
    print("=" * 60)
    print()

    # Initial prompt (quite basic)
    initial_prompt = "Write a Python function"

    print(f"Initial prompt: {initial_prompt}")
    print(f"Initial score: {code_quality_evaluator(initial_prompt):.2f}")
    print()

    # Method 1: Using the convenience function
    print("Method 1: Using evolve_prompt()")
    print("-" * 40)

    best_prompt = evolve_prompt(
        initial_prompt=initial_prompt,
        evaluator=code_quality_evaluator,
        generations=5,
        population_size=6,
    )

    print(f"Best prompt: {best_prompt}")
    print(f"Best score: {code_quality_evaluator(best_prompt):.2f}")
    print()

    # Method 2: Using EvolutionEngine for more control
    print("Method 2: Using EvolutionEngine")
    print("-" * 40)

    config = EvolutionConfig(
        generations=5,
        population_size=6,
        mutation_strategies=[
            MutationStrategy.EXPAND,
            MutationStrategy.STRUCTURE,
            MutationStrategy.EXAMPLE,
        ],
        temperature=0.8,
    )

    engine = EvolutionEngine(config)
    result = engine.evolve(initial_prompt, code_quality_evaluator)

    print(f"Best prompt: {result.best_prompt.content}")
    print(f"Best score: {result.best_prompt.score:.2f}")
    print()

    # Show evolution history
    print("Evolution History:")
    print("-" * 40)
    for entry in result.history:
        print(f"Gen {entry['generation']}: "
              f"Best={entry['best_score']:.2f}, "
              f"Avg={entry['avg_score']:.2f}")


if __name__ == "__main__":
    main()
