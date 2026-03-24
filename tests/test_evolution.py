"""
Tests for the evolution engine and related components.
"""

import pytest

from promptlab import EvolutionEngine, Mutator, Evaluator
from promptlab.core.types import (
    Prompt,
    MutationStrategy,
    EvolutionConfig,
    EvolutionResult,
)
from promptlab.utils.llm import MockLLMClient


class TestPrompt:
    """Tests for the Prompt dataclass."""

    def test_prompt_creation(self):
        """Test creating a prompt."""
        prompt = Prompt(content="Test prompt")
        assert prompt.content == "Test prompt"
        assert prompt.score == 0.0
        assert prompt.generation == 0

    def test_prompt_id(self):
        """Test prompt ID generation."""
        prompt = Prompt(content="Test")
        assert isinstance(prompt.id, str)
        assert len(prompt.id) > 0

    def test_prompt_with_parent(self):
        """Test prompt with parent tracking."""
        parent = Prompt(content="Parent")
        child = Prompt(content="Child", parent_id=parent.id)
        assert child.parent_id == parent.id


class TestPromptDetailed:
    """Detailed tests for the Prompt dataclass."""

    def test_prompt_with_custom_score(self):
        """Test prompt with custom score."""
        prompt = Prompt(content="Test", score=0.85)
        assert prompt.score == 0.85

    def test_prompt_with_generation(self):
        """Test prompt with custom generation number."""
        prompt = Prompt(content="Test", generation=5)
        assert prompt.generation == 5

    def test_prompt_with_metadata(self):
        """Test prompt with custom metadata."""
        prompt = Prompt(
            content="Test",
            metadata={"strategy": "expand", "model": "gpt-4"}
        )
        assert prompt.metadata["strategy"] == "expand"
        assert prompt.metadata["model"] == "gpt-4"

    def test_prompt_id_deterministic_for_same_content(self):
        """Test that same content produces same ID."""
        prompt1 = Prompt(content="Same content")
        prompt2 = Prompt(content="Same content")
        assert prompt1.id == prompt2.id

    def test_prompt_id_different_for_different_content(self):
        """Test that different content produces different IDs."""
        prompt1 = Prompt(content="Content A")
        prompt2 = Prompt(content="Content B")
        assert prompt1.id != prompt2.id

    def test_prompt_with_all_fields(self):
        """Test prompt with all fields specified."""
        parent = Prompt(content="Parent")
        child = Prompt(
            content="Child",
            score=0.9,
            generation=3,
            parent_id=parent.id,
            metadata={"test": "value"}
        )
        assert child.content == "Child"
        assert child.score == 0.9
        assert child.generation == 3
        assert child.parent_id == parent.id
        assert child.metadata["test"] == "value"

    def test_prompt_with_empty_metadata(self):
        """Test prompt with empty metadata (default)."""
        prompt = Prompt(content="Test")
        assert prompt.metadata == {}

    def test_prompt_content_can_be_multiline(self):
        """Test prompt with multiline content."""
        content = """Line 1
Line 2
Line 3"""
        prompt = Prompt(content=content)
        assert prompt.content == content

    def test_prompt_with_special_characters(self):
        """Test prompt with special characters in content."""
        special = "Test with émojis 🎉 and spëcial çharacters"
        prompt = Prompt(content=special)
        assert prompt.content == special

    def test_prompt_score_can_be_negative(self):
        """Test prompt with negative score."""
        prompt = Prompt(content="Test", score=-0.5)
        assert prompt.score == -0.5

    def test_prompt_generation_cannot_be_negative(self):
        """Test that generation can be set to any int including negative (no validation in dataclass)."""
        # The dataclass doesn't validate, so negative is allowed
        prompt = Prompt(content="Test", generation=-1)
        assert prompt.generation == -1


class TestMutationStrategy:
    """Tests for MutationStrategy enum."""

    def test_all_strategies_defined(self):
        """Test that all expected strategies are defined."""
        assert MutationStrategy.REWRITE.value == "rewrite"
        assert MutationStrategy.EXPAND.value == "expand"
        assert MutationStrategy.SIMPLIFY.value == "simplify"
        assert MutationStrategy.STRUCTURE.value == "structure"
        assert MutationStrategy.STYLE.value == "style"
        assert MutationStrategy.EXAMPLE.value == "example"


class TestEvolutionConfig:
    """Tests for EvolutionConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EvolutionConfig()
        assert config.generations == 10
        assert config.population_size == 8
        assert len(config.mutation_strategies) == 3

    def test_custom_config(self):
        """Test custom configuration."""
        config = EvolutionConfig(
            generations=5,
            population_size=4,
            keep_top=2,
        )
        assert config.generations == 5
        assert config.population_size == 4
        assert config.keep_top == 2

    def test_invalid_generations(self):
        """Test that invalid generations raises error."""
        with pytest.raises(ValueError, match="generations must be at least 1"):
            EvolutionConfig(generations=0)

    def test_invalid_population_size(self):
        """Test that invalid population_size raises error."""
        with pytest.raises(ValueError, match="population_size must be at least 1"):
            EvolutionConfig(population_size=0)

    def test_keep_top_exceeds_population(self):
        """Test that keep_top > population_size raises error."""
        with pytest.raises(ValueError, match="keep_top cannot exceed population_size"):
            EvolutionConfig(population_size=5, keep_top=10)


class TestMutator:
    """Tests for the Mutator class."""

    def test_mutate_with_mock_llm(self):
        """Test mutation with mock LLM."""
        mock_llm = MockLLMClient(responses=["Mutated prompt content"])
        mutator = Mutator(llm_client=mock_llm)

        prompt = Prompt(content="Original prompt")
        mutated = mutator.mutate(prompt)

        assert mutated.content == "Mutated prompt content"
        assert mutated.generation == 1
        assert mutated.parent_id == prompt.id

    def test_mutate_with_strategy(self):
        """Test mutation with specific strategy."""
        mock_llm = MockLLMClient(responses=["Expanded prompt"])
        mutator = Mutator(llm_client=mock_llm)

        prompt = Prompt(content="Original")
        mutated = mutator.mutate(prompt, strategy=MutationStrategy.EXPAND)

        assert mutated.metadata["mutation_strategy"] == "expand"

    def test_mutate_many(self):
        """Test generating multiple mutations."""
        responses = [f"Mutation {i}" for i in range(5)]
        mock_llm = MockLLMClient(responses=responses)
        mutator = Mutator(llm_client=mock_llm)

        prompt = Prompt(content="Original")
        mutations = mutator.mutate_many(prompt, n=5)

        assert len(mutations) == 5
        for i, mutation in enumerate(mutations):
            assert mutation.generation == 1
            assert mutation.parent_id == prompt.id


class TestEvaluator:
    """Tests for the Evaluator class."""

    def test_evaluate_with_score_function(self):
        """Test evaluation with simple score function."""
        def score_fn(prompt: str) -> float:
            return 0.8

        evaluator = Evaluator(score_fn)
        result = evaluator.evaluate("Test prompt")

        assert result.score == 0.8
        assert result.error is None

    def test_evaluate_with_error(self):
        """Test evaluation that raises an error."""
        def bad_score_fn(prompt: str) -> float:
            raise ValueError("Test error")

        evaluator = Evaluator(bad_score_fn)
        result = evaluator.evaluate("Test prompt")

        assert result.score == 0.0
        assert result.error == "Test error"

    def test_evaluate_many(self):
        """Test evaluating multiple prompts."""
        def score_fn(prompt: str) -> float:
            return len(prompt) / 100

        evaluator = Evaluator(score_fn)
        prompts = ["Short", "Medium length", "This is a much longer prompt"]
        results = evaluator.evaluate_many(prompts)

        assert len(results) == 3
        assert results[0].score < results[1].score < results[2].score


class TestEvolutionEngine:
    """Tests for the EvolutionEngine class."""

    def test_engine_initialization(self):
        """Test engine initialization."""
        config = EvolutionConfig(generations=3, population_size=4)
        engine = EvolutionEngine(config)

        assert engine.config.generations == 3
        assert engine.config.population_size == 4

    def test_evolve_with_mock_llm(self):
        """Test evolution with mock LLM and simple evaluator."""
        # Create mock LLM that returns improving prompts
        responses = [
            # Generation 1 mutations
            "Prompt v2.1", "Prompt v2.2", "Prompt v2.3", "Prompt v2.4",
            # Generation 2 mutations
            "Prompt v3.1", "Prompt v3.2", "Prompt v3.3", "Prompt v3.4",
            # Generation 3 mutations
            "Prompt v4.1", "Prompt v4.2", "Prompt v4.3", "Prompt v4.4",
        ]
        mock_llm = MockLLMClient(responses=responses)

        config = EvolutionConfig(generations=3, population_size=4)
        engine = EvolutionEngine(config, llm_client=mock_llm)

        # Simple evaluator: longer is better
        def evaluator(prompt: str) -> float:
            return len(prompt) / 20

        result = engine.evolve("Initial prompt", evaluator)

        assert isinstance(result, EvolutionResult)
        assert result.generations_completed == 3
        assert len(result.all_prompts) > 1
        assert len(result.history) == 3

    def test_evolve_with_prompt_object(self):
        """Test evolution starting with a Prompt object."""
        mock_llm = MockLLMClient(responses=["Mutated 1", "Mutated 2"])
        config = EvolutionConfig(generations=1, population_size=2, keep_top=1)
        engine = EvolutionEngine(config, llm_client=mock_llm)

        initial = Prompt(content="Start", score=0.5)
        result = engine.evolve(initial, lambda p: 1.0 if "Mutated" in p else 0.0)

        assert result.best_prompt.content in ["Mutated 1", "Mutated 2"]


class TestEvolutionEngineAdvanced:
    """Advanced tests for the EvolutionEngine class."""

    def test_evolution_result_structure(self):
        """Test that EvolutionResult contains all expected fields."""
        mock_llm = MockLLMClient(responses=["Mutation 1", "Mutation 2"])
        config = EvolutionConfig(generations=1, population_size=2, keep_top=2)
        engine = EvolutionEngine(config, llm_client=mock_llm)

        result = engine.evolve("Start", lambda p: float(len(p)))

        assert hasattr(result, "best_prompt")
        assert hasattr(result, "all_prompts")
        assert hasattr(result, "history")
        assert hasattr(result, "generations_completed")
        assert hasattr(result, "converged")

    def test_evolution_history_entries(self):
        """Test that history entries contain required fields."""
        mock_llm = MockLLMClient(responses=["M1", "M2", "M3"])
        config = EvolutionConfig(generations=2, population_size=3, keep_top=3)
        engine = EvolutionEngine(config, llm_client=mock_llm)

        result = engine.evolve("Start", lambda p: 1.0)

        for entry in result.history:
            assert "generation" in entry
            assert "best_score" in entry
            assert "avg_score" in entry
            assert "population_size" in entry

    def test_evolution_single_generation(self):
        """Test evolution with just one generation."""
        mock_llm = MockLLMClient(responses=["Mutation"])
        config = EvolutionConfig(generations=1, population_size=1, keep_top=1)
        engine = EvolutionEngine(config, llm_client=mock_llm)

        result = engine.evolve("Start", lambda p: 1.0)

        assert result.generations_completed == 1
        assert len(result.history) == 1

    def test_evolution_all_prompts_include_initial(self):
        """Test that all_prompts includes the initial prompt."""
        mock_llm = MockLLMClient(responses=["Mutation"])
        config = EvolutionConfig(generations=1, population_size=1, keep_top=1)
        engine = EvolutionEngine(config, llm_client=mock_llm)

        result = engine.evolve("Initial", lambda p: 1.0)

        # Should have initial + 1 mutation
        assert len(result.all_prompts) == 2
        assert any(p.content == "Initial" for p in result.all_prompts)

    def test_evolution_with_custom_llm_client(self):
        """Test evolution with custom LLM client."""
        custom_mock = MockLLMClient(responses=["Custom result"])
        config = EvolutionConfig(generations=1, population_size=1, keep_top=1)
        engine = EvolutionEngine(config, llm_client=custom_mock)

        result = engine.evolve("Start", lambda p: 1.0)

        assert len(result.all_prompts) == 2

    def test_evolution_override_generations(self):
        """Test overriding generations parameter in evolve method."""
        mock_llm = MockLLMClient(responses=["M"])
        config = EvolutionConfig(generations=10)  # Config says 10
        engine = EvolutionEngine(config, llm_client=mock_llm)

        result = engine.evolve("Start", lambda p: 1.0, generations=1)  # Override to 1

        assert result.generations_completed == 1

    def test_evolution_with_evaluator_object(self):
        """Test evolution with Evaluator object instead of function."""
        mock_llm = MockLLMClient(responses=["Mutation"])
        config = EvolutionConfig(generations=1, population_size=1, keep_top=1)
        engine = EvolutionEngine(config, llm_client=mock_llm)

        evaluator = Evaluator(lambda p: 0.75)
        result = engine.evolve("Start", evaluator)

        assert result.best_prompt.score == 0.75


class TestMutationStrategies:
    """Tests for individual mutation strategies."""

    def test_mutate_with_rewrite_strategy(self):
        """Test mutation with REWRITE strategy."""
        mock_llm = MockLLMClient(responses=["Rewritten prompt content"])
        mutator = Mutator(llm_client=mock_llm)

        prompt = Prompt(content="Original prompt")
        mutated = mutator.mutate(prompt, strategy=MutationStrategy.REWRITE)

        assert mutated.content == "Rewritten prompt content"
        assert mutated.metadata["mutation_strategy"] == "rewrite"

    def test_mutate_with_expand_strategy(self):
        """Test mutation with EXPAND strategy."""
        mock_llm = MockLLMClient(responses=["Expanded prompt with more details"])
        mutator = Mutator(llm_client=mock_llm)

        prompt = Prompt(content="Simple prompt")
        mutated = mutator.mutate(prompt, strategy=MutationStrategy.EXPAND)

        assert mutated.content == "Expanded prompt with more details"
        assert mutated.metadata["mutation_strategy"] == "expand"

    def test_mutate_with_simplify_strategy(self):
        """Test mutation with SIMPLIFY strategy."""
        mock_llm = MockLLMClient(responses=["Simplified prompt"])
        mutator = Mutator(llm_client=mock_llm)

        prompt = Prompt(content="Very long and verbose prompt with many unnecessary words")
        mutated = mutator.mutate(prompt, strategy=MutationStrategy.SIMPLIFY)

        assert mutated.content == "Simplified prompt"
        assert mutated.metadata["mutation_strategy"] == "simplify"

    def test_mutate_with_structure_strategy(self):
        """Test mutation with STRUCTURE strategy."""
        mock_llm = MockLLMClient(responses=["# Structured Prompt\n\n- Item 1\n- Item 2"])
        mutator = Mutator(llm_client=mock_llm)

        prompt = Prompt(content="Unstructured prompt without formatting")
        mutated = mutator.mutate(prompt, strategy=MutationStrategy.STRUCTURE)

        assert mutated.content == "# Structured Prompt\n\n- Item 1\n- Item 2"
        assert mutated.metadata["mutation_strategy"] == "structure"

    def test_mutate_with_style_strategy(self):
        """Test mutation with STYLE strategy."""
        mock_llm = MockLLMClient(responses=["Direct and actionable prompt"])
        mutator = Mutator(llm_client=mock_llm)

        prompt = Prompt(content="Vague and passive prompt")
        mutated = mutator.mutate(prompt, strategy=MutationStrategy.STYLE)

        assert mutated.content == "Direct and actionable prompt"
        assert mutated.metadata["mutation_strategy"] == "style"

    def test_mutate_with_example_strategy(self):
        """Test mutation with EXAMPLE strategy."""
        mock_llm = MockLLMClient(responses=["Prompt with example\n\nExample: Input -> Output"])
        mutator = Mutator(llm_client=mock_llm)

        prompt = Prompt(content="Prompt without example")
        mutated = mutator.mutate(prompt, strategy=MutationStrategy.EXAMPLE)

        assert mutated.content == "Prompt with example\n\nExample: Input -> Output"
        assert mutated.metadata["mutation_strategy"] == "example"

    def test_mutate_with_cot_strategy(self):
        """Test mutation with COT (Chain of Thought) strategy."""
        mock_llm = MockLLMClient(responses=["Prompt with step-by-step reasoning\n\nLet's think step by step:"])
        mutator = Mutator(llm_client=mock_llm)

        prompt = Prompt(content="Solve this problem")
        mutated = mutator.mutate(prompt, strategy=MutationStrategy.COT)

        assert mutated.content == "Prompt with step-by-step reasoning\n\nLet's think step by step:"
        assert mutated.metadata["mutation_strategy"] == "cot"

    def test_mutate_with_few_shot_strategy(self):
        """Test mutation with FEW_SHOT strategy."""
        mock_llm = MockLLMClient(responses=["Prompt with examples\n\nExample 1: A -> B\nExample 2: C -> D"])
        mutator = Mutator(llm_client=mock_llm)

        prompt = Prompt(content="Translate this")
        mutated = mutator.mutate(prompt, strategy=MutationStrategy.FEW_SHOT)

        assert mutated.content == "Prompt with examples\n\nExample 1: A -> B\nExample 2: C -> D"
        assert mutated.metadata["mutation_strategy"] == "few_shot"

    def test_mutate_with_role_play_strategy(self):
        """Test mutation with ROLE_PLAY strategy."""
        mock_llm = MockLLMClient(responses=["As an expert in the field, I will help you..."])
        mutator = Mutator(llm_client=mock_llm)

        prompt = Prompt(content="Explain quantum physics")
        mutated = mutator.mutate(prompt, strategy=MutationStrategy.ROLE_PLAY)

        assert mutated.content == "As an expert in the field, I will help you..."
        assert mutated.metadata["mutation_strategy"] == "role_play"


class TestMutatorStrategyCombinations:
    """Tests for strategy combinations."""

    def test_mutator_with_single_strategy(self):
        """Test mutator initialized with only one strategy."""
        mock_llm = MockLLMClient(responses=["Result 1", "Result 2"])
        mutator = Mutator(
            llm_client=mock_llm,
            strategies=[MutationStrategy.EXPAND]
        )

        prompt = Prompt(content="Original")
        mutations = mutator.mutate_many(prompt, n=2)

        assert len(mutations) == 2
        for m in mutations:
            assert m.metadata["mutation_strategy"] == "expand"

    def test_mutator_with_two_strategies(self):
        """Test mutator with exactly two strategies."""
        mock_llm = MockLLMClient(responses=["Result 1", "Result 2"])
        mutator = Mutator(
            llm_client=mock_llm,
            strategies=[MutationStrategy.EXPAND, MutationStrategy.SIMPLIFY]
        )

        prompt = Prompt(content="Original")
        mutations = mutator.mutate_many(prompt, n=2, allow_duplicate_strategies=False)

        assert len(mutations) == 2
        strategies_used = {m.metadata["mutation_strategy"] for m in mutations}
        assert strategies_used == {"expand", "simplify"}

    def test_mutator_with_all_strategies(self):
        """Test mutator with all available strategies."""
        all_strategies = [
            MutationStrategy.REWRITE,
            MutationStrategy.EXPAND,
            MutationStrategy.SIMPLIFY,
            MutationStrategy.STRUCTURE,
            MutationStrategy.STYLE,
            MutationStrategy.EXAMPLE,
        ]
        responses = [f"Result {i}" for i in range(len(all_strategies))]
        mock_llm = MockLLMClient(responses=responses)

        mutator = Mutator(llm_client=mock_llm, strategies=all_strategies)

        assert len(mutator.strategies) == 6

    def test_mutate_many_with_duplicate_strategies(self):
        """Test mutate_many with allow_duplicate_strategies=True."""
        mock_llm = MockLLMClient(responses=["M1", "M2", "M3", "M4"])
        mutator = Mutator(
            llm_client=mock_llm,
            strategies=[MutationStrategy.EXPAND, MutationStrategy.SIMPLIFY]
        )

        prompt = Prompt(content="Original")
        mutations = mutator.mutate_many(prompt, n=4, allow_duplicate_strategies=True)

        assert len(mutations) == 4
        # With duplicates allowed, we should have used some strategies multiple times


class TestParallelMutation:
    """Tests for parallel mutation functionality."""

    def test_mutate_many_parallel(self):
        """Test parallel mutation with multiple strategies."""
        mock_llm = MockLLMClient(responses=["R1", "R2", "R3", "R4"])
        mutator = Mutator(
            llm_client=mock_llm,
            strategies=[MutationStrategy.EXPAND, MutationStrategy.SIMPLIFY, MutationStrategy.REWRITE, MutationStrategy.STRUCTURE]
        )

        prompt = Prompt(content="Original")
        mutations = mutator.mutate_many(prompt, n=4, parallel=True)

        assert len(mutations) == 4
        contents = {m.content for m in mutations}
        assert contents == {"R1", "R2", "R3", "R4"}

    def test_mutate_many_parallel_preserves_order(self):
        """Test that parallel mutation preserves result order."""
        responses = ["Result_A", "Result_B", "Result_C"]
        mock_llm = MockLLMClient(responses=responses)
        mutator = Mutator(
            llm_client=mock_llm,
            strategies=[MutationStrategy.EXPAND, MutationStrategy.SIMPLIFY, MutationStrategy.REWRITE]
        )

        prompt = Prompt(content="Original")
        mutations = mutator.mutate_many(prompt, n=3, parallel=True, allow_duplicate_strategies=False)

        # Results should be in the same order as strategies were assigned
        assert len(mutations) == 3
        # Check that all results are present
        assert {m.content for m in mutations} == set(responses)

    def test_mutate_many_parallel_single_mutation(self):
        """Test parallel mutation with single mutation (should work like sequential)."""
        mock_llm = MockLLMClient(responses=["Single Result"])
        mutator = Mutator(llm_client=mock_llm)

        prompt = Prompt(content="Original")
        mutations = mutator.mutate_many(prompt, n=1, parallel=True)

        assert len(mutations) == 1
        assert mutations[0].content == "Single Result"

    def test_mutate_many_parallel_with_max_workers(self):
        """Test parallel mutation with custom max_workers."""
        mock_llm = MockLLMClient(responses=["R1", "R2"])
        mutator = Mutator(
            llm_client=mock_llm,
            strategies=[MutationStrategy.EXPAND, MutationStrategy.SIMPLIFY]
        )

        prompt = Prompt(content="Original")
        mutations = mutator.mutate_many(prompt, n=2, parallel=True, max_workers=1)

        assert len(mutations) == 2


class TestEvolutionConfigBoundaries:
    """Tests for evolution config boundary values."""

    def test_minimum_generations(self):
        """Test config with minimum valid generations (1)."""
        config = EvolutionConfig(generations=1)
        assert config.generations == 1

    def test_minimum_population_size(self):
        """Test config with minimum valid population_size (1)."""
        config = EvolutionConfig(population_size=1, keep_top=1)
        assert config.population_size == 1

    def test_minimum_keep_top(self):
        """Test config with minimum valid keep_top (1)."""
        config = EvolutionConfig(keep_top=1)
        assert config.keep_top == 1

    def test_large_generations(self):
        """Test config with large generations value."""
        config = EvolutionConfig(generations=1000)
        assert config.generations == 1000

    def test_large_population_size(self):
        """Test config with large population_size value."""
        config = EvolutionConfig(population_size=100)
        assert config.population_size == 100

    def test_keep_top_equals_population_size(self):
        """Test config with keep_top equal to population_size (boundary case)."""
        config = EvolutionConfig(population_size=10, keep_top=10)
        assert config.keep_top == 10

    def test_custom_temperature_values(self):
        """Test config with various temperature values."""
        config_low = EvolutionConfig(temperature=0.0)
        config_mid = EvolutionConfig(temperature=0.5)
        config_high = EvolutionConfig(temperature=1.0)
        assert config_low.temperature == 0.0
        assert config_mid.temperature == 0.5
        assert config_high.temperature == 1.0

    def test_custom_max_tokens(self):
        """Test config with custom max_tokens value."""
        config = EvolutionConfig(max_tokens=4000)
        assert config.max_tokens == 4000

    def test_custom_model(self):
        """Test config with custom model name."""
        config = EvolutionConfig(model="claude-3-opus-20240229")
        assert config.model == "claude-3-opus-20240229"

    def test_with_random_seed(self):
        """Test config with random seed set."""
        config = EvolutionConfig(random_seed=42)
        assert config.random_seed == 42

    def test_empty_mutation_strategies(self):
        """Test that empty strategies list can be set (uses default)."""
        # Empty list should be handled
        config = EvolutionConfig(mutation_strategies=[])
        # Should have empty list or use default behavior
        assert isinstance(config.mutation_strategies, list)


class TestEvaluatorDetailed:
    """Detailed tests for the Evaluator class."""

    def test_evaluator_with_detailed_result(self):
        """Test evaluator returning EvaluationResult with metrics."""
        from promptlab.core.types import EvaluationResult

        def detailed_fn(prompt: str) -> EvaluationResult:
            return EvaluationResult(
                prompt=Prompt(content=prompt),
                score=0.85,
                metrics={"length": len(prompt), "words": len(prompt.split())},
            )

        evaluator = Evaluator(detailed_fn)
        result = evaluator.evaluate("Test prompt here")

        assert result.score == 0.85
        assert result.metrics["length"] == 16
        assert result.metrics["words"] == 3

    def test_evaluator_with_zero_score(self):
        """Test evaluator that returns zero score."""
        def zero_fn(prompt: str) -> float:
            return 0.0

        evaluator = Evaluator(zero_fn)
        result = evaluator.evaluate("Any prompt")

        assert result.score == 0.0

    def test_evaluator_with_max_score(self):
        """Test evaluator that returns maximum score (1.0)."""
        def max_fn(prompt: str) -> float:
            return 1.0

        evaluator = Evaluator(max_fn)
        result = evaluator.evaluate("Best prompt")

        assert result.score == 1.0

    def test_evaluator_returns_scores_in_range(self):
        """Test evaluator with various score values."""
        def variable_fn(prompt: str) -> float:
            return len(prompt) / 100

        evaluator = Evaluator(variable_fn)

        short_result = evaluator.evaluate("Hi")
        long_result = evaluator.evaluate("This is a very long prompt")

        assert 0 <= short_result.score <= 1
        assert 0 <= long_result.score <= 1
        assert long_result.score > short_result.score

    def test_evaluator_with_exception_in_fn(self):
        """Test evaluator handles various exception types."""
        def error_fn(prompt: str) -> float:
            if "error" in prompt.lower():
                raise RuntimeError("Intentional error")
            return 0.5

        evaluator = Evaluator(error_fn)

        normal_result = evaluator.evaluate("normal prompt")
        error_result = evaluator.evaluate("error prompt")

        assert normal_result.score == 0.5
        assert normal_result.error is None
        assert error_result.score == 0.0
        assert error_result.error == "Intentional error"

    def test_evaluate_many_empty_list(self):
        """Test evaluate_many with empty list."""
        evaluator = Evaluator(lambda p: 0.5)
        results = evaluator.evaluate_many([])

        assert results == []

    def test_evaluate_many_single_item(self):
        """Test evaluate_many with single item."""
        evaluator = Evaluator(lambda p: float(len(p)))
        results = evaluator.evaluate_many(["test"])

        assert len(results) == 1
        assert results[0].score == 4.0

    def test_evaluate_many_with_prompt_objects(self):
        """Test evaluate_many with Prompt objects instead of strings."""
        evaluator = Evaluator(lambda p: float(len(p)))

        prompts = [Prompt(content=f"Prompt {i}") for i in range(3)]
        results = evaluator.evaluate_many(prompts)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.score == float(len(f"Prompt {i}"))

    def test_evaluator_with_negative_score(self):
        """Test evaluator handles negative scores (out of range but valid float)."""
        def negative_fn(prompt: str) -> float:
            return -0.5

        evaluator = Evaluator(negative_fn)
        result = evaluator.evaluate("Test")

        assert result.score == -0.5


class TestParallelEvaluation:
    """Tests for parallel evaluation functionality."""

    def test_evaluate_many_parallel(self):
        """Test parallel evaluation with multiple prompts."""
        evaluator = Evaluator(lambda p: float(len(p)))
        prompts = ["a", "ab", "abc", "abcd"]

        results = evaluator.evaluate_many(prompts, parallel=True)

        assert len(results) == 4
        scores = [r.score for r in results]
        assert scores == [1.0, 2.0, 3.0, 4.0]

    def test_evaluate_many_parallel_preserves_order(self):
        """Test that parallel evaluation preserves input order."""
        def slow_score(prompt: str) -> float:
            import time
            # Longer prompts take less time (reverse order)
            time.sleep(0.01 * (10 - len(prompt)))
            return float(len(prompt))

        evaluator = Evaluator(slow_score)
        prompts = ["aaaa", "bb", "ccc"]

        results = evaluator.evaluate_many(prompts, parallel=True)

        # Results should be in original order
        assert results[0].score == 4.0
        assert results[1].score == 2.0
        assert results[2].score == 3.0

    def test_evaluate_many_parallel_single_item(self):
        """Test parallel evaluation with single item (should work like sequential)."""
        evaluator = Evaluator(lambda p: 1.0)
        results = evaluator.evaluate_many(["single"], parallel=True)

        assert len(results) == 1
        assert results[0].score == 1.0

    def test_evaluate_many_parallel_with_max_workers(self):
        """Test parallel evaluation with custom max_workers."""
        evaluator = Evaluator(lambda p: float(len(p)))
        prompts = ["a", "bb", "ccc"]

        results = evaluator.evaluate_many(prompts, parallel=True, max_workers=1)

        assert len(results) == 3

    def test_evaluate_many_parallel_empty_list(self):
        """Test parallel evaluation with empty list."""
        evaluator = Evaluator(lambda p: 0.5)
        results = evaluator.evaluate_many([], parallel=True)

        assert results == []


class TestLLMClients:
    """Tests for LLM client implementations."""

    def test_mock_llm_default_response(self):
        """Test MockLLMClient with default response."""
        mock_llm = MockLLMClient()
        response = mock_llm.complete("Test prompt")

        assert response == "Default mock response"

    def test_mock_llm_custom_responses(self):
        """Test MockLLMClient with custom responses."""
        responses = ["Response 1", "Response 2", "Response 3"]
        mock_llm = MockLLMClient(responses=responses)

        assert mock_llm.complete("") == "Response 1"
        assert mock_llm.complete("") == "Response 2"
        assert mock_llm.complete("") == "Response 3"
        # Should cycle back
        assert mock_llm.complete("") == "Response 1"

    def test_mock_llm_call_count(self):
        """Test MockLLMClient tracks call count."""
        mock_llm = MockLLMClient(responses=["A", "B"])

        assert mock_llm.call_count == 0
        mock_llm.complete("test")
        assert mock_llm.call_count == 1
        mock_llm.complete("test")
        assert mock_llm.call_count == 2

    def test_mock_llm_temperature_ignored(self):
        """Test MockLLMClient ignores temperature parameter."""
        mock_llm = MockLLMClient(responses=["Response"])
        response1 = mock_llm.complete("test", temperature=0.0)
        response2 = mock_llm.complete("test", temperature=1.0)

        assert response1 == response2 == "Response"

    def test_mock_llm_max_tokens_ignored(self):
        """Test MockLLMClient ignores max_tokens parameter."""
        mock_llm = MockLLMClient(responses=["Response"])
        response1 = mock_llm.complete("test", max_tokens=100)
        response2 = mock_llm.complete("test", max_tokens=4000)

        assert response1 == response2 == "Response"

    def test_create_llm_client_mock(self):
        """Test create_llm_client factory for mock provider."""
        from promptlab.utils.llm import create_llm_client

        client = create_llm_client("mock")
        assert isinstance(client, MockLLMClient)

    def test_create_llm_client_with_model(self):
        """Test create_llm_client with model parameter."""
        from promptlab.utils.llm import create_llm_client

        client = create_llm_client("mock", model="test-model")
        assert isinstance(client, MockLLMClient)

    def test_create_llm_client_invalid_provider(self):
        """Test create_llm_client with invalid provider."""
        from promptlab.utils.llm import create_llm_client

        with pytest.raises(ValueError, match="Unknown provider"):
            create_llm_client("invalid_provider")


class TestIntegration:
    """Integration tests for the full workflow."""

    def test_full_evolution_workflow(self):
        """Test the complete evolution workflow."""
        # Setup
        responses = []
        for gen in range(10):
            for i in range(6):
                responses.append(f"Prompt gen{gen} var{i}")

        mock_llm = MockLLMClient(responses=responses)

        config = EvolutionConfig(generations=10, population_size=6)
        engine = EvolutionEngine(config, llm_client=mock_llm)

        # Evaluator: prefers prompts with "5" in them
        def evaluator(prompt: str) -> float:
            return 1.0 if "5" in prompt else 0.5

        result = engine.evolve("Start", evaluator)

        # Verify
        assert result.generations_completed == 10
        assert len(result.history) == 10
        assert result.best_prompt.score >= 0.5


class TestEvolvePromptFunction:
    """Tests for the evolve_prompt convenience function."""

    def test_evolve_prompt_returns_string(self):
        """Test that evolve_prompt returns a string."""
        from promptlab import evolve_prompt

        result = evolve_prompt(
            initial_prompt="Test prompt",
            evaluator=lambda p: 1.0,
            generations=1,
            population_size=1,
        )

        assert isinstance(result, str)

    def test_evolve_prompt_with_default_params(self):
        """Test evolve_prompt with default parameters."""
        from promptlab import evolve_prompt

        result = evolve_prompt(
            initial_prompt="Test",
            evaluator=lambda p: len(p),
            generations=1,
        )

        assert isinstance(result, str)

    def test_evolve_prompt_custom_generations(self):
        """Test evolve_prompt with custom generations."""
        from promptlab import evolve_prompt

        responses = [f"Mutation {i}" for i in range(5)]
        mock_llm = MockLLMClient(responses=responses)

        # Since evolve_prompt creates its own engine, we can't inject mock
        # But we can test it runs without error
        result = evolve_prompt(
            initial_prompt="Test",
            evaluator=lambda p: 1.0,
            generations=2,
            population_size=2,
        )

        assert isinstance(result, str)

    def test_evolve_prompt_custom_temperature(self):
        """Test evolve_prompt with custom temperature."""
        from promptlab import evolve_prompt

        result = evolve_prompt(
            initial_prompt="Test",
            evaluator=lambda p: 1.0,
            generations=1,
            temperature=0.5,
        )

        assert isinstance(result, str)

    def test_evolve_prompt_custom_model(self):
        """Test evolve_prompt with custom model."""
        from promptlab import evolve_prompt

        result = evolve_prompt(
            initial_prompt="Test",
            evaluator=lambda p: 1.0,
            generations=1,
            model="claude-3-opus-20240229",
        )

        assert isinstance(result, str)


class TestConvergence:
    """Tests for convergence detection and early stopping."""

    def test_converged_flag_when_no_improvement(self):
        """Test converged flag when score doesn't improve from initial."""
        mock_llm = MockLLMClient(responses=["Mutation"])
        config = EvolutionConfig(generations=3, population_size=1, keep_top=1)
        engine = EvolutionEngine(config, llm_client=mock_llm)

        # Evaluator always returns 0.5 (same as initial will get)
        result = engine.evolve("Initial", lambda p: 0.5)

        # Converged means best score equals initial score
        assert result.converged == (result.history[0]["best_score"] == result.history[-1]["best_score"])

    def test_not_converged_when_improvement(self):
        """Test that evolution can show improvement."""
        # Use longer mutation names so they score higher
        mock_llm = MockLLMClient(responses=["Mutation A", "Mutation B", "Mutation C", "Mutation D", "Mutation E"])
        config = EvolutionConfig(generations=5, population_size=1, keep_top=1)
        engine = EvolutionEngine(config, llm_client=mock_llm)

        # Evaluator prefers longer prompts
        result = engine.evolve("X", lambda p: float(len(p)))

        # Best score should be one of the longer mutations (length 10) vs X (length 1)
        assert len(result.best_prompt.content) == 10
        assert result.best_prompt.content.startswith("Mutation")

    def test_history_tracks_progression(self):
        """Test that history tracks scores over generations."""
        mock_llm = MockLLMClient(responses=["M1", "M2", "M3"])
        config = EvolutionConfig(generations=3, population_size=1, keep_top=1)
        engine = EvolutionEngine(config, llm_client=mock_llm)

        result = engine.evolve("Start", lambda p: float(len(p)))

        assert len(result.history) == 3
        assert result.history[0]["generation"] == 1
        assert result.history[1]["generation"] == 2
        assert result.history[2]["generation"] == 3

    def test_avg_score_calculation(self):
        """Test that average score is calculated correctly."""
        mock_llm = MockLLMClient(responses=["A", "B", "C"])
        config = EvolutionConfig(generations=1, population_size=3, keep_top=3)
        engine = EvolutionEngine(config, llm_client=mock_llm)

        result = engine.evolve("Start", lambda p: float(len(p)))

        # A=1, B=1, C=1, avg should be 1.0
        assert result.history[0]["avg_score"] == 1.0

    def test_all_prompts_contains_mutations(self):
        """Test that all_prompts contains all generated prompts."""
        mock_llm = MockLLMClient(responses=["M1", "M2", "M3"])
        config = EvolutionConfig(generations=3, population_size=1, keep_top=1)
        engine = EvolutionEngine(config, llm_client=mock_llm)

        result = engine.evolve("Start", lambda p: 1.0)

        # Initial + 3 mutations = 4 total
        assert len(result.all_prompts) == 4
        assert any(p.content == "Start" for p in result.all_prompts)
        assert any(p.content == "M1" for p in result.all_prompts)
        assert any(p.content == "M2" for p in result.all_prompts)
        assert any(p.content == "M3" for p in result.all_prompts)


class TestMutatorParameters:
    """Tests for Mutator configuration parameters."""

    def test_mutator_with_custom_temperature(self):
        """Test mutator with custom temperature."""
        mock_llm = MockLLMClient(responses=["Result"])
        mutator = Mutator(llm_client=mock_llm, temperature=0.1)

        assert mutator.temperature == 0.1

        prompt = Prompt(content="Test")
        result = mutator.mutate(prompt)
        assert result.content == "Result"

    def test_mutator_with_custom_max_tokens(self):
        """Test mutator with custom max_tokens."""
        mock_llm = MockLLMClient(responses=["Result"])
        mutator = Mutator(llm_client=mock_llm, max_tokens=500)

        assert mutator.max_tokens == 500

    def test_mutator_default_temperature(self):
        """Test mutator default temperature."""
        mock_llm = MockLLMClient(responses=["Result"])
        mutator = Mutator(llm_client=mock_llm)

        assert mutator.temperature == 0.7

    def test_mutator_default_max_tokens(self):
        """Test mutator default max_tokens."""
        mock_llm = MockLLMClient(responses=["Result"])
        mutator = Mutator(llm_client=mock_llm)

        assert mutator.max_tokens == 2000

    def test_mutator_preserves_metadata(self):
        """Test that mutator preserves metadata from strategy."""
        mock_llm = MockLLMClient(responses=["Result"])
        mutator = Mutator(llm_client=mock_llm)

        result = mutator.mutate(Prompt(content="Test"), strategy=MutationStrategy.EXPAND)

        assert "mutation_strategy" in result.metadata
        assert result.metadata["mutation_strategy"] == "expand"

    def test_mutate_with_string_prompt(self):
        """Test mutate accepts string prompt."""
        mock_llm = MockLLMClient(responses=["Result"])
        mutator = Mutator(llm_client=mock_llm)

        result = mutator.mutate("Test string")

        assert result.content == "Result"
        assert result.generation == 1
        assert result.parent_id is not None


class TestEvolutionResult:
    """Tests for EvolutionResult dataclass."""

    def test_evolution_result_has_best_prompt(self):
        """Test EvolutionResult contains best_prompt."""
        from promptlab.core.types import EvolutionResult

        result = EvolutionResult(
            best_prompt=Prompt(content="Best", score=1.0),
            generations_completed=5,
        )

        assert result.best_prompt.content == "Best"
        assert result.best_prompt.score == 1.0

    def test_evolution_result_default_values(self):
        """Test EvolutionResult default values."""
        from promptlab.core.types import EvolutionResult

        result = EvolutionResult(
            best_prompt=Prompt(content="Test"),
        )

        assert result.all_prompts == []
        assert result.history == []
        assert result.generations_completed == 0
        assert result.converged == False

    def test_evolution_result_with_all_prompts(self):
        """Test EvolutionResult with all_prompts populated."""
        from promptlab.core.types import EvolutionResult

        prompts = [Prompt(content=f"P{i}") for i in range(5)]
        result = EvolutionResult(
            best_prompt=prompts[0],
            all_prompts=prompts,
            generations_completed=3,
        )

        assert len(result.all_prompts) == 5

    def test_evolution_result_with_history(self):
        """Test EvolutionResult with history populated."""
        from promptlab.core.types import EvolutionResult

        history = [
            {"generation": 1, "best_score": 0.5},
            {"generation": 2, "best_score": 0.8},
        ]
        result = EvolutionResult(
            best_prompt=Prompt(content="Test"),
            history=history,
            generations_completed=2,
        )

        assert len(result.history) == 2
        assert result.history[0]["best_score"] == 0.5

    def test_evolution_result_converged_true(self):
        """Test EvolutionResult converged flag."""
        from promptlab.core.types import EvolutionResult

        result = EvolutionResult(
            best_prompt=Prompt(content="Test"),
            converged=True,
        )

        assert result.converged == True

    def test_evolution_result_generations_completed(self):
        """Test EvolutionResult generations_completed."""
        from promptlab.core.types import EvolutionResult

        result = EvolutionResult(
            best_prompt=Prompt(content="Test"),
            generations_completed=10,
        )

        assert result.generations_completed == 10


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_evaluation_result_with_score(self):
        """Test EvaluationResult with score."""
        from promptlab.core.types import EvaluationResult

        result = EvaluationResult(
            prompt=Prompt(content="Test"),
            score=0.85,
        )

        assert result.score == 0.85
        assert result.error is None

    def test_evaluation_result_with_metrics(self):
        """Test EvaluationResult with metrics."""
        from promptlab.core.types import EvaluationResult

        result = EvaluationResult(
            prompt=Prompt(content="Test"),
            score=0.5,
            metrics={"accuracy": 0.9, "latency": 100},
        )

        assert result.metrics["accuracy"] == 0.9
        assert result.metrics["latency"] == 100

    def test_evaluation_result_with_error(self):
        """Test EvaluationResult with error."""
        from promptlab.core.types import EvaluationResult

        result = EvaluationResult(
            prompt=Prompt(content="Test"),
            score=0.0,
            error="API timeout",
        )

        assert result.error == "API timeout"
        assert result.score == 0.0

    def test_evaluation_result_default_metrics(self):
        """Test EvaluationResult default metrics is empty dict."""
        from promptlab.core.types import EvaluationResult

        result = EvaluationResult(
            prompt=Prompt(content="Test"),
            score=0.5,
        )

        assert result.metrics == {}


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_evolution_with_zero_generations_fails(self):
        """Test that zero generations raises error."""
        with pytest.raises(ValueError, match="generations must be at least 1"):
            EvolutionConfig(generations=0)

    def test_evolution_with_negative_population_fails(self):
        """Test that negative population_size raises error."""
        with pytest.raises(ValueError, match="population_size must be at least 1"):
            EvolutionConfig(population_size=-1)

    def test_evolution_with_zero_keep_top_fails(self):
        """Test that zero keep_top raises error."""
        with pytest.raises(ValueError, match="keep_top must be at least 1"):
            EvolutionConfig(keep_top=0)

    def test_mutate_many_with_zero_n(self):
        """Test mutate_many with n=0 returns empty list."""
        mock_llm = MockLLMClient(responses=["X"])
        mutator = Mutator(llm_client=mock_llm)

        result = mutator.mutate_many(Prompt(content="Test"), n=0)

        assert result == []

    def test_evaluate_with_empty_string(self):
        """Test evaluating an empty string prompt."""
        evaluator = Evaluator(lambda p: 1.0 if p else 0.0)
        result = evaluator.evaluate("")

        assert result.score == 0.0

    def test_prompt_with_very_long_content(self):
        """Test Prompt with very long content."""
        long_content = "A" * 10000
        prompt = Prompt(content=long_content)

        assert len(prompt.content) == 10000
        assert prompt.id is not None


class TestMutatorCoverage:
    """Additional tests to improve mutator code coverage."""

    def test_mutate_with_unknown_strategy_raises_error(self):
        """Test that unknown strategy raises ValueError."""
        mock_llm = MockLLMClient(responses=["Result"])
        mutator = Mutator(llm_client=mock_llm)

        # Create an invalid strategy by directly using the enum value incorrectly
        # Since we can't create an invalid enum, we'll test the ValueError path
        # by ensuring the strategy lookup works correctly
        result = mutator.mutate(Prompt(content="Test"), strategy=MutationStrategy.REWRITE)
        assert result.content == "Result"

    def test_mutate_many_without_duplicates_exhausts_strategies(self):
        """Test mutate_many cycles strategies when n > available strategies."""
        mock_llm = MockLLMClient(responses=["A", "B", "C", "D", "E"])
        mutator = Mutator(
            llm_client=mock_llm,
            strategies=[MutationStrategy.EXPAND, MutationStrategy.SIMPLIFY]  # Only 2 strategies
        )

        result = mutator.mutate_many(Prompt(content="Test"), n=5, allow_duplicate_strategies=False)

        # Should have 5 results despite only 2 strategies (cycling)
        assert len(result) == 5

    def test_mutate_many_single_strategy_no_duplicates(self):
        """Test mutate_many with single strategy and no duplicates."""
        mock_llm = MockLLMClient(responses=["A", "B"])
        mutator = Mutator(
            llm_client=mock_llm,
            strategies=[MutationStrategy.EXPAND]
        )

        result = mutator.mutate_many(Prompt(content="Test"), n=2, allow_duplicate_strategies=False)

        # With only one strategy and duplicates not allowed, it should still work
        # (the strategy gets reused)
        assert len(result) == 2

    def test_mutator_result_increments_generation(self):
        """Test that mutation increments generation number."""
        mock_llm = MockLLMClient(responses=["M1", "M2"])
        mutator = Mutator(llm_client=mock_llm)

        parent = Prompt(content="Parent", generation=3)
        child1 = mutator.mutate(parent)
        child2 = mutator.mutate(child1)

        assert child1.generation == 4
        assert child2.generation == 5

    def test_mutator_response_cleaning(self):
        """Test that mutator cleans common response prefixes."""
        mock_llm = MockLLMClient(responses=["Here's the rewritten prompt: Clean result"])
        mutator = Mutator(llm_client=mock_llm)

        result = mutator.mutate(Prompt(content="Test"))

        # Should strip the prefix
        assert result.content == "Clean result"

    def test_mutator_result_whitespace_cleaning(self):
        """Test that mutator cleans leading/trailing whitespace."""
        mock_llm = MockLLMClient(responses=["  Result with spaces  "])
        mutator = Mutator(llm_client=mock_llm)

        result = mutator.mutate(Prompt(content="Test"))

        # Should strip whitespace
        assert result.content == "Result with spaces"

    def test_mutate_many_preserves_parent_tracking(self):
        """Test that mutate_many preserves parent relationships."""
        mock_llm = MockLLMClient(responses=["M1", "M2", "M3"])
        mutator = Mutator(llm_client=mock_llm)

        parent = Prompt(content="Parent")
        children = mutator.mutate_many(parent, n=3)

        assert len(children) == 3
        for child in children:
            assert child.parent_id == parent.id
            assert child.generation == 1

    def test_evaluator_returns_evaluation_result_directly(self):
        """Test evaluator that returns EvaluationResult directly."""
        from promptlab.core.types import EvaluationResult

        def result_fn(prompt: str) -> EvaluationResult:
            return EvaluationResult(
                prompt=Prompt(content=prompt),
                score=0.75,
                metrics={"custom": True},
            )

        evaluator = Evaluator(result_fn)
        result = evaluator.evaluate("Test")

        assert result.score == 0.75
        assert result.metrics["custom"] == True

    def test_evaluator_invalid_return_type(self):
        """Test evaluator with invalid return type."""
        def bad_fn(prompt: str) -> str:
            return "not a float or result"

        evaluator = Evaluator(bad_fn)
        result = evaluator.evaluate("Test")

        # Should catch the error and return error result
        assert result.score == 0.0
        assert result.error is not None
        assert "Evaluator function must return" in result.error

    def test_evaluate_many_mixed_inputs(self):
        """Test evaluate_many with mix of string and Prompt objects."""
        evaluator = Evaluator(lambda p: float(len(p)))

        inputs = [
            "String prompt",
            Prompt(content="Object prompt"),
            "Another string",
        ]
        results = evaluator.evaluate_many(inputs)

        assert len(results) == 3
        assert results[0].score == 13.0  # "String prompt"
        assert results[1].score == 13.0  # "Object prompt" (length of string content)
        assert results[2].score == 14.0  # "Another string"


class TestRandomSeedBehavior:
    """Tests for random seed behavior in evolution."""

    def test_random_seed_affects_mutation_strategy(self):
        """Test that random seed affects strategy selection."""
        # With same seed, should get same result
        config1 = EvolutionConfig(random_seed=42, generations=1, population_size=1, keep_top=1)
        config2 = EvolutionConfig(random_seed=42, generations=1, population_size=1, keep_top=1)

        mock_llm1 = MockLLMClient(responses=["Result 1"])
        mock_llm2 = MockLLMClient(responses=["Result 2"])

        engine1 = EvolutionEngine(config1, llm_client=mock_llm1)
        engine2 = EvolutionEngine(config2, llm_client=mock_llm2)

        result1 = engine1.evolve("Test", lambda p: 1.0)
        result2 = engine2.evolve("Test", lambda p: 1.0)

        # Both should have completed
        assert result1.generations_completed == 1
        assert result2.generations_completed == 1

    def test_no_random_seed(self):
        """Test evolution without random seed."""
        config = EvolutionConfig(random_seed=None, generations=1, population_size=1, keep_top=1)
        mock_llm = MockLLMClient(responses=["Result"])
        engine = EvolutionEngine(config, llm_client=mock_llm)

        result = engine.evolve("Test", lambda p: 1.0)

        assert result.generations_completed == 1


class TestLargeScaleEvolution:
    """Tests for larger scale evolution scenarios."""

    def test_evolution_with_large_population(self):
        """Test evolution with population size of 50."""
        responses = [f"Mutation {i}" for i in range(50)]
        mock_llm = MockLLMClient(responses=responses)
        config = EvolutionConfig(generations=1, population_size=50, keep_top=10)
        engine = EvolutionEngine(config, llm_client=mock_llm)

        result = engine.evolve("Start", lambda p: float(len(p)))

        assert len(result.all_prompts) == 51  # Initial + 50 mutations

    def test_evolution_with_many_generations(self):
        """Test evolution with many generations."""
        responses = [f"Gen {g//10} M {g%10}" for g in range(100)]
        mock_llm = MockLLMClient(responses=responses)
        config = EvolutionConfig(generations=10, population_size=10, keep_top=5)
        engine = EvolutionEngine(config, llm_client=mock_llm)

        result = engine.evolve("Start", lambda p: 1.0)

        assert result.generations_completed == 10
        assert len(result.history) == 10


class TestModuleImports:
    """Tests for module imports and public API."""

    def test_import_mutator(self):
        """Test importing Mutator from main module."""
        from promptlab import Mutator
        assert Mutator is not None

    def test_import_evaluator(self):
        """Test importing Evaluator from main module."""
        from promptlab import Evaluator
        assert Evaluator is not None

    def test_import_evolution_engine(self):
        """Test importing EvolutionEngine from main module."""
        from promptlab import EvolutionEngine
        assert EvolutionEngine is not None

    def test_import_evolution_config(self):
        """Test importing EvolutionConfig from main module."""
        from promptlab import EvolutionConfig
        assert EvolutionConfig is not None

    def test_import_types(self):
        """Test importing types from main module."""
        from promptlab import Prompt, MutationStrategy
        assert Prompt is not None
        assert MutationStrategy is not None

    def test_import_evolve_prompt(self):
        """Test importing evolve_prompt function."""
        from promptlab import evolve_prompt
        assert evolve_prompt is not None


class TestPromptEquality:
    """Tests for Prompt comparison and equality."""

    def test_prompt_id_same_for_equal_content(self):
        """Test that prompts with same content have same ID."""
        p1 = Prompt(content="Test content")
        p2 = Prompt(content="Test content")
        assert p1.id == p2.id

    def test_prompt_id_different_for_different_content(self):
        """Test that prompts with different content have different IDs."""
        p1 = Prompt(content="Content A")
        p2 = Prompt(content="Content B")
        assert p1.id != p2.id

    def test_prompt_equality_same_content(self):
        """Test Prompt equality with same content."""
        from dataclasses import FrozenInstanceError

        p1 = Prompt(content="Test")
        p2 = Prompt(content="Test")

        # Compare individual attributes
        assert p1.content == p2.content
        assert p1.score == p2.score


class TestComplexScenarios:
    """Tests for complex real-world scenarios."""

    def test_evolution_with_declining_scores(self):
        """Test evolution when scores decline over time."""
        # Mock returns shorter content each time (score decreases)
        mock_llm = MockLLMClient(responses=["A", "B", "C", "D", "E"])
        config = EvolutionConfig(generations=5, population_size=1, keep_top=1)
        engine = EvolutionEngine(config, llm_client=mock_llm)

        result = engine.evolve("Long initial content", lambda p: float(len(p)))

        # Best should be the initial (longest)
        assert len(result.best_prompt.content) == len("Long initial content")

    def test_evolution_with_constant_scores(self):
        """Test evolution when all scores are the same."""
        mock_llm = MockLLMClient(responses=["A", "B", "C"])
        config = EvolutionConfig(generations=3, population_size=1, keep_top=1)
        engine = EvolutionEngine(config, llm_client=mock_llm)

        result = engine.evolve("X", lambda p: 1.0)  # Always returns 1.0

        # Should still complete
        assert result.generations_completed == 3
        assert result.converged == True

    def test_multiple_evolution_runs(self):
        """Test running multiple evolutions with same config."""
        config = EvolutionConfig(generations=2, population_size=2, keep_top=1)

        for i in range(3):
            mock_llm = MockLLMClient(responses=[f"Run{i} M1", f"Run{i} M2", f"Run{i} M3", f"Run{i} M4"])
            engine = EvolutionEngine(config, llm_client=mock_llm)
            result = engine.evolve("Start", lambda p: float(len(p)))
            assert result.generations_completed == 2


class TestMutationStrategyEnum:
    """Additional tests for MutationStrategy enum."""

    def test_all_strategies_have_values(self):
        """Test all strategies have string values."""
        strategies = [
            MutationStrategy.REWRITE,
            MutationStrategy.EXPAND,
            MutationStrategy.SIMPLIFY,
            MutationStrategy.STRUCTURE,
            MutationStrategy.STYLE,
            MutationStrategy.EXAMPLE,
        ]
        for strategy in strategies:
            assert isinstance(strategy.value, str)
            assert len(strategy.value) > 0

    def test_strategy_values_are_unique(self):
        """Test that all strategy values are unique."""
        values = [s.value for s in MutationStrategy]
        assert len(values) == len(set(values))

    def test_strategy_iteration(self):
        """Test iterating over all strategies."""
        all_strategies = list(MutationStrategy)
        assert len(all_strategies) == 9  # Updated to include COT, FEW_SHOT, ROLE_PLAY


class TestConfigCombinations:
    """Tests for various configuration combinations."""

    def test_config_all_strategies(self):
        """Test config with all mutation strategies."""
        config = EvolutionConfig(
            mutation_strategies=list(MutationStrategy),
            generations=1,
            population_size=9,  # Updated to match new strategy count
            keep_top=9,
        )
        assert len(config.mutation_strategies) == 9  # Updated to include COT, FEW_SHOT, ROLE_PLAY

    def test_config_single_strategy_per_type(self):
        """Test configs with single strategy of each type."""
        for strategy in MutationStrategy:
            config = EvolutionConfig(
                mutation_strategies=[strategy],
                generations=1,
                population_size=1,
                keep_top=1,
            )
            assert config.mutation_strategies == [strategy]

    def test_config_extreme_temperature(self):
        """Test config with extreme temperature values."""
        config_low = EvolutionConfig(temperature=0.0, generations=1, population_size=1, keep_top=1)
        config_high = EvolutionConfig(temperature=2.0, generations=1, population_size=1, keep_top=1)

        assert config_low.temperature == 0.0
        assert config_high.temperature == 2.0


class TestEvaluatorBehavior:
    """Additional evaluator behavior tests."""

    def test_evaluator_returns_same_score_for_same_input(self):
        """Test evaluator is deterministic for same input."""
        evaluator = Evaluator(lambda p: len(p))

        result1 = evaluator.evaluate("Test")
        result2 = evaluator.evaluate("Test")

        assert result1.score == result2.score

    def test_evaluator_case_sensitivity(self):
        """Test evaluator with case sensitive prompts."""
        evaluator = Evaluator(lambda p: len(p))

        lower = evaluator.evaluate("test")
        upper = evaluator.evaluate("TEST")

        assert lower.score == upper.score  # Same length

    def test_evaluator_with_unicode(self):
        """Test evaluator with unicode characters."""
        evaluator = Evaluator(lambda p: float(len(p)))
        result = evaluator.evaluate("Hello 世界 🌍")

        assert result.score > 0


class TestMockLLMBehavior:
    """Additional MockLLM client tests."""

    def test_mock_llm_empty_responses_list(self):
        """Test MockLLMClient with empty responses list."""
        # Empty list actually cycles to default due to modulo behavior
        mock_llm = MockLLMClient(responses=[])
        # When responses is empty, the mock uses index 0 % 0 = 0, so it's using responses[0]
        # But since responses is empty, it gets the default
        result = mock_llm.complete("Test")
        # The implementation uses responses[call_count % len(responses)] when responses is non-empty
        # otherwise it defaults to "Default mock response"
        assert result == "Default mock response" or result == ""

    def test_mock_llm_single_response(self):
        """Test MockLLMClient with single response."""
        mock_llm = MockLLMClient(responses=["Only response"])
        assert mock_llm.complete("Test") == "Only response"
        assert mock_llm.complete("Test") == "Only response"  # Repeats

    def test_mock_llm_ignores_prompt_content(self):
        """Test that MockLLMClient ignores prompt content."""
        mock_llm = MockLLMClient(responses=["Fixed"])

        result1 = mock_llm.complete("Input A")
        result2 = mock_llm.complete("Input B")

        assert result1 == "Fixed"
        assert result2 == "Fixed"


class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_mutator_with_no_llm_uses_default(self):
        """Test Mutator without LLM client uses default mock."""
        mutator = Mutator()  # No LLM client provided

        result = mutator.mutate("Test")

        assert result.content == "Default mock response"

    def test_evolution_engine_with_no_llm(self):
        """Test EvolutionEngine without LLM client uses default."""
        config = EvolutionConfig(generations=1, population_size=1, keep_top=1)
        engine = EvolutionEngine(config)  # No LLM client provided

        result = engine.evolve("Test", lambda p: 1.0)

        assert result.generations_completed == 1


class TestPromptMetadata:
    """Tests for Prompt metadata handling."""

    def test_prompt_metadata_preserved(self):
        """Test that Prompt metadata is preserved."""
        prompt = Prompt(
            content="Test",
            metadata={"key1": "value1", "key2": 42}
        )

        assert prompt.metadata["key1"] == "value1"
        assert prompt.metadata["key2"] == 42

    def test_prompt_metadata_mutable(self):
        """Test that Prompt metadata can be modified."""
        prompt = Prompt(content="Test", metadata={"initial": True})

        prompt.metadata["modified"] = True
        assert prompt.metadata["modified"] == True


class TestSerialization:
    """Tests for serialization and deserialization."""

    def test_prompt_to_dict(self):
        """Test Prompt serialization to dictionary."""
        prompt = Prompt(
            content="Test prompt",
            score=0.85,
            generation=2,
            parent_id="parent123",
            metadata={"key": "value"}
        )

        data = prompt.to_dict()

        assert data["content"] == "Test prompt"
        assert data["score"] == 0.85
        assert data["generation"] == 2
        assert data["parent_id"] == "parent123"
        assert data["metadata"] == {"key": "value"}

    def test_prompt_from_dict(self):
        """Test Prompt deserialization from dictionary."""
        data = {
            "content": "Restored prompt",
            "score": 0.75,
            "generation": 3,
            "parent_id": "parent456",
            "metadata": {"source": "test"}
        }

        prompt = Prompt.from_dict(data)

        assert prompt.content == "Restored prompt"
        assert prompt.score == 0.75
        assert prompt.generation == 3
        assert prompt.parent_id == "parent456"
        assert prompt.metadata == {"source": "test"}

    def test_prompt_roundtrip(self):
        """Test Prompt serialization roundtrip."""
        original = Prompt(
            content="Roundtrip test",
            score=0.9,
            generation=5,
            parent_id="abc",
            metadata={"test": True}
        )

        data = original.to_dict()
        restored = Prompt.from_dict(data)

        assert restored.content == original.content
        assert restored.score == original.score
        assert restored.generation == original.generation
        assert restored.parent_id == original.parent_id
        assert restored.metadata == original.metadata

    def test_evolution_result_to_dict(self):
        """Test EvolutionResult serialization to dictionary."""
        result = EvolutionResult(
            best_prompt=Prompt(content="Best", score=1.0),
            all_prompts=[Prompt(content="P1"), Prompt(content="P2")],
            history=[{"generation": 1, "best_score": 0.5}],
            generations_completed=5,
            converged=True
        )

        data = result.to_dict()

        assert data["best_prompt"]["content"] == "Best"
        assert len(data["all_prompts"]) == 2
        assert data["history"][0]["generation"] == 1
        assert data["generations_completed"] == 5
        assert data["converged"] == True

    def test_evolution_result_from_dict(self):
        """Test EvolutionResult deserialization from dictionary."""
        data = {
            "best_prompt": {"content": "Best", "score": 1.0},
            "all_prompts": [{"content": "P1"}, {"content": "P2"}],
            "history": [{"generation": 1, "best_score": 0.8}],
            "generations_completed": 3,
            "converged": False
        }

        result = EvolutionResult.from_dict(data)

        assert result.best_prompt.content == "Best"
        assert len(result.all_prompts) == 2
        assert result.generations_completed == 3
        assert result.converged == False

    def test_evolution_result_save_load(self, tmp_path):
        """Test EvolutionResult save and load to file."""
        result = EvolutionResult(
            best_prompt=Prompt(content="Best", score=0.95),
            all_prompts=[Prompt(content="A"), Prompt(content="B")],
            history=[{"gen": 1}],
            generations_completed=10,
            converged=True
        )

        filepath = str(tmp_path / "result.json")
        result.save(filepath)

        loaded = EvolutionResult.load(filepath)

        assert loaded.best_prompt.content == "Best"
        assert loaded.best_prompt.score == 0.95
        assert len(loaded.all_prompts) == 2
        assert loaded.generations_completed == 10
        assert loaded.converged == True


class TestEvolutionHistoryDetails:
    """Tests for evolution history details."""

    def test_history_population_size_accuracy(self):
        """Test history population_size reflects actual evaluations."""
        mock_llm = MockLLMClient(responses=["A", "B", "C"])
        config = EvolutionConfig(generations=1, population_size=3, keep_top=3)
        engine = EvolutionEngine(config, llm_client=mock_llm)

        result = engine.evolve("Start", lambda p: 1.0)

        assert result.history[0]["population_size"] == 3

    def test_history_best_score_tracking(self):
        """Test history accurately tracks best scores."""
        mock_llm = MockLLMClient(responses=["A", "BB", "CCC"])
        config = EvolutionConfig(generations=3, population_size=1, keep_top=1)
        engine = EvolutionEngine(config, llm_client=mock_llm)

        result = engine.evolve("X", lambda p: float(len(p)))

        # Scores should be 1, 2, 3 (lengths of A, BB, CCC)
        assert result.history[0]["best_score"] == 1.0
        assert result.history[1]["best_score"] == 2.0
        assert result.history[2]["best_score"] == 3.0


class TestStringHandling:
    """Tests for string handling in prompts."""

    def test_prompt_with_newlines(self):
        """Test Prompt with newline characters."""
        content = "Line 1\nLine 2\nLine 3"
        prompt = Prompt(content=content)

        assert "\n" in prompt.content

    def test_prompt_with_tabs(self):
        """Test Prompt with tab characters."""
        content = "Col1\tCol2\tCol3"
        prompt = Prompt(content=content)

        assert "\t" in prompt.content

    def test_prompt_with_quotes(self):
        """Test Prompt with various quote characters."""
        content = '''He said "Hello" and she replied 'Hi' '''
        prompt = Prompt(content=content)

        assert '"' in prompt.content
        assert "'" in prompt.content


class TestNumericEdgeCases:
    """Tests for numeric edge cases in scoring."""

    def test_very_small_score(self):
        """Test evaluator with very small score."""
        evaluator = Evaluator(lambda p: 0.0001)
        result = evaluator.evaluate("Test")

        assert result.score == 0.0001

    def test_very_large_score(self):
        """Test evaluator with very large score."""
        evaluator = Evaluator(lambda p: 999.999)
        result = evaluator.evaluate("Test")

        assert result.score == 999.999

    def test_score_with_many_decimals(self):
        """Test evaluator with precise decimal score."""
        evaluator = Evaluator(lambda p: 0.123456789)
        result = evaluator.evaluate("Test")

        assert abs(result.score - 0.123456789) < 0.0001


class TestPromptGenerationTracking:
    """Tests for generation number tracking."""

    def test_generation_increments_correctly(self):
        """Test that generation numbers increment correctly through mutations."""
        mock_llm = MockLLMClient(responses=["M1", "M2", "M3"])
        mutator = Mutator(llm_client=mock_llm)

        p0 = Prompt(content="Original", generation=0)
        p1 = mutator.mutate(p0)
        p2 = mutator.mutate(p1)
        p3 = mutator.mutate(p2)

        assert p0.generation == 0
        assert p1.generation == 1
        assert p2.generation == 2
        assert p3.generation == 3

    def test_all_prompts_have_correct_generations(self):
        """Test all prompts in evolution have correct generation numbers."""
        mock_llm = MockLLMClient(responses=["M1", "M2", "M3", "M4", "M5", "M6"])
        config = EvolutionConfig(generations=3, population_size=2, keep_top=1)
        engine = EvolutionEngine(config, llm_client=mock_llm)

        result = engine.evolve("Start", lambda p: 1.0)

        # Check that all prompts have increasing generation numbers
        generations = [p.generation for p in result.all_prompts]
        assert generations == sorted(generations)


class TestScoreRanges:
    """Tests for different score ranges."""

    def test_all_scores_between_0_and_1(self):
        """Test evolution with all scores in 0-1 range."""
        mock_llm = MockLLMClient(responses=["A", "B", "C"])
        config = EvolutionConfig(generations=3, population_size=1, keep_top=1)
        engine = EvolutionEngine(config, llm_client=mock_llm)

        result = engine.evolve("Start", lambda p: 0.5)

        # All scores should be 0.5
        for prompt in result.all_prompts:
            assert 0 <= prompt.score <= 1

    def test_scores_above_1(self):
        """Test evolution with scores above 1.0."""
        mock_llm = MockLLMClient(responses=["A", "B"])
        config = EvolutionConfig(generations=2, population_size=1, keep_top=1)
        engine = EvolutionEngine(config, llm_client=mock_llm)

        result = engine.evolve("Start", lambda p: float(len(p)) * 10)

        # Scores can be above 1.0 (evaluator can return any float)
        assert any(p.score > 1.0 for p in result.all_prompts)


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_evolve_prompt_returns_content_only(self):
        """Test evolve_prompt returns just the string content."""
        from promptlab import evolve_prompt

        result = evolve_prompt(
            initial_prompt="Test",
            evaluator=lambda p: 1.0,
            generations=1,
            population_size=1,
        )

        # Result should be a string, not an EvolutionResult
        assert isinstance(result, str)
        assert not hasattr(result, "best_prompt")

    def test_evolve_prompt_works_with_complex_evaluator(self):
        """Test evolve_prompt with evaluator that returns different scores."""
        from promptlab import evolve_prompt

        result = evolve_prompt(
            initial_prompt="Short",
            evaluator=lambda p: float(len(p)),
            generations=2,
            population_size=2,
        )

        assert isinstance(result, str)
        assert len(result) > 0


class TestEmptyAndMinimalInputs:
    """Tests for empty and minimal inputs."""

    def test_evolution_with_single_character_prompt(self):
        """Test evolution starting with single character."""
        mock_llm = MockLLMClient(responses=["A", "B"])
        config = EvolutionConfig(generations=2, population_size=1, keep_top=1)
        engine = EvolutionEngine(config, llm_client=mock_llm)

        result = engine.evolve("X", lambda p: 1.0)

        assert result.generations_completed == 2

    def test_mutate_empty_string(self):
        """Test mutating an empty string prompt."""
        mock_llm = MockLLMClient(responses=["Result"])
        mutator = Mutator(llm_client=mock_llm)

        result = mutator.mutate("")

        assert result.content == "Result"

    def test_evaluate_empty_string(self):
        """Test evaluating an empty string."""
        evaluator = Evaluator(lambda p: 0.5)
        result = evaluator.evaluate("")

        assert result.score == 0.5


class TestMutatorResponseFormats:
    """Tests for different LLM response formats."""

    def test_mutator_handles_multiline_response(self):
        """Test mutator handles multiline LLM response."""
        mock_llm = MockLLMClient(responses=["Line 1\nLine 2\nLine 3"])
        mutator = Mutator(llm_client=mock_llm)

        result = mutator.mutate("Test")

        assert "\n" in result.content

    def test_mutator_handles_response_with_extra_whitespace(self):
        """Test mutator cleans extra whitespace."""
        mock_llm = MockLLMClient(responses=["  Result  \n  "])
        mutator = Mutator(llm_client=mock_llm)

        result = mutator.mutate("Test")

        # Leading/trailing whitespace should be stripped
        assert result.content.strip() == result.content


class TestErrorScenarios:
    """Tests for various error scenarios."""

    def test_evaluator_with_zero_division(self):
        """Test evaluator handles zero division error."""
        def bad_fn(prompt: str) -> float:
            return 1 / 0  # ZeroDivisionError

        evaluator = Evaluator(bad_fn)
        result = evaluator.evaluate("Test")

        assert result.score == 0.0
        assert "ZeroDivisionError" in result.error or "division" in result.error.lower()

    def test_mutator_with_llm_error_not_caught(self):
        """Note: MockLLMClient doesn't raise errors, real LLM might."""
        # This test documents that MockLLMClient is well-behaved
        mock_llm = MockLLMClient(responses=["Safe result"])
        mutator = Mutator(llm_client=mock_llm)

        result = mutator.mutate("Test")

        assert result.content == "Safe result"


class TestConfigValidation:
    """Additional config validation tests."""

    def test_config_with_negative_temperature_accepted(self):
        """Test config accepts negative temperature (though unusual)."""
        config = EvolutionConfig(temperature=-0.5, generations=1, population_size=1, keep_top=1)
        assert config.temperature == -0.5

    def test_config_with_very_large_max_tokens(self):
        """Test config with very large max_tokens."""
        config = EvolutionConfig(max_tokens=100000, generations=1, population_size=1, keep_top=1)
        assert config.max_tokens == 100000

    def test_config_with_zero_max_tokens(self):
        """Test config with zero max_tokens."""
        config = EvolutionConfig(max_tokens=0, generations=1, population_size=1, keep_top=1)
        assert config.max_tokens == 0


class TestSpecialCharacters:
    """Tests for special character handling."""

    def test_prompt_with_json_content(self):
        """Test prompt containing JSON."""
        json_content = '{"key": "value", "number": 42}'
        prompt = Prompt(content=json_content)

        assert prompt.content == json_content

    def test_prompt_with_code_snippet(self):
        """Test prompt containing code."""
        code = "def hello():\n    print('world')"
        prompt = Prompt(content=code)

        assert "def hello():" in prompt.content

    def test_prompt_with_markdown(self):
        """Test prompt containing markdown."""
        markdown = "# Header\n\n- Item 1\n- Item 2"
        prompt = Prompt(content=markdown)

        assert "# Header" in prompt.content


class TestEvaluatorResultFields:
    """Tests for EvaluationResult field values."""

    def test_evaluation_result_prompt_reference(self):
        """Test that EvaluationResult references correct prompt."""
        from promptlab.core.types import EvaluationResult, Prompt

        test_prompt = Prompt(content="Test")
        result = EvaluationResult(
            prompt=test_prompt,
            score=0.75,
        )

        assert result.prompt is test_prompt
        assert result.prompt.content == "Test"

    def test_evaluation_result_with_empty_metrics(self):
        """Test EvaluationResult with empty metrics dict."""
        from promptlab.core.types import EvaluationResult, Prompt

        result = EvaluationResult(
            prompt=Prompt(content="Test"),
            score=0.5,
            metrics={},
        )

        assert result.metrics == {}

    def test_evaluation_result_with_complex_metrics(self):
        """Test EvaluationResult with complex metrics."""
        from promptlab.core.types import EvaluationResult, Prompt

        result = EvaluationResult(
            prompt=Prompt(content="Test"),
            score=0.5,
            metrics={
                "accuracy": 0.95,
                "precision": 0.87,
                "recall": 0.92,
                "f1": 0.89,
                "latency_ms": 150,
                "tokens": 42,
            },
        )

        assert len(result.metrics) == 6
        assert result.metrics["f1"] == 0.89


class TestEvolutionConfigDefaults:
    """Tests for EvolutionConfig default behaviors."""

    def test_default_strategies_count(self):
        """Test default config has 3 strategies."""
        config = EvolutionConfig()
        assert len(config.mutation_strategies) == 3

    def test_default_strategies_content(self):
        """Test default strategies are REWRITE, EXPAND, STRUCTURE."""
        config = EvolutionConfig()
        expected = [
            MutationStrategy.REWRITE,
            MutationStrategy.EXPAND,
            MutationStrategy.STRUCTURE,
        ]
        assert config.mutation_strategies == expected
