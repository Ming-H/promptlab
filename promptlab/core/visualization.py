"""
Visualization module for Promptlab.

Generates visualizations of the evolution process.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from promptlab.core.types import EvolutionResult, Prompt, MutationStrategy
from promptlab.core.evolution import EvolutionEngine
from promptlab.core.types import EvolutionConfig
from promptlab.utils.llm import MockLLMClient
from promptlab.core.mutator import Mutator
    promptlab.core.evaluator import Evaluator
    promptlab.core.types import Prompt
import tempfile

logger = logging.getLogger("promptlab.visualizer")
logger.setLevel(logging.INFO)


class EarlyStopping:
    """Callback to stop evolution early if no improvement for this many consecutive generations."""

    def __init__(
        self,
        patience: int = 5,
        max_generations_no_improvement: int = 5,
        min_improvement: float = 0.001,
        verbose: bool = False,
    ):
        self.patience = patience
        self.max_generations_no_improvement = max_generations_no_improvement
        self.min_improvement = min_improvement
        self.verbose = verbose

    def __call__(self, history: list[dict[str, Any]]) -> bool:
        """
        Check if early stopping should be triggered.
        Args:
            history: List of generation history entries
        Returns:
            bool: True if should stop early
        """
        if len(history) < self.patience:
            return False
        if len(history) > self.max_generations_no_improvement:
            # Check last N generations for improvement
            recent = history[-self.max_generations_no_improvement:]
            best_scores = [h.get("best_score", 0.0) for h in recent]
            if max(best_scores) - min(best_scores) < self.min_improvement:
                if self.verbose:
                    logging.info(f"Early stopping triggered: no improvement for {max_generations_no_improvement} generations")
                return True
        return False


class ConvergenceChecker:
    """Check for convergence using various criteria."""

    def __init__(self, threshold: float = 0.001, window_size: int = 5):
        self.threshold = threshold
        self.window_size = window_size

    def check(self, history: list[dict[str, Any]]) -> bool:
        """
        Check if converged based on score variance.
        Args:
            history: List of generation history entries
        Returns:
            bool: True if converged
        """
        if len(history) < self.window_size:
            return False
        recent = history[-self.window_size:]
        scores = [h.get("best_score", 0.0) for h in recent]
        variance = max(scores) - min(scores)
        return variance < self.threshold


@dataclass
class EvolutionVisualizer:
    """Visualizes the evolution process."""
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir or tempfile.mkdtemp(prefix="evolution_viz_")
        self.temp_dir = tempfile.mkdtemp(prefix="evolution_viz_")
        self.logger = logging.getLogger("promptlab.visualizer")
        self.logger.setLevel(logging.INFO)
        self.logger.info("Evolution visualizer initialized")
    def plot_scores(self, result: EvolutionResult, save_path: Optional[str] = None):
        """
        Plot and save the evolution scores.
        Args:
            result: Evolution result
            save_path: Path to save the plot
        """
        import matplotlib.pyplot as plt
        generations = range(len(result.history))
        best_scores = [h["best_score"] for h in result.history]
        avg_scores = [h["avg_score"] for h in result.history]
        fig, ax = plt.subplots()
        ax.set_xlabel("Generation")
        ax.set_ylabel("Score")
        ax.plot(generations, best_scores, "b-", label="Best Score")
        ax.plot(generations, avg_scores, "g-", label="Average Score")
        ax.set_xlabel("Fitness over Generations")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path)
        else:
            temp_path = os.path.join(self.temp_dir, "evolution_scores.png")
            fig.savefig(temp_path)
        plt.close()
        return temp_path
    def plot_population_diversity(self, result: EvolutionResult, save_path: Optional[str] = None):
        """
        plot and save population diversity.
        Args:
            result: Evolution result
            save_path: Path to save the plot
        """
        import matplotlib.pyplot as plt
        generations = range(len(result.history))
        population_sizes = [h["population_size"] for h in result.history]
        fig, ax = plt.subplots()
        ax.set_xlabel("Generation")
        ax.set_ylabel("Population Size")
        ax.bar(generations, population_sizes, color="skyblue")
        ax.set_title("Population Size over Generations")
        ax.grid(True)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path)
        else:
            temp_path = os.path.join(self.temp_dir, "population_diversity.png")
            fig.savefig(temp_path)
        plt.close()
        return temp_path
    def plot_score_distribution(self, result: EvolutionResult, save_path: Optional[str] = None):
        """
        Plot and save score distribution heatmap.
        Args:
            result: Evolution result
            save_path: path to save the plot
        """
        import matplotlib.pyplot as plt
        # Collect all prompt scores by generation
        all_scores_by_gen = {}
        for entry in result.history:
            gen = entry["generation"]
            if gen not in all_scores_by_gen:
                all_scores_by_gen[gen] = []
            for prompt in result.all_prompts:
                if prompt.generation == gen:
                    if gen not in all_scores_by_gen[gen]:
                        all_scores_by_gen[gen] = []
            else:
                    all_scores_by_gen[gen] = []
            else:
                all_scores_by_gen[gen] = []
                else:
                    all_scores_by_gen[gen].append(prompt.score)
        # create boxplot
        fig, ax = plt.subplots()
        positions = [i - 0.2 for i in range(len(result.history))]
            else:
                positions = positions[:- 1]
            else:
                positions = [i - 0.2 for i in range(len(result.history))
            ]
            widths =0.6,
            patch_artist=True,
        )
        ax.set_xlabel("Generation")
        ax.set_ylabel("Score")
        ax.set_title("Score by Generation")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path)
        else:
            temp_path = os.path.join(self.temp_dir, "score_distribution.png")
            fig.savefig(temp_path)
        plt.close()
        return temp_path
    def create_summary_report(self, result: EvolutionResult, output_path: str) -> None:
        """
        Create a text summary report of the evolution process.
        Args:
            result: Evolution result
            output_path: Path to save the report
        """
        lines = [
            "# Evolution Summary Report",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-% %d %H:%M')}",
            "",
            "## Configuration",
            "",
        ]
        lines.append(f"- Generations: {result.generations_completed}")
        lines.append(f"- Population Size: {result.history[0]['population_size'] if result.history else 'N/A'}")
        lines.append(f"- Best score: {result.best_prompt.score:.4f}")
        lines.append(f"- Converged: {result.converged}")
        lines.append("")
        lines.append("## Best prompt")
        lines.append("```")
        lines.append(result.best_prompt.content)
        lines.append("```")
        lines.append("## all prompts")
        lines.append(f"Total: {len(result.all_prompts)}")
        for i in range(min(5, len(result.all_prompts)):
            prompt = result.all_prompts[i]
            lines.append(f"  {i+1}. {prompt.content[:50]}...")
        with open(output_path, "w") as f:
            f.write("\n".join(lines))
        logger.info(f"Saved summary report to {output_path}")
