"""Visualization utilities for agent training and evaluation."""

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


class TrainingVisualizer:
    """Visualize agent training metrics."""
    
    def __init__(self, output_dir: Path | str = "data/plots"):
        """Initialize visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for metrics
        self.metrics: Dict[str, List] = {
            "episode": [],
            "win_rate": [],
            "lose_rate": [],
            "draw_rate": [],
            "avg_return": [],
            "epsilon": [],
            "loss": [],  # For deep learning agents
        }
    
    def add_metrics(
        self,
        episode: int,
        win_rate: float,
        lose_rate: float,
        draw_rate: float,
        avg_return: float,
        epsilon: float,
        loss: Optional[float] = None,
    ) -> None:
        """Add training metrics.
        
        Args:
            episode: Episode number
            win_rate: Win rate (0-1)
            lose_rate: Lose rate (0-1)
            draw_rate: Draw rate (0-1)
            avg_return: Average return per episode
            epsilon: Current epsilon value
            loss: Current loss (optional, for deep learning)
        """
        self.metrics["episode"].append(episode)
        self.metrics["win_rate"].append(win_rate)
        self.metrics["lose_rate"].append(lose_rate)
        self.metrics["draw_rate"].append(draw_rate)
        self.metrics["avg_return"].append(avg_return)
        self.metrics["epsilon"].append(epsilon)
        if loss is not None:
            self.metrics["loss"].append(loss)
    
    def plot_training_curves(self, agent_name: str, save: bool = True) -> None:
        """Plot training curves.
        
        Args:
            agent_name: Name of agent
            save: Whether to save plot
        """
        if len(self.metrics["episode"]) == 0:
            print("No metrics to plot!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Training Curves: {agent_name}", fontsize=16, fontweight="bold")
        
        episodes = self.metrics["episode"]
        
        # Win Rate
        ax = axes[0, 0]
        ax.plot(episodes, self.metrics["win_rate"], label="Win Rate", linewidth=2)
        ax.fill_between(episodes, self.metrics["win_rate"], alpha=0.3)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Win Rate")
        ax.set_title("Win Rate Evolution")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Average Return
        ax = axes[0, 1]
        ax.plot(episodes, self.metrics["avg_return"], label="Avg Return", linewidth=2, color="green")
        ax.fill_between(episodes, self.metrics["avg_return"], alpha=0.3, color="green")
        ax.axhline(y=0, color="r", linestyle="--", alpha=0.5)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Average Return")
        ax.set_title("Average Return Evolution")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Win/Lose/Draw Rates
        ax = axes[1, 0]
        ax.plot(episodes, self.metrics["win_rate"], label="Win", linewidth=2)
        ax.plot(episodes, self.metrics["lose_rate"], label="Lose", linewidth=2)
        ax.plot(episodes, self.metrics["draw_rate"], label="Draw", linewidth=2)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Rate")
        ax.set_title("Outcome Rates")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Epsilon & Loss
        ax = axes[1, 1]
        ax2 = ax.twinx()
        
        line1 = ax.plot(episodes, self.metrics["epsilon"], label="Epsilon", linewidth=2, color="orange")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Epsilon", color="orange")
        ax.tick_params(axis="y", labelcolor="orange")
        ax.grid(True, alpha=0.3)
        
        if self.metrics["loss"]:
            line2 = ax2.plot(episodes[: len(self.metrics["loss"])], self.metrics["loss"], 
                            label="Loss", linewidth=2, color="red")
            ax2.set_ylabel("Loss", color="red")
            ax2.tick_params(axis="y", labelcolor="red")
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc="upper right")
        else:
            ax.legend(loc="upper right")
        
        ax.set_title("Exploration & Loss")
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / f"{agent_name}_training_curves.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved plot to {save_path}")
        
        plt.close()
    
    def plot_eval_comparison(
        self,
        agent_name: str,
        train_metrics: Dict,
        eval_metrics: Dict,
        save: bool = True,
    ) -> None:
        """Plot training vs evaluation comparison.
        
        Args:
            agent_name: Name of agent
            train_metrics: Training metrics
            eval_metrics: Evaluation metrics
            save: Whether to save plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Train vs Eval: {agent_name}", fontsize=14, fontweight="bold")
        
        categories = ["Win Rate", "Avg Return"]
        
        # Win Rate comparison
        ax = axes[0]
        agents = ["Train", "Eval"]
        win_rates = [train_metrics.get("win_rate", 0), eval_metrics.get("win_rate", 0)]
        colors = ["#1f77b4", "#ff7f0e"]
        bars = ax.bar(agents, win_rates, color=colors, alpha=0.7, edgecolor="black", linewidth=2)
        ax.set_ylabel("Win Rate")
        ax.set_title("Win Rate Comparison")
        ax.set_ylim([0, 1])
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f"{height:.1%}", ha="center", va="bottom", fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        
        # Average Return comparison
        ax = axes[1]
        returns = [train_metrics.get("avg_return", 0), eval_metrics.get("avg_return", 0)]
        bars = ax.bar(agents, returns, color=colors, alpha=0.7, edgecolor="black", linewidth=2)
        ax.set_ylabel("Average Return")
        ax.set_title("Average Return Comparison")
        ax.axhline(y=0, color="r", linestyle="--", alpha=0.5)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f"{height:.3f}", ha="center", va="bottom", fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / f"{agent_name}_train_vs_eval.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved plot to {save_path}")
        
        plt.close()
    
    def plot_test_results(
        self,
        agent_name: str,
        test_results: Dict[str, List],
        save: bool = True,
    ) -> None:
        """Plot test results.
        
        Args:
            agent_name: Name of agent
            test_results: Dictionary with test metrics
            save: Whether to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Test Results: {agent_name}", fontsize=14, fontweight="bold")
        
        # Episodes
        episodes = list(range(1, len(test_results.get("returns", [])) + 1))
        
        # Returns distribution
        ax = axes[0, 0]
        returns = test_results.get("returns", [])
        ax.hist(returns, bins=20, edgecolor="black", alpha=0.7)
        ax.axvline(np.mean(returns), color="r", linestyle="--", linewidth=2, label=f"Mean: {np.mean(returns):.3f}")
        ax.set_xlabel("Return")
        ax.set_ylabel("Frequency")
        ax.set_title("Return Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        
        # Returns over episodes
        ax = axes[0, 1]
        ax.plot(episodes, returns, linewidth=1.5, alpha=0.7)
        ax.axhline(np.mean(returns), color="r", linestyle="--", linewidth=2, label="Mean")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Return")
        ax.set_title("Returns Over Episodes")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Outcome distribution
        ax = axes[1, 0]
        outcomes = test_results.get("outcomes", [])
        outcome_counts = {"Win": 0, "Lose": 0, "Draw": 0}
        for outcome in outcomes:
            outcome_counts[outcome] += 1
        
        colors = ["#2ecc71", "#e74c3c", "#95a5a6"]
        bars = ax.bar(outcome_counts.keys(), outcome_counts.values(), color=colors, alpha=0.7, edgecolor="black", linewidth=2)
        ax.set_ylabel("Count")
        ax.set_title("Outcome Distribution")
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f"{int(height)}", ha="center", va="bottom", fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        
        # Statistics summary
        ax = axes[1, 1]
        ax.axis("off")
        
        stats_text = f"""
        Test Statistics Summary
        ━━━━━━━━━━━━━━━━━━━━━━━
        Total Episodes: {len(episodes)}
        Mean Return: {np.mean(returns):.4f}
        Std Return: {np.std(returns):.4f}
        Min Return: {np.min(returns):.4f}
        Max Return: {np.max(returns):.4f}
        
        Outcome Distribution:
        ─ Win: {outcome_counts['Win']} ({outcome_counts['Win']/len(outcomes)*100:.1f}%)
        ─ Lose: {outcome_counts['Lose']} ({outcome_counts['Lose']/len(outcomes)*100:.1f}%)
        ─ Draw: {outcome_counts['Draw']} ({outcome_counts['Draw']/len(outcomes)*100:.1f}%)
        """
        
        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=11,
               verticalalignment="top", fontfamily="monospace",
               bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / f"{agent_name}_test_results.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved plot to {save_path}")
        
        plt.close()
    
    def save_metrics_json(self, agent_name: str) -> None:
        """Save metrics to JSON file.
        
        Args:
            agent_name: Name of agent
        """
        save_path = self.output_dir / f"{agent_name}_metrics.json"
        with open(save_path, "w") as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Saved metrics to {save_path}")


def compare_agents(
    agents_metrics: Dict[str, Dict],
    output_dir: Path | str = "data/plots",
) -> None:
    """Compare multiple agents.
    
    Args:
        agents_metrics: Dictionary with agent names and their metrics
        output_dir: Output directory for plots
    """
    output_dir = Path(output_dir)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Agent Comparison", fontsize=14, fontweight="bold")
    
    agent_names = list(agents_metrics.keys())
    win_rates = [metrics.get("final_win_rate", 0) for metrics in agents_metrics.values()]
    avg_returns = [metrics.get("final_avg_return", 0) for metrics in agents_metrics.values()]
    training_times = [metrics.get("training_time", 0) for metrics in agents_metrics.values()]
    
    # Win Rate comparison
    ax = axes[0]
    colors = plt.cm.get_cmap('Set3')(np.linspace(0, 1, len(agent_names)))
    bars = ax.barh(agent_names, win_rates, color=colors, edgecolor="black", linewidth=1.5)
    ax.set_xlabel("Win Rate")
    ax.set_title("Final Win Rate Comparison")
    ax.set_xlim([0, 1])
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
               f"{width:.1%}", ha="left", va="center", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    
    # Average Return comparison
    ax = axes[1]
    bars = ax.barh(agent_names, avg_returns, color=colors, edgecolor="black", linewidth=1.5)
    ax.set_xlabel("Average Return")
    ax.set_title("Final Average Return Comparison")
    ax.axvline(x=0, color="r", linestyle="--", alpha=0.5)
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
               f"{width:.3f}", ha="left" if width > 0 else "right", va="center", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    
    plt.tight_layout()
    
    save_path = output_dir / "agents_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved comparison plot to {save_path}")
    
    plt.close()
