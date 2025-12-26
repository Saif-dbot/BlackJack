"""Hyperparameter optimization and fine-tuning utilities."""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm


class HyperparameterOptimizer:
    """Optimize agent hyperparameters."""
    
    def __init__(self, output_dir: Path | str = "data/optimization"):
        """Initialize optimizer.
        
        Args:
            output_dir: Directory to save optimization results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[Dict] = []
    
    def grid_search(
        self,
        param_grid: Dict[str, List[Any]],
        evaluate_fn: Callable[[Dict], float],
        agent_name: str = "agent",
        verbose: bool = True,
    ) -> Dict:
        """Perform grid search over hyperparameters.
        
        Args:
            param_grid: Dictionary mapping parameter names to lists of values
            evaluate_fn: Function that takes hyperparameters and returns score
            agent_name: Name of agent
            verbose: Whether to print progress
            
        Returns:
            Best hyperparameters and score
        """
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        
        # Generate all combinations
        from itertools import product
        combinations = list(product(*param_values))
        
        if verbose:
            print(f"Starting grid search with {len(combinations)} combinations...")
        
        best_score = -np.inf
        best_params = {}
        
        for combo in tqdm(combinations, desc=f"Grid Search for {agent_name}", disable=not verbose):
            params = dict(zip(param_names, combo))
            
            try:
                score = evaluate_fn(params)
                self.results.append({"params": params, "score": score})
                
                if score > best_score:
                    best_score = score
                    best_params = params
            except Exception as e:
                if verbose:
                    print(f"Error evaluating {params}: {e}")
                continue
        
        result = {
            "best_params": best_params,
            "best_score": best_score,
            "total_evaluations": len(self.results),
            "all_results": self.results,
        }
        
        if verbose:
            print(f"Best score: {best_score:.4f}")
            print(f"Best params: {best_params}")
        
        return result
    
    def random_search(
        self,
        param_distributions: Dict[str, Callable],
        evaluate_fn: Callable[[Dict], float],
        num_trials: int = 20,
        agent_name: str = "agent",
        verbose: bool = True,
    ) -> Dict:
        """Perform random search over hyperparameters.
        
        Args:
            param_distributions: Dictionary mapping parameter names to sampling functions
            evaluate_fn: Function that takes hyperparameters and returns score
            num_trials: Number of random trials
            agent_name: Name of agent
            verbose: Whether to print progress
            
        Returns:
            Best hyperparameters and score
        """
        if verbose:
            print(f"Starting random search with {num_trials} trials...")
        
        best_score = -np.inf
        best_params = {}
        
        for trial in tqdm(range(num_trials), desc=f"Random Search for {agent_name}", disable=not verbose):
            # Sample random parameters
            params = {name: dist_fn() for name, dist_fn in param_distributions.items()}
            
            try:
                score = evaluate_fn(params)
                self.results.append({"params": params, "score": score})
                
                if score > best_score:
                    best_score = score
                    best_params = params
            except Exception as e:
                if verbose:
                    print(f"Error evaluating trial {trial}: {e}")
                continue
        
        result = {
            "best_params": best_params,
            "best_score": best_score,
            "total_evaluations": len(self.results),
            "all_results": self.results,
        }
        
        if verbose:
            print(f"Best score: {best_score:.4f}")
            print(f"Best params: {best_params}")
        
        return result
    
    def save_results(self, agent_name: str) -> None:
        """Save optimization results.
        
        Args:
            agent_name: Name of agent
        """
        save_path = self.output_dir / f"{agent_name}_optimization_results.json"
        
        # Convert numpy types to Python types for JSON serialization
        results = []
        for result in self.results:
            params = {k: float(v) if isinstance(v, np.number) else v 
                     for k, v in result["params"].items()}
            results.append({
                "params": params,
                "score": float(result["score"]),
            })
        
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Saved optimization results to {save_path}")


class FineTuner:
    """Fine-tune trained agents."""
    
    def __init__(self, output_dir: Path | str = "data/finetuned"):
        """Initialize fine-tuner.
        
        Args:
            output_dir: Directory to save fine-tuned models
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history: Dict[str, List] = {
            "episode": [],
            "win_rate": [],
            "avg_return": [],
        }
    
    def finetune_agent(
        self,
        agent: Any,
        env: Any,
        num_episodes: int = 10000,
        eval_frequency: int = 500,
        eval_episodes: int = 100,
        learning_rate_decay: float = 0.98,
        verbose: bool = True,
    ) -> Dict:
        """Fine-tune a trained agent.
        
        Args:
            agent: Trained agent to fine-tune
            env: Training environment
            num_episodes: Number of fine-tuning episodes
            eval_frequency: Evaluation frequency
            eval_episodes: Number of eval episodes
            learning_rate_decay: Decay factor for learning rate
            verbose: Whether to print progress
            
        Returns:
            Fine-tuning history
        """
        from tqdm import tqdm
        
        original_alpha = getattr(agent, 'alpha', None)
        
        iterator = tqdm(range(1, num_episodes + 1), desc="Fine-tuning") if verbose else range(1, num_episodes + 1)
        
        for episode in iterator:
            obs, _ = env.reset()
            done = False
            
            # Apply learning rate decay
            if hasattr(agent, 'alpha') and original_alpha:
                agent.alpha = original_alpha * (learning_rate_decay ** (episode / 1000))
            
            while not done:
                action = agent.act(obs, explore=True)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                agent.update(obs, action, reward, next_obs, done)
                obs = next_obs
            
            # Evaluation
            if episode % eval_frequency == 0:
                wins = 0
                total_return = 0
                
                for _ in range(eval_episodes):
                    obs, _ = env.reset()
                    done = False
                    episode_return = 0
                    
                    while not done:
                        action = agent.act(obs, explore=False)
                        obs, reward, terminated, truncated, _ = env.step(action)
                        done = terminated or truncated
                        episode_return += reward
                    
                    total_return += episode_return
                    if episode_return > 0:
                        wins += 1
                
                win_rate = wins / eval_episodes
                avg_return = total_return / eval_episodes
                
                self.history["episode"].append(episode)
                self.history["win_rate"].append(win_rate)
                self.history["avg_return"].append(avg_return)
                
                if verbose:
                    print(f"Episode {episode}: WR={win_rate:.1%}, AR={avg_return:.3f}")
        
        return self.history
    
    def save_history(self, agent_name: str) -> None:
        """Save fine-tuning history.
        
        Args:
            agent_name: Name of agent
        """
        save_path = self.output_dir / f"{agent_name}_finetuning_history.json"
        
        history = {
            "episode": [int(e) for e in self.history["episode"]],
            "win_rate": [float(w) for w in self.history["win_rate"]],
            "avg_return": [float(r) for r in self.history["avg_return"]],
        }
        
        with open(save_path, "w") as f:
            json.dump(history, f, indent=2)
        
        print(f"Saved fine-tuning history to {save_path}")


class AdaptiveScheduler:
    """Adaptive scheduling for learning rates and exploration."""
    
    @staticmethod
    def linear_schedule(initial_value: float, final_value: float, total_steps: int) -> Callable:
        """Create linear schedule.
        
        Args:
            initial_value: Initial value
            final_value: Final value
            total_steps: Total steps
            
        Returns:
            Schedule function
        """
        def schedule(step: int) -> float:
            return initial_value + (final_value - initial_value) * (step / total_steps)
        return schedule
    
    @staticmethod
    def exponential_schedule(initial_value: float, final_value: float, decay_rate: float) -> Callable:
        """Create exponential schedule.
        
        Args:
            initial_value: Initial value
            final_value: Final value
            decay_rate: Decay rate (0-1)
            
        Returns:
            Schedule function
        """
        def schedule(step: int) -> float:
            return max(final_value, initial_value * (decay_rate ** step))
        return schedule
    
    @staticmethod
    def cosine_schedule(initial_value: float, final_value: float, total_steps: int) -> Callable:
        """Create cosine annealing schedule.
        
        Args:
            initial_value: Initial value
            final_value: Final value
            total_steps: Total steps
            
        Returns:
            Schedule function
        """
        def schedule(step: int) -> float:
            progress = min(step / total_steps, 1.0)
            return final_value + 0.5 * (initial_value - final_value) * (1 + np.cos(np.pi * progress))
        return schedule
    
    @staticmethod
    def step_schedule(initial_value: float, step_size: int, gamma: float) -> Callable:
        """Create step decay schedule.
        
        Args:
            initial_value: Initial value
            step_size: Number of steps between decay
            gamma: Decay factor
            
        Returns:
            Schedule function
        """
        def schedule(step: int) -> float:
            return initial_value * (gamma ** (step // step_size))
        return schedule


def create_optimization_config(agent_name: str) -> Dict:
    """Create default optimization configuration.
    
    Args:
        agent_name: Name of agent
        
    Returns:
        Optimization configuration
    """
    configs = {
        "monte_carlo": {
            "grid_search": {
                "alpha": [0.01, 0.05, 0.1, 0.2],
                "gamma": [0.95, 0.99, 1.0],
                "epsilon_start": [0.5, 1.0],
                "epsilon_decay": [0.9995, 0.999, 0.99],
            }
        },
        "qlearning": {
            "grid_search": {
                "alpha": [0.01, 0.05, 0.1, 0.2, 0.3],
                "gamma": [0.95, 0.99, 1.0],
                "epsilon_start": [0.5, 1.0],
                "epsilon_decay": [0.9995, 0.999, 0.99],
            }
        },
        "sarsa": {
            "grid_search": {
                "alpha": [0.01, 0.05, 0.1, 0.2],
                "gamma": [0.95, 0.99, 1.0],
                "epsilon_start": [0.5, 1.0],
                "epsilon_decay": [0.9995, 0.999, 0.99],
            }
        },
        "dqn": {
            "grid_search": {
                "learning_rate": [0.0001, 0.0005, 0.001, 0.002],
                "batch_size": [32, 64, 128],
                "target_update_freq": [500, 1000, 2000],
                "hidden_dim": [64, 128, 256],
            }
        },
        "double_dqn": {
            "grid_search": {
                "learning_rate": [0.0001, 0.0005, 0.001, 0.002],
                "batch_size": [32, 64, 128],
                "target_update_freq": [500, 1000, 2000],
                "hidden_dim": [64, 128, 256],
            }
        },
    }
    
    return configs.get(agent_name, {})
