#!/usr/bin/env python3
"""Enhanced training script for naive agents with visualization and reporting."""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.naive import MonteCarloAgent, QLearningAgent, SARSAAgent, DQNAgent, DoubleDQNAgent
from src.environment import BlackjackEnv, DeckConfig
from src.evaluation.reporter import DecisionReporter, CardSequenceAnalyzer
from src.evaluation.visualization import TrainingVisualizer
from src.utils import load_config, log_metrics, setup_logger


def evaluate_agent(env: BlackjackEnv, agent, num_episodes: int = 1000) -> dict:
    """Evaluate agent performance.
    
    Args:
        env: Blackjack environment
        agent: Trained agent
        num_episodes: Number of evaluation episodes
        
    Returns:
        Evaluation metrics
    """
    wins = 0
    losses = 0
    draws = 0
    total_return = 0
    returns = []
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_return = 0
        
        while not done:
            action = agent.act(obs, explore=False)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_return += reward
        
        total_return += episode_return
        returns.append(episode_return)
        
        if episode_return > 0:
            wins += 1
        elif episode_return < 0:
            losses += 1
        else:
            draws += 1
    
    return {
        "win_rate": wins / num_episodes,
        "lose_rate": losses / num_episodes,
        "draw_rate": draws / num_episodes,
        "avg_return": total_return / num_episodes,
        "returns": returns,
    }


def test_agent(
    env: BlackjackEnv, 
    agent, 
    reporter: Optional[DecisionReporter] = None,
    analyzer: Optional[CardSequenceAnalyzer] = None,
    num_episodes: int = 1000
) -> dict:
    """Test agent and record decisions.
    
    Args:
        env: Blackjack environment
        agent: Trained agent
        reporter: Optional decision reporter
        analyzer: Optional card sequence analyzer
        num_episodes: Number of test episodes
        
    Returns:
        Test results with metrics
    """
    wins = 0
    losses = 0
    draws = 0
    total_return = 0
    returns = []
    outcomes = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_return = 0
        episode_decisions = []
        step = 0
        
        # Track cards for the episode
        episode_player_cards = info.get('player_cards', [])
        episode_dealer_cards = info.get('dealer_cards', [])
        
        while not done:
            action = agent.act(obs, explore=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Update card tracking
            if 'player_cards' in info:
                episode_player_cards = info['player_cards']
            if 'dealer_cards' in info:
                episode_dealer_cards = info['dealer_cards']
            
            # Record decision with card information
            if reporter:
                reporter.record_decision(
                    state=obs,
                    action=action,
                    reward=reward,
                    next_state=next_obs,
                    episode=episode,
                    step=step,
                    player_cards=episode_player_cards,
                    dealer_cards=episode_dealer_cards,
                    dealer_final_sum=info.get('dealer_final_sum'),
                )
            
            episode_return += reward
            episode_decisions.append("HIT" if action == 1 else "STAND")
            obs = next_obs
            step += 1
        
        total_return += episode_return
        returns.append(episode_return)
        
        if episode_return > 0:
            wins += 1
            outcomes.append("Win")
        elif episode_return < 0:
            losses += 1
            outcomes.append("Lose")
        else:
            draws += 1
            outcomes.append("Draw")
    
    return {
        "win_rate": wins / num_episodes,
        "lose_rate": losses / num_episodes,
        "draw_rate": draws / num_episodes,
        "avg_return": total_return / num_episodes,
        "returns": returns,
        "outcomes": outcomes,
    }


def train_agent(
    config_path: str,
    output_dir: str | Path = "data",
    enable_visualization: bool = True,
    enable_reporting: bool = True,
    enable_optimization: bool = False,
) -> None:
    """Train agent using configuration file.
    
    Args:
        config_path: Path to agent configuration YAML
        output_dir: Output directory for models and logs
        enable_visualization: Whether to generate plots
        enable_reporting: Whether to generate reports
        enable_optimization: Whether to perform fine-tuning
    """
    start_time = time.time()
    
    # Load configuration
    config = load_config(config_path)
    
    # Setup paths
    output_dir = Path(output_dir)
    models_dir = output_dir / "models" / "naive"
    logs_dir = output_dir / "logs"
    plots_dir = output_dir / "plots"
    reports_dir = output_dir / "reports"
    
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    agent_type = config["agent"]["type"]
    logger = setup_logger(f"train_{agent_type}", logs_dir)
    logger.info(f"Starting enhanced training for {agent_type}")
    logger.info(f"Configuration: {config}")
    
    # Create environment
    env_config = config["environment"]
    deck_config = DeckConfig(
        deck_type=env_config["deck_type"],
        num_decks=env_config["num_decks"],
        natural=env_config.get("natural", True),
        sab=env_config.get("sab", True),
    )
    env = BlackjackEnv(deck_config=deck_config, enable_counting=False)
    
    # Set random seed
    seed = config["training"]["seed"]
    np.random.seed(seed)
    
    # Create agent
    agent_classes = {
        "monte_carlo": MonteCarloAgent,
        "qlearning": QLearningAgent,
        "sarsa": SARSAAgent,
        "dqn": DQNAgent,
        "double_dqn": DoubleDQNAgent,
    }
    
    if agent_type not in agent_classes:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    agent = agent_classes[agent_type](config["hyperparameters"])
    
    # Initialize visualization and reporting
    visualizer = TrainingVisualizer(plots_dir) if enable_visualization else None
    reporter = DecisionReporter(reports_dir) if enable_reporting else None
    
    # Training parameters
    total_episodes = config["training"]["episodes"]
    eval_frequency = config["training"]["eval_frequency"]
    eval_episodes = config["training"]["eval_episodes"]
    
    # Training loop
    logger.info(f"Training for {total_episodes} episodes")
    
    for episode in tqdm(range(1, total_episodes + 1), desc=f"Training {agent_type}"):
        obs, _ = env.reset()
        done = False
        action: int = agent.act(obs, explore=True)
        
        while not done:
            if agent_type == "sarsa":
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                if not done:
                    next_action: int = agent.act(next_obs, explore=True)
                    agent.update(obs, action, reward, next_obs, done, next_action=next_action)
                    action = next_action
                else:
                    agent.update(obs, action, reward, next_obs, done, next_action=None)
                
                obs = next_obs
            else:
                action = agent.act(obs, explore=True)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                agent.update(obs, action, reward, next_obs, done)
                obs = next_obs
        
        # Evaluation
        if episode % eval_frequency == 0:
            eval_metrics = evaluate_agent(env, agent, eval_episodes)
            metrics = {
                "episode": episode,
                "epsilon": agent.epsilon,
                **eval_metrics,
            }
            log_metrics(logger, metrics)
            
            if visualizer:
                visualizer.add_metrics(
                    episode=episode,
                    win_rate=eval_metrics["win_rate"],
                    lose_rate=eval_metrics["lose_rate"],
                    draw_rate=eval_metrics["draw_rate"],
                    avg_return=eval_metrics["avg_return"],
                    epsilon=agent.epsilon,
                    loss=None,
                )
            
            print(f"\nEpisode {episode}/{total_episodes}")
            print(f"  Win Rate: {eval_metrics['win_rate']:.1%}")
            print(f"  Avg Return: {eval_metrics['avg_return']:.3f}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
    
    # Final save (only final model)
    final_model_path = models_dir / f"{agent_type}_final.pkl"
    agent.save(final_model_path)
    logger.info(f"Training complete! Final model saved to {final_model_path}")
    
    # Final evaluation
    final_eval_metrics = evaluate_agent(env, agent, eval_episodes * 2)
    logger.info(f"Final evaluation metrics: {final_eval_metrics}")
    
    # Test phase with reporting
    print(f"\n{'='*50}")
    print(f"Testing phase with decision recording...")
    print(f"{'='*50}")
    
    test_metrics = test_agent(
        env,
        agent,
        reporter=reporter,
        num_episodes=min(1000, total_episodes // 10),
    )
    
    # Generate visualizations
    if visualizer:
        visualizer.plot_training_curves(agent_type, save=True)
        visualizer.plot_eval_comparison(agent_type, final_eval_metrics, final_eval_metrics, save=True)
        visualizer.plot_test_results(agent_type, test_metrics, save=True)
        visualizer.save_metrics_json(agent_type)
    
    # Generate reports
    if reporter:
        reporter.save_report(agent_type)
    
    # Training summary
    training_time = time.time() - start_time
    
    summary = {
        "agent_type": agent_type,
        "training_time_seconds": training_time,
        "total_episodes": total_episodes,
        "final_win_rate": final_eval_metrics["win_rate"],
        "final_avg_return": final_eval_metrics["avg_return"],
        "test_win_rate": test_metrics["win_rate"],
        "test_avg_return": test_metrics["avg_return"],
        "model_path": str(final_model_path),
    }
    
    summary_path = logs_dir / f"{agent_type}_training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Training Complete!")
    print(f"{'='*50}")
    print(f"Agent Type: {agent_type}")
    print(f"Training Time: {training_time:.2f}s")
    print(f"Final Win Rate: {final_eval_metrics['win_rate']:.1%}")
    print(f"Final Avg Return: {final_eval_metrics['avg_return']:.3f}")
    print(f"Test Win Rate: {test_metrics['win_rate']:.1%}")
    print(f"Model saved to: {final_model_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*50}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train naive Blackjack agents with enhanced features")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to agent configuration YAML file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for models and logs (default: data)",
    )
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Disable plot generation",
    )
    parser.add_argument(
        "--no-reporting",
        action="store_true",
        help="Disable JSON report generation",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Enable fine-tuning after training",
    )
    
    args = parser.parse_args()
    
    train_agent(
        args.config,
        args.output_dir,
        enable_visualization=not args.no_visualization,
        enable_reporting=not args.no_reporting,
        enable_optimization=args.optimize,
    )


if __name__ == "__main__":
    main()
