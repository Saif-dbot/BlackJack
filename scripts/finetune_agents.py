"""Script pour fine-tuner les meilleurs agents."""

import json
import pickle
import sys
from pathlib import Path
from datetime import datetime
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.blackjack_env import BlackjackEnv
from src.environment.deck_config import DeckConfig
from src.agents.naive.qlearning import QLearningAgent
from src.agents.naive.sarsa import SARSAAgent
from src.agents.naive.monte_carlo import MonteCarloAgent
from src.agents.naive.dqn import DQNAgent
from src.agents.naive.double_dqn import DoubleDQNAgent
from src.agents.counting.qlearning_count import QLearningCountAgent
from src.agents.counting.sarsa_count import SARSACountAgent
from src.agents.counting.dqn_count import DQNCountAgent
from tqdm import tqdm


def evaluate_agent(agent, env, num_episodes=1000):
    """Évalue un agent."""
    total_return = 0
    wins = 0
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_return = 0
        
        while not done:
            action = agent.act(obs, explore=False)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_return += reward
            obs = next_obs
        
        total_return += episode_return
        if episode_return > 0:
            wins += 1
    
    return {
        'win_rate': wins / num_episodes,
        'avg_return': total_return / num_episodes,
    }


def finetune_agent(agent, env, config, episodes=50000):
    """Fine-tune un agent avec learning rate réduit."""
    
    # Réduire le learning rate
    if hasattr(agent, 'alpha'):
        original_alpha = agent.alpha
        agent.alpha = original_alpha * 0.1  # 10% du LR original
    elif hasattr(agent, 'learning_rate'):
        original_lr = agent.learning_rate
        agent.learning_rate = original_lr * 0.1
    
    # Réduire epsilon pour moins d'exploration
    if hasattr(agent, 'epsilon'):
        agent.epsilon = 0.05  # Exploration minimale
        agent.epsilon_min = 0.01
    
    print(f"  Fine-tuning avec {episodes} épisodes...")
    
    # Training loop
    with tqdm(total=episodes, desc="  Fine-tuning", leave=False) as pbar:
        for episode in range(episodes):
            obs, _ = env.reset()
            done = False
            
            while not done:
                action = agent.act(obs, explore=True)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Update agent
                agent.update(obs, action, reward, next_obs, done)
                obs = next_obs
            
            pbar.update(1)
    
    return agent


def main():
    """Fine-tune les meilleurs agents."""
    print("\n" + "="*70)
    print("🎯 FINE-TUNING DES MEILLEURS AGENTS")
    print("="*70 + "\n")
    
    # Créer l'environnement
    deck_config = DeckConfig(deck_type="infinite", natural=True, sab=False)
    env = BlackjackEnv(deck_config=deck_config, enable_counting=False)
    
    # Charger les résumés pour identifier les meilleurs agents
    logs_dir = Path("data/logs")
    summaries = {}
    
    for summary_file in logs_dir.glob("*_training_summary.json"):
        with open(summary_file, 'r') as f:
            data = json.load(f)
            agent_name = summary_file.stem.replace("_training_summary", "")
            summaries[agent_name] = data
    
    # Trier par win rate
    sorted_agents = sorted(
        summaries.items(),
        key=lambda x: x[1].get('test_win_rate', 0),
        reverse=True
    )
    
    print("📊 Agents disponibles (triés par win rate):")
    for i, (name, data) in enumerate(sorted_agents[:5], 1):
        print(f"  {i}. {name:20s} → {data.get('test_win_rate', 0)*100:.1f}%")
    print()
    
    # Sélectionner les 3 meilleurs agents naïfs
    agents_to_finetune = []
    for name, data in sorted_agents[:3]:
        if '_count' not in name:  # Seulement les agents naïfs
            agents_to_finetune.append((name, data))
    
    print(f"🎯 Fine-tuning des {len(agents_to_finetune)} meilleurs agents naïfs:\n")
    
    results = {}
    
    for agent_name, original_data in agents_to_finetune:
        print(f"{'='*70}")
        print(f"Agent: {agent_name}")
        print(f"{'='*70}")
        
        # Charger le modèle
        model_path = Path(f"data/models/naive/{agent_name}_final.pkl")
        
        if not model_path.exists():
            print(f"  ✗ Modèle non trouvé: {model_path}")
            continue
        
        # Déterminer le type d'agent
        agent_classes = {
            'qlearning': QLearningAgent,
            'sarsa': SARSAAgent,
            'monte_carlo': MonteCarloAgent,
            'dqn': DQNAgent,
            'double_dqn': DoubleDQNAgent,
        }
        
        agent_type = None
        for key in agent_classes.keys():
            if key in agent_name:
                agent_type = key
                break
        
        if not agent_type:
            print(f"  ✗ Type d'agent inconnu")
            continue
        
        # Créer et charger l'agent
        agent_class = agent_classes[agent_type]
        
        # Charger la config
        config_path = Path(f"config/agents_naive/{agent_name}.yaml")
        if not config_path.exists():
            # Essayer sans le suffixe
            config_path = Path(f"config/agents_naive/{agent_type}.yaml")
        
        if config_path.exists():
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            agent_config = config.get('hyperparameters', {})
        else:
            agent_config = {}
        
        agent = agent_class(agent_config)
        agent.load(model_path)
        
        # Évaluer avant fine-tuning
        print("  📊 Évaluation avant fine-tuning...")
        before_metrics = evaluate_agent(agent, env, num_episodes=1000)
        print(f"     Win Rate: {before_metrics['win_rate']*100:.1f}%")
        print(f"     Avg Return: {before_metrics['avg_return']:.4f}")
        
        # Fine-tuning
        start_time = datetime.now()
        agent = finetune_agent(agent, env, agent_config, episodes=50000)
        finetune_time = (datetime.now() - start_time).total_seconds()
        
        # Évaluer après fine-tuning
        print("  📊 Évaluation après fine-tuning...")
        after_metrics = evaluate_agent(agent, env, num_episodes=1000)
        print(f"     Win Rate: {after_metrics['win_rate']*100:.1f}%")
        print(f"     Avg Return: {after_metrics['avg_return']:.4f}")
        
        # Calculer l'amélioration
        win_rate_improvement = (after_metrics['win_rate'] - before_metrics['win_rate']) * 100
        return_improvement = after_metrics['avg_return'] - before_metrics['avg_return']
        
        print(f"\n  📈 Amélioration:")
        print(f"     Win Rate: {win_rate_improvement:+.1f} points")
        print(f"     Avg Return: {return_improvement:+.4f}")
        print(f"     Temps: {finetune_time:.1f}s")
        
        # Sauvegarder si amélioration
        if after_metrics['win_rate'] > before_metrics['win_rate']:
            print(f"  ✓ Amélioration détectée - Sauvegarde du modèle fine-tuné")
            finetuned_path = Path(f"data/models/naive/{agent_name}_finetuned.pkl")
            agent.save(finetuned_path)
        else:
            print(f"  ✗ Pas d'amélioration - Conservation du modèle original")
        
        results[agent_name] = {
            'before': before_metrics,
            'after': after_metrics,
            'improvement': {
                'win_rate': win_rate_improvement,
                'avg_return': return_improvement,
            },
            'time': finetune_time,
        }
        
        print()
    
    # Résumé final
    print("="*70)
    print("📊 RÉSUMÉ DU FINE-TUNING")
    print("="*70)
    
    for agent_name, data in results.items():
        before = data['before']
        after = data['after']
        improvement = data['improvement']
        
        print(f"\n{agent_name}:")
        print(f"  Avant:  {before['win_rate']*100:.1f}% win rate, {before['avg_return']:.4f} avg return")
        print(f"  Après:  {after['win_rate']*100:.1f}% win rate, {after['avg_return']:.4f} avg return")
        print(f"  Gain:   {improvement['win_rate']:+.1f} points, {improvement['avg_return']:+.4f} return")
        
        if after['win_rate'] > before['win_rate']:
            print(f"  ✓ Amélioré")
        else:
            print(f"  → Stable")
    
    # Sauvegarder les résultats
    results_path = Path("data/logs/finetuning_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✓ Résultats sauvegardés dans {results_path}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
