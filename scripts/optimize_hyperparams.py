"""Optimisation des hyperparamètres pour améliorer les performances."""

import json
import sys
from pathlib import Path
import numpy as np
from itertools import product
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.blackjack_env import BlackjackEnv
from src.environment.deck_config import DeckConfig
from src.agents.naive.qlearning import QLearningAgent
from src.agents.naive.sarsa import SARSAAgent
from src.agents.naive.dqn import DQNAgent
from tqdm import tqdm


def evaluate_config(agent_class, config, num_episodes=10000):
    """Évalue une configuration d'hyperparamètres."""
    deck_config = DeckConfig(deck_type="infinite", natural=True, sab=False)
    env = BlackjackEnv(deck_config=deck_config, enable_counting=False)
    
    # Créer l'agent
    agent = agent_class(config)
    
    # Phase d'entraînement rapide
    for _ in range(50000):
        obs, _ = env.reset()
        done = False
        
        while not done:
            action = agent.act(obs, explore=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(obs, action, reward, next_obs, done)
            obs = next_obs
    
    # Évaluation
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
    
    win_rate = wins / num_episodes
    avg_return = total_return / num_episodes
    
    return win_rate, avg_return


def optimize_qlearning():
    """Optimise les hyperparamètres de Q-Learning."""
    print("\n" + "="*70)
    print("🔍 OPTIMISATION Q-LEARNING")
    print("="*70)
    
    # Grille de recherche
    alphas = [0.05, 0.1, 0.2, 0.3]
    gammas = [0.95, 0.99, 1.0]
    epsilons = [0.1, 0.2, 0.3]
    epsilon_decays = [0.9995, 0.9999, 1.0]
    
    best_config = None
    best_win_rate = 0
    results = []
    
    total = len(alphas) * len(gammas) * len(epsilons) * len(epsilon_decays)
    
    print(f"\n🎯 Test de {total} configurations...\n")
    
    with tqdm(total=total, desc="Recherche") as pbar:
        for alpha, gamma, epsilon, epsilon_decay in product(alphas, gammas, epsilons, epsilon_decays):
            config = {
                'alpha': alpha,
                'gamma': gamma,
                'epsilon': epsilon,
                'epsilon_decay': epsilon_decay,
                'epsilon_min': 0.01,
            }
            
            win_rate, avg_return = evaluate_config(QLearningAgent, config)
            
            results.append({
                'config': config,
                'win_rate': win_rate,
                'avg_return': avg_return,
            })
            
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_config = config
                tqdm.write(f"✨ Nouveau meilleur: {win_rate*100:.1f}% (alpha={alpha}, gamma={gamma}, epsilon={epsilon}, decay={epsilon_decay})")
            
            pbar.update(1)
    
    # Trier par win rate
    results = sorted(results, key=lambda x: x['win_rate'], reverse=True)
    
    print(f"\n{'='*70}")
    print("🏆 TOP 5 CONFIGURATIONS")
    print(f"{'='*70}")
    
    for i, result in enumerate(results[:5], 1):
        config = result['config']
        print(f"\n{i}. Win Rate: {result['win_rate']*100:.1f}%, Avg Return: {result['avg_return']:.4f}")
        print(f"   alpha={config['alpha']}, gamma={config['gamma']}, epsilon={config['epsilon']}, decay={config['epsilon_decay']}")
    
    return results


def optimize_sarsa():
    """Optimise les hyperparamètres de SARSA."""
    print("\n" + "="*70)
    print("🔍 OPTIMISATION SARSA")
    print("="*70)
    
    # Grille de recherche (similaire à Q-Learning)
    alphas = [0.05, 0.1, 0.2, 0.3]
    gammas = [0.95, 0.99, 1.0]
    epsilons = [0.1, 0.2, 0.3]
    epsilon_decays = [0.9995, 0.9999, 1.0]
    
    best_config = None
    best_win_rate = 0
    results = []
    
    total = len(alphas) * len(gammas) * len(epsilons) * len(epsilon_decays)
    
    print(f"\n🎯 Test de {total} configurations...\n")
    
    with tqdm(total=total, desc="Recherche") as pbar:
        for alpha, gamma, epsilon, epsilon_decay in product(alphas, gammas, epsilons, epsilon_decays):
            config = {
                'alpha': alpha,
                'gamma': gamma,
                'epsilon': epsilon,
                'epsilon_decay': epsilon_decay,
                'epsilon_min': 0.01,
            }
            
            win_rate, avg_return = evaluate_config(SARSAAgent, config)
            
            results.append({
                'config': config,
                'win_rate': win_rate,
                'avg_return': avg_return,
            })
            
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_config = config
                tqdm.write(f"✨ Nouveau meilleur: {win_rate*100:.1f}% (alpha={alpha}, gamma={gamma}, epsilon={epsilon}, decay={epsilon_decay})")
            
            pbar.update(1)
    
    # Trier par win rate
    results = sorted(results, key=lambda x: x['win_rate'], reverse=True)
    
    print(f"\n{'='*70}")
    print("🏆 TOP 5 CONFIGURATIONS")
    print(f"{'='*70}")
    
    for i, result in enumerate(results[:5], 1):
        config = result['config']
        print(f"\n{i}. Win Rate: {result['win_rate']*100:.1f}%, Avg Return: {result['avg_return']:.4f}")
        print(f"   alpha={config['alpha']}, gamma={config['gamma']}, epsilon={config['epsilon']}, decay={config['epsilon_decay']}")
    
    return results


def optimize_dqn():
    """Optimise les hyperparamètres de DQN."""
    print("\n" + "="*70)
    print("🔍 OPTIMISATION DQN")
    print("="*70)
    
    # Grille de recherche
    learning_rates = [0.0001, 0.0005, 0.001]
    batch_sizes = [32, 64, 128]
    gammas = [0.95, 0.99]
    
    best_config = None
    best_win_rate = 0
    results = []
    
    total = len(learning_rates) * len(batch_sizes) * len(gammas)
    
    print(f"\n🎯 Test de {total} configurations...\n")
    
    with tqdm(total=total, desc="Recherche") as pbar:
        for lr, batch_size, gamma in product(learning_rates, batch_sizes, gammas):
            config = {
                'learning_rate': lr,
                'gamma': gamma,
                'epsilon': 0.2,
                'epsilon_decay': 0.9995,
                'epsilon_min': 0.01,
                'memory_size': 10000,
                'batch_size': batch_size,
                'hidden_layers': [128, 128],
                'target_update_freq': 1000,
            }
            
            win_rate, avg_return = evaluate_config(DQNAgent, config, num_episodes=5000)
            
            results.append({
                'config': config,
                'win_rate': win_rate,
                'avg_return': avg_return,
            })
            
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_config = config
                tqdm.write(f"✨ Nouveau meilleur: {win_rate*100:.1f}% (lr={lr}, batch={batch_size}, gamma={gamma})")
            
            pbar.update(1)
    
    # Trier par win rate
    results = sorted(results, key=lambda x: x['win_rate'], reverse=True)
    
    print(f"\n{'='*70}")
    print("🏆 TOP 3 CONFIGURATIONS")
    print(f"{'='*70}")
    
    for i, result in enumerate(results[:3], 1):
        config = result['config']
        print(f"\n{i}. Win Rate: {result['win_rate']*100:.1f}%, Avg Return: {result['avg_return']:.4f}")
        print(f"   lr={config['learning_rate']}, batch={config['batch_size']}, gamma={config['gamma']}")
    
    return results


def main():
    """Optimise les hyperparamètres des meilleurs agents."""
    
    all_results = {}
    
    # Q-Learning
    qlearning_results = optimize_qlearning()
    all_results['qlearning'] = qlearning_results[:5]
    
    # SARSA
    sarsa_results = optimize_sarsa()
    all_results['sarsa'] = sarsa_results[:5]
    
    # DQN (prend plus de temps)
    print("\n⚠️  DQN prendra environ 30-45 minutes...")
    response = input("Continuer avec DQN? (o/n): ")
    
    if response.lower() == 'o':
        dqn_results = optimize_dqn()
        all_results['dqn'] = dqn_results[:3]
    
    # Sauvegarder les résultats
    results_path = Path("data/logs/hyperparameter_optimization.json")
    
    # Convertir les résultats pour JSON
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        return obj
    
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=convert_for_json)
    
    print(f"\n✓ Résultats sauvegardés dans {results_path}")
    
    # Résumé final
    print("\n" + "="*70)
    print("📊 RÉSUMÉ DE L'OPTIMISATION")
    print("="*70)
    
    if 'qlearning' in all_results and all_results['qlearning']:
        best_ql = all_results['qlearning'][0]
        print(f"\n🏆 Meilleur Q-Learning: {best_ql['win_rate']*100:.1f}%")
        print(f"   Config: {best_ql['config']}")
    
    if 'sarsa' in all_results and all_results['sarsa']:
        best_sarsa = all_results['sarsa'][0]
        print(f"\n🏆 Meilleur SARSA: {best_sarsa['win_rate']*100:.1f}%")
        print(f"   Config: {best_sarsa['config']}")
    
    if 'dqn' in all_results and all_results['dqn']:
        best_dqn = all_results['dqn'][0]
        print(f"\n🏆 Meilleur DQN: {best_dqn['win_rate']*100:.1f}%")
        print(f"   Config: {best_dqn['config']}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
