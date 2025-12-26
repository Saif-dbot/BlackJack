"""Compare les agents RL avec la Basic Strategy optimale."""

import sys
import os
from pathlib import Path
import json
import pickle
from tqdm import tqdm
import numpy as np

# Changer vers le répertoire du projet
project_root = Path(__file__).parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

from src.environment.blackjack_env import BlackjackEnv
from src.environment.deck_config import DeckConfig
from src.agents.basic_strategy import BasicStrategyAgent
from src.agents.naive.qlearning import QLearningAgent
from src.agents.naive.sarsa import SARSAAgent
from src.agents.naive.dqn import DQNAgent
from src.agents.naive.monte_carlo import MonteCarloAgent


def evaluate_agent(agent, num_episodes=10000, agent_name="Agent"):
    """Évalue un agent sur un nombre d'épisodes."""
    deck_config = DeckConfig(deck_type="infinite", natural=True, sab=False)
    env = BlackjackEnv(deck_config=deck_config, enable_counting=False)
    
    wins = 0
    losses = 0
    draws = 0
    returns = []
    
    print(f"\n📊 Évaluation de {agent_name} sur {num_episodes} épisodes...")
    
    for _ in tqdm(range(num_episodes), desc=f"  {agent_name}"):
        obs, info = env.reset()
        done = False
        episode_return = 0
        
        while not done:
            action = agent.act(obs, explore=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_return += reward
            obs = next_obs
        
        returns.append(episode_return)
        
        if episode_return > 0:
            wins += 1
        elif episode_return < 0:
            losses += 1
        else:
            draws += 1
    
    win_rate = wins / num_episodes
    avg_return = np.mean(returns)
    std_return = np.std(returns)
    
    return {
        'agent_name': agent_name,
        'num_episodes': num_episodes,
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'win_rate': win_rate,
        'loss_rate': losses / num_episodes,
        'draw_rate': draws / num_episodes,
        'avg_return': avg_return,
        'std_return': std_return
    }


def compare_decisions(rl_agent, basic_agent, num_episodes=1000):
    """Compare les décisions d'un agent RL avec la Basic Strategy."""
    deck_config = DeckConfig(deck_type="infinite", natural=True, sab=False)
    env = BlackjackEnv(deck_config=deck_config, enable_counting=False)
    
    same_decisions = 0
    different_decisions = 0
    disagreements = []
    
    print(f"\n🔍 Comparaison des décisions sur {num_episodes} épisodes...")
    
    for _ in tqdm(range(num_episodes), desc="  Analyse"):
        obs, info = env.reset()
        done = False
        
        while not done:
            rl_action = rl_agent.act(obs, explore=False)
            basic_action = basic_agent.act(obs, explore=False)
            
            if rl_action == basic_action:
                same_decisions += 1
            else:
                different_decisions += 1
                disagreements.append({
                    'state': obs,
                    'rl_action': rl_action,
                    'basic_action': basic_action
                })
            
            next_obs, reward, terminated, truncated, info = env.step(rl_action)
            done = terminated or truncated
            obs = next_obs
    
    total_decisions = same_decisions + different_decisions
    agreement_rate = same_decisions / total_decisions if total_decisions > 0 else 0
    
    return {
        'total_decisions': total_decisions,
        'same_decisions': same_decisions,
        'different_decisions': different_decisions,
        'agreement_rate': agreement_rate,
        'top_disagreements': disagreements[:20]  # Top 20 désaccords
    }


def load_trained_agent(agent_name, model_path):
    """Charge un agent entraîné."""
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        # Si c'est un dictionnaire, recréer l'agent
        if isinstance(data, dict):
            # Déterminer le type d'agent
            if 'qlearning' in agent_name.lower():
                from src.agents.naive.qlearning import QLearningAgent
                agent = QLearningAgent(data['config'])
                agent.q_table = data['q_table']
                agent.epsilon = data['epsilon']
            elif 'sarsa' in agent_name.lower():
                from src.agents.naive.sarsa import SARSAAgent
                agent = SARSAAgent(data['config'])
                agent.q_table = data['q_table']
                agent.epsilon = data['epsilon']
            elif 'monte' in agent_name.lower():
                from src.agents.naive.monte_carlo import MonteCarloAgent
                agent = MonteCarloAgent(data['config'])
                agent.returns = data['returns']
                agent.q_table = data.get('q_table', {})
            elif 'dqn' in agent_name.lower() or 'double' in agent_name.lower():
                if 'double' in agent_name.lower():
                    from src.agents.naive.double_dqn import DoubleDQNAgent
                    agent = DoubleDQNAgent(data['config'])
                else:
                    from src.agents.naive.dqn import DQNAgent
                    agent = DQNAgent(data['config'])
                # Charger les poids du réseau
                if 'q_network_state' in data:
                    agent.q_network.load_state_dict(data['q_network_state'])
                if 'target_network_state' in data:
                    agent.target_network.load_state_dict(data['target_network_state'])
                agent.epsilon = data['epsilon']
            else:
                print(f"❌ Type d'agent inconnu: {agent_name}")
                return None
            
            return agent
        else:
            # Déjà un objet agent
            return data
            
    except Exception as e:
        print(f"❌ Erreur de chargement {agent_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Compare tous les agents RL avec la Basic Strategy."""
    print("\n" + "="*70)
    print("🎯 COMPARAISON AVEC BASIC STRATEGY")
    print("="*70)
    
    # Créer l'agent Basic Strategy
    basic_agent = BasicStrategyAgent()
    
    # Afficher la stratégie
    print("\n" + basic_agent.get_policy_string())
    
    # Évaluer Basic Strategy
    basic_results = evaluate_agent(basic_agent, num_episodes=50000, agent_name="Basic Strategy")
    
    print(f"\n{'='*70}")
    print("📊 RÉSULTATS BASIC STRATEGY")
    print(f"{'='*70}")
    print(f"Win Rate:    {basic_results['win_rate']*100:.2f}%")
    print(f"Loss Rate:   {basic_results['loss_rate']*100:.2f}%")
    print(f"Draw Rate:   {basic_results['draw_rate']*100:.2f}%")
    print(f"Avg Return:  {basic_results['avg_return']:.4f} ± {basic_results['std_return']:.4f}")
    
    # Charger et évaluer les agents RL
    agents_to_test = [
        ('Q-Learning', 'data/models/naive/qlearning_final.pkl'),
        ('SARSA', 'data/models/naive/sarsa_final.pkl'),
        ('DQN', 'data/models/naive/dqn_final.pkl'),
        ('Monte Carlo', 'data/models/naive/monte_carlo_final.pkl'),
        ('Double DQN', 'data/models/naive/double_dqn_final.pkl'),
        ('Q-Learning Optimized', 'data/models/optimized/qlearning_optimized.pkl'),
        ('SARSA Optimized', 'data/models/optimized/sarsa_optimized.pkl'),
        ('DQN Optimized', 'data/models/optimized/dqn_optimized.pkl'),
    ]
    
    all_results = [basic_results]
    comparison_results = {}
    
    for agent_name, model_path in agents_to_test:
        model_path = Path(model_path)
        
        if not model_path.exists():
            print(f"\n⚠️  Modèle non trouvé: {model_path}")
            continue
        
        agent = load_trained_agent(agent_name, model_path)
        
        if agent is None:
            continue
        
        # Évaluer l'agent
        results = evaluate_agent(agent, num_episodes=50000, agent_name=agent_name)
        all_results.append(results)
        
        # Comparer les décisions avec Basic Strategy
        comparison = compare_decisions(agent, basic_agent, num_episodes=1000)
        comparison_results[agent_name] = comparison
        
        print(f"\n{'='*70}")
        print(f"📊 {agent_name}")
        print(f"{'='*70}")
        print(f"Win Rate:    {results['win_rate']*100:.2f}%")
        print(f"Avg Return:  {results['avg_return']:.4f} ± {results['std_return']:.4f}")
        print(f"Agreement:   {comparison['agreement_rate']*100:.2f}% avec Basic Strategy")
        print(f"Différences: {comparison['different_decisions']} / {comparison['total_decisions']} décisions")
    
    # Tableau comparatif final
    print(f"\n{'='*70}")
    print("📊 TABLEAU COMPARATIF FINAL")
    print(f"{'='*70}")
    print(f"{'Agent':<20} {'Win Rate':<12} {'Avg Return':<15} {'Écart vs Basic'}")
    print("-" * 70)
    
    basic_wr = basic_results['win_rate']
    basic_ar = basic_results['avg_return']
    
    for result in all_results:
        name = result['agent_name']
        wr = result['win_rate']
        ar = result['avg_return']
        
        wr_diff = (wr - basic_wr) * 100
        ar_diff = ar - basic_ar
        
        print(f"{name:<20} {wr*100:>6.2f}%      {ar:>8.4f}        {wr_diff:+.2f}% / {ar_diff:+.4f}")
    
    # Sauvegarder les résultats
    output_dir = Path("data/logs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'basic_strategy': basic_results,
        'rl_agents': {r['agent_name']: r for r in all_results[1:]},
        'decision_comparisons': comparison_results
    }
    
    output_path = output_dir / "basic_strategy_comparison.json"
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Résultats sauvegardés: {output_path}")
    
    # Analyser les désaccords majeurs
    print(f"\n{'='*70}")
    print("🔍 ANALYSE DES DÉSACCORDS")
    print(f"{'='*70}")
    
    for agent_name, comparison in comparison_results.items():
        print(f"\n{agent_name}:")
        print(f"  Taux d'accord: {comparison['agreement_rate']*100:.2f}%")
        
        if comparison['top_disagreements']:
            print("  Désaccords fréquents:")
            disagreement_counts = {}
            
            for dis in comparison['top_disagreements']:
                state = dis['state']
                key = (state[0], state[1], state[2])
                disagreement_counts[key] = disagreement_counts.get(key, 0) + 1
            
            top_5 = sorted(disagreement_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            for (player_sum, dealer_card, usable_ace), count in top_5:
                ace_str = "Soft" if usable_ace else "Hard"
                print(f"    {ace_str} {player_sum} vs Dealer {dealer_card}: {count} fois")


if __name__ == "__main__":
    main()
