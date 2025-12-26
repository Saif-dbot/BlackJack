"""Réentraîne les agents avec les hyperparamètres optimisés."""

import json
import sys
from pathlib import Path
from datetime import datetime
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.blackjack_env import BlackjackEnv
from src.environment.deck_config import DeckConfig
from src.agents.naive.qlearning import QLearningAgent
from src.agents.naive.sarsa import SARSAAgent
from src.agents.naive.dqn import DQNAgent
from src.evaluation.reporter import DecisionReporter
from src.evaluation.visualization import TrainingVisualizer
from tqdm import tqdm


def train_agent(agent_class, config, agent_name, num_episodes=250000):
    """Entraîne un agent avec les hyperparamètres optimisés."""
    print(f"\n{'='*70}")
    print(f"🎯 Entraînement: {agent_name}")
    print(f"{'='*70}")
    print(f"Configuration: {config}")
    
    # Créer l'environnement
    deck_config = DeckConfig(deck_type="infinite", natural=True, sab=False)
    env = BlackjackEnv(deck_config=deck_config, enable_counting=False)
    
    # Créer l'agent
    agent = agent_class(config)
    
    # Créer le reporter
    reporter = DecisionReporter()
    
    # Métriques
    episode_returns = []
    episode_lengths = []
    wins = 0
    
    print(f"\n⏳ Entraînement sur {num_episodes} épisodes...")
    start_time = datetime.now()
    
    # Training loop
    with tqdm(total=num_episodes, desc="  Entraînement") as pbar:
        for episode in range(num_episodes):
            obs, info = env.reset()
            done = False
            episode_return = 0
            episode_length = 0
            
            while not done:
                action = agent.act(obs, explore=True)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                agent.update(obs, action, reward, next_obs, done)
                
                episode_return += reward
                episode_length += 1
                obs = next_obs
            
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
            
            if episode_return > 0:
                wins += 1
            
            pbar.update(1)
            
            if (episode + 1) % 50000 == 0:
                avg_return = sum(episode_returns[-10000:]) / 10000
                win_rate = sum(1 for r in episode_returns[-10000:] if r > 0) / 10000
                pbar.set_postfix({
                    'win_rate': f'{win_rate*100:.1f}%',
                    'avg_return': f'{avg_return:.4f}'
                })
    
    train_time = (datetime.now() - start_time).total_seconds()
    
    # Évaluation finale
    print("\n📊 Évaluation finale (10000 épisodes)...")
    test_wins = 0
    test_returns = []
    
    for _ in tqdm(range(10000), desc="  Test", leave=False):
        obs, info = env.reset()
        done = False
        episode_return = 0
        episode_player_cards = []
        episode_dealer_cards = []
        
        while not done:
            action = agent.act(obs, explore=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Récupérer les cartes de l'info
            if 'player_cards' in info:
                episode_player_cards = info['player_cards']
            if 'dealer_cards' in info:
                episode_dealer_cards = info['dealer_cards']
            
            # Enregistrer la décision
            if len(episode_player_cards) > 0 and len(episode_dealer_cards) > 0:
                dealer_final_sum = info.get('dealer_final_sum', 0) if done else 0
                
                reporter.record_decision(
                    state=obs,
                    action=action,
                    reward=reward,
                    player_cards=episode_player_cards,
                    dealer_cards=episode_dealer_cards,
                    dealer_final_sum=dealer_final_sum
                )
            
            episode_return += reward
            obs = next_obs
        
        test_returns.append(episode_return)
        if episode_return > 0:
            test_wins += 1
    
    test_win_rate = test_wins / 10000
    test_avg_return = sum(test_returns) / 10000
    
    print(f"\n✅ Entraînement terminé en {train_time:.1f}s")
    print(f"   Win Rate: {test_win_rate*100:.1f}%")
    print(f"   Avg Return: {test_avg_return:.4f}")
    
    # Sauvegarder le modèle
    model_path = Path(f"data/models/optimized/{agent_name}_optimized.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(model_path)
    print(f"   Modèle: {model_path}")
    
    # Générer les plots (simplifié - pas de plots pour le moment)
    # Les plots peuvent être générés séparément si nécessaire
    
    # Générer le rapport
    reports_dir = Path("data/reports/optimized")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Sauvegarder les décisions enregistrées en JSON
    report_data = {
        'agent_type': agent_name,
        'decisions': reporter.decisions if hasattr(reporter, 'decisions') else []
    }
    
    with open(reports_dir / f"{agent_name}_report.json", 'w') as f:
        json.dump(report_data, f, indent=2)
    
    # Résumé
    summary = {
        'agent_type': agent_name,
        'config': config,
        'train_episodes': num_episodes,
        'train_time': train_time,
        'train_win_rate': wins / num_episodes,
        'test_win_rate': test_win_rate,
        'test_avg_return': test_avg_return,
        'timestamp': datetime.now().isoformat(),
    }
    
    summary_path = Path(f"data/logs/optimized/{agent_name}_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary


def main():
    """Réentraîne les agents avec les hyperparamètres optimisés."""
    print("\n" + "="*70)
    print("🚀 RÉENTRAÎNEMENT AVEC HYPERPARAMÈTRES OPTIMISÉS")
    print("="*70)
    
    # Créer les dossiers nécessaires
    Path("data/logs").mkdir(parents=True, exist_ok=True)
    Path("data/models/optimized").mkdir(parents=True, exist_ok=True)
    Path("data/logs/optimized").mkdir(parents=True, exist_ok=True)
    
    # Charger les résultats d'optimisation
    opt_path = Path("data/logs/hyperparameter_optimization.json")
    
    if not opt_path.exists():
        print("❌ Fichier d'optimisation non trouvé!")
        print("   Lancez d'abord: python scripts/optimize_hyperparams.py")
        return
    
    with open(opt_path, 'r') as f:
        optimization = json.load(f)
    
    results = {}
    
    # Q-Learning
    if 'qlearning' in optimization and optimization['qlearning']:
        best_ql_config = optimization['qlearning'][0]['config']
        results['qlearning'] = train_agent(
            QLearningAgent, 
            best_ql_config, 
            'qlearning',
            num_episodes=250000
        )
    
    # SARSA
    if 'sarsa' in optimization and optimization['sarsa']:
        best_sarsa_config = optimization['sarsa'][0]['config']
        results['sarsa'] = train_agent(
            SARSAAgent, 
            best_sarsa_config, 
            'sarsa',
            num_episodes=250000
        )
    
    # DQN
    if 'dqn' in optimization and optimization['dqn']:
        best_dqn_config = optimization['dqn'][0]['config']
        results['dqn'] = train_agent(
            DQNAgent, 
            best_dqn_config, 
            'dqn',
            num_episodes=250000
        )
    
    # Résumé final
    print("\n" + "="*70)
    print("📊 RÉSUMÉ FINAL - AGENTS OPTIMISÉS")
    print("="*70)
    
    # Sauvegarder les résultats de réentraînement
    results_path = Path("data/logs/retraining_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Résultats sauvegardés: {results_path}")
    
    # Comparer avec les résultats originaux
    original_summaries = {}
    for summary_file in Path("data/logs").glob("*_training_summary.json"):
        with open(summary_file, 'r') as f:
            data = json.load(f)
            agent_name = summary_file.stem.replace("_training_summary", "")
            original_summaries[agent_name] = data
    
    for agent_name, result in results.items():
        print(f"\n🎯 {agent_name.upper()}")
        print(f"   Optimisé: {result['test_win_rate']*100:.1f}% win rate, {result['test_avg_return']:.4f} avg return")
        
        if agent_name in original_summaries:
            orig = original_summaries[agent_name]
            orig_wr = orig.get('test_win_rate', 0)
            orig_ar = orig.get('test_avg_return', 0)
            
            print(f"   Original: {orig_wr*100:.1f}% win rate, {orig_ar:.4f} avg return")
            
            wr_diff = (result['test_win_rate'] - orig_wr) * 100
            ar_diff = result['test_avg_return'] - orig_ar
            
            print(f"   Gain:     {wr_diff:+.1f} points, {ar_diff:+.4f} return")
            
            if result['test_win_rate'] > orig_wr:
                print(f"   ✅ Amélioration!")
            elif result['test_win_rate'] == orig_wr:
                print(f"   → Performance égale")
            else:
                print(f"   ⚠️  Légère régression")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
