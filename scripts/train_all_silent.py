"""Script pour lancer tous les entraînements de manière silencieuse."""

import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_training(config_path: str, agent_name: str):
    """Lance un entraînement de manière silencieuse."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Démarrage: {agent_name}...", end=" ", flush=True)
    
    result = subprocess.run(
        [sys.executable, "scripts/train_naive_enhanced.py", "--config", config_path],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(f"✓ Terminé")
    else:
        print(f"✗ Erreur")
        if result.stderr:
            print(f"  Erreur: {result.stderr[:200]}")
    
    return result.returncode == 0


def run_counting_training(config_path: str, agent_name: str):
    """Lance un entraînement avec comptage de cartes de manière silencieuse."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Démarrage: {agent_name} (counting)...", end=" ", flush=True)
    
    result = subprocess.run(
        [sys.executable, "scripts/train_counting_enhanced.py", "--config", config_path],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(f"✓ Terminé")
    else:
        print(f"✗ Erreur")
        if result.stderr:
            print(f"  Erreur: {result.stderr[:200]}")
    
    return result.returncode == 0


def main():
    """Lance tous les entraînements."""
    print("\n" + "="*70)
    print("ENTRAÎNEMENT DE TOUS LES AGENTS - MODE SILENCIEUX")
    print("="*70 + "\n")
    
    start_time = datetime.now()
    
    # Agents naïfs
    print("📊 AGENTS NAÏFS:")
    print("-" * 70)
    
    naive_agents = [
        ("config/agents_naive/qlearning.yaml", "Q-Learning"),
        ("config/agents_naive/sarsa.yaml", "SARSA"),
        ("config/agents_naive/mc.yaml", "Monte Carlo"),
        ("config/agents_naive/dqn.yaml", "DQN"),
        ("config/agents_naive/double_dqn.yaml", "Double DQN"),
    ]
    
    naive_success = 0
    for config, name in naive_agents:
        if run_training(config, name):
            naive_success += 1
    
    print(f"\nRésultat: {naive_success}/{len(naive_agents)} agents naïfs entraînés\n")
    
    # Agents avec comptage
    print("🎴 AGENTS AVEC COMPTAGE DE CARTES:")
    print("-" * 70)
    
    counting_agents = [
        ("config/agents_counting/qlearning_count.yaml", "Q-Learning"),
        ("config/agents_counting/sarsa_count.yaml", "SARSA"),
        ("config/agents_counting/monte_carlo_count.yaml", "Monte Carlo"),
        ("config/agents_counting/dqn_count.yaml", "DQN"),
    ]
    
    counting_success = 0
    for config, name in counting_agents:
        if run_counting_training(config, name):
            counting_success += 1
    
    print(f"\nRésultat: {counting_success}/{len(counting_agents)} agents avec counting entraînés\n")
    
    # Résumé final
    elapsed = datetime.now() - start_time
    print("="*70)
    print("RÉSUMÉ FINAL")
    print("="*70)
    print(f"✓ Agents naïfs: {naive_success}/{len(naive_agents)}")
    print(f"✓ Agents counting: {counting_success}/{len(counting_agents)}")
    print(f"✓ Total: {naive_success + counting_success}/{len(naive_agents) + len(counting_agents)}")
    print(f"⏱️  Temps total: {elapsed}")
    print("="*70 + "\n")
    
    # Afficher les fichiers générés
    print("📁 FICHIERS GÉNÉRÉS:")
    print("-" * 70)
    
    models_naive = list(Path("data/models/naive").glob("*_final.pkl"))
    models_counting = list(Path("data/models/counting").glob("*_final.pkl"))
    plots = list(Path("data/plots").glob("*.png"))
    reports = list(Path("data/reports").glob("*.json"))
    
    print(f"Modèles naïfs: {len(models_naive)}")
    print(f"Modèles counting: {len(models_counting)}")
    print(f"Plots: {len(plots)}")
    print(f"Rapports: {len(reports)}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
