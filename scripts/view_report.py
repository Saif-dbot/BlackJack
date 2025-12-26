"""Script pour visualiser les rapports JSON des agents de manière lisible."""

import json
import argparse
from pathlib import Path
from typing import Dict, List


def format_decision(decision: Dict, index: int) -> str:
    """Formate une décision de manière lisible.
    
    Args:
        decision: Dictionnaire de la décision
        index: Numéro de la décision
        
    Returns:
        String formaté
    """
    lines = [
        f"\n{'='*70}",
        f"Décision #{index + 1} - Épisode {decision['episode']}, Étape {decision['step']}",
        f"{'='*70}",
        f"État du Joueur:",
        f"  • Somme: {decision['player_sum']}",
        f"  • Cartes approximées: {decision['player_cards']}",
        f"  • As utilisable: {'Oui' if decision['usable_ace'] else 'Non'}",
        f"",
        f"État du Dealer:",
        f"  • Carte visible: {decision['dealer_visible_card']}",
        f"  • Toutes les cartes: {decision['dealer_cards']}",
    ]
    
    if decision['dealer_final_sum'] is not None:
        lines.append(f"  • Somme finale: {decision['dealer_final_sum']}")
    
    lines.extend([
        f"",
        f"Décision de l'Agent:",
        f"  • Action: {decision['action_name']}",
        f"  • Récompense: {decision['reward']:+.1f}",
    ])
    
    if decision['next_player_sum'] is not None:
        lines.append(f"  • Nouvelle somme: {decision['next_player_sum']}")
    
    return "\n".join(lines)


def display_summary(report: Dict) -> None:
    """Affiche le résumé du rapport.
    
    Args:
        report: Rapport complet
    """
    summary = report['summary']
    
    print(f"\n{'='*70}")
    print(f"RÉSUMÉ - Agent: {summary['agent_name']}")
    print(f"{'='*70}")
    print(f"Décisions totales: {summary['total_decisions']}")
    print(f"Récompense totale: {summary['total_reward']:.2f}")
    print(f"Récompense moyenne/décision: {summary['avg_reward_per_decision']:.4f}")
    print(f"")
    print(f"Distribution des Actions:")
    print(f"  • HIT:   {summary['hit_count']:4d} ({summary['hit_ratio']*100:.1f}%)")
    print(f"  • STAND: {summary['stand_count']:4d} ({summary['stand_ratio']*100:.1f}%)")
    print(f"{'='*70}\n")


def display_dealer_stats(report: Dict) -> None:
    """Affiche les statistiques par carte du dealer.
    
    Args:
        report: Rapport complet
    """
    dealer_dist = report['summary'].get('dealer_distribution', {})
    
    if not dealer_dist:
        return
    
    print(f"\n{'='*70}")
    print(f"DISTRIBUTION PAR CARTE DU DEALER")
    print(f"{'='*70}")
    print(f"{'Carte':<10} {'Total':<10} {'HIT':<10} {'STAND':<10} {'HIT %':<10}")
    print(f"{'-'*70}")
    
    for card in sorted(dealer_dist.keys(), key=int):
        stats = dealer_dist[card]
        total = stats['total']
        hit = stats['hit']
        stand = stats['stand']
        hit_pct = (hit / total * 100) if total > 0 else 0
        
        print(f"{card:<10} {total:<10} {hit:<10} {stand:<10} {hit_pct:<10.1f}")
    
    print(f"{'='*70}\n")


def display_player_sum_stats(report: Dict) -> None:
    """Affiche les statistiques par somme du joueur.
    
    Args:
        report: Rapport complet
    """
    sum_dist = report['summary'].get('sum_distribution', {})
    
    if not sum_dist:
        return
    
    print(f"\n{'='*70}")
    print(f"DISTRIBUTION PAR SOMME DU JOUEUR")
    print(f"{'='*70}")
    print(f"{'Somme':<10} {'Total':<10} {'HIT':<10} {'STAND':<10} {'HIT %':<10}")
    print(f"{'-'*70}")
    
    for player_sum in sorted(sum_dist.keys(), key=int):
        stats = sum_dist[player_sum]
        total = stats['total']
        hit = stats['hit']
        stand = stats['stand']
        hit_pct = (hit / total * 100) if total > 0 else 0
        
        print(f"{player_sum:<10} {total:<10} {hit:<10} {stand:<10} {hit_pct:<10.1f}")
    
    print(f"{'='*70}\n")


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Visualiser les rapports JSON des agents"
    )
    parser.add_argument(
        "report_path",
        type=str,
        help="Chemin vers le fichier rapport JSON (ex: data/reports/qlearning_report.json)"
    )
    parser.add_argument(
        "--num-decisions",
        type=int,
        default=5,
        help="Nombre de décisions à afficher (défaut: 5)"
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="Afficher uniquement les décisions d'un épisode spécifique"
    )
    parser.add_argument(
        "--show-stats",
        action="store_true",
        help="Afficher les statistiques détaillées"
    )
    
    args = parser.parse_args()
    
    # Charger le rapport
    report_path = Path(args.report_path)
    if not report_path.exists():
        print(f"❌ Erreur: Le fichier {report_path} n'existe pas")
        return
    
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    # Afficher le résumé
    display_summary(report)
    
    # Afficher les statistiques si demandé
    if args.show_stats:
        display_dealer_stats(report)
        display_player_sum_stats(report)
    
    # Filtrer les décisions par épisode si spécifié
    all_decisions = report.get('all_decisions', [])
    
    if args.episode is not None:
        decisions = [d for d in all_decisions if d['episode'] == args.episode]
        print(f"\n📋 Affichage des décisions de l'épisode {args.episode}")
    else:
        decisions = all_decisions[:args.num_decisions]
        print(f"\n📋 Affichage des {min(args.num_decisions, len(all_decisions))} premières décisions")
    
    # Afficher les décisions
    for i, decision in enumerate(decisions):
        print(format_decision(decision, i))
    
    # Afficher un résumé final
    print(f"\n{'='*70}")
    print(f"Total de décisions dans le rapport: {len(all_decisions)}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
