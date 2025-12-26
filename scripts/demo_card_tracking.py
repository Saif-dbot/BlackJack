"""Démonstration des nouvelles fonctionnalités de tracking des cartes."""

import json
from pathlib import Path


def main():
    """Démonstration des rapports améliorés."""
    
    print("\n" + "="*70)
    print("🎴 DÉMONSTRATION - TRACKING DES CARTES DANS LES RAPPORTS")
    print("="*70 + "\n")
    
    # Chemins des rapports
    reports_dir = Path("data/reports")
    
    # Trouver tous les rapports
    reports = list(reports_dir.glob("*.json"))
    
    if not reports:
        print("❌ Aucun rapport trouvé dans data/reports/")
        print("\n💡 Conseil: Entraînez un agent d'abord:")
        print("   python scripts/train_naive_enhanced.py --config config/agents_naive/qlearning_test.yaml")
        return
    
    print(f"📁 {len(reports)} rapport(s) trouvé(s):\n")
    
    for report_path in sorted(reports):
        print(f"\n{'='*70}")
        print(f"📊 Rapport: {report_path.name}")
        print(f"{'='*70}")
        
        # Charger le rapport
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        summary = report['summary']
        
        # Afficher les informations de base
        print(f"\n📈 Informations Générales:")
        print(f"  • Agent: {summary['agent_name']}")
        print(f"  • Décisions totales: {summary['total_decisions']}")
        print(f"  • Récompense moyenne: {summary['avg_reward_per_decision']:.4f}")
        print(f"  • HIT ratio: {summary['hit_ratio']*100:.1f}%")
        print(f"  • STAND ratio: {summary['stand_ratio']*100:.1f}%")
        
        # Exemples de décisions avec cartes
        all_decisions = report.get('all_decisions', [])
        
        if all_decisions:
            print(f"\n🎴 Nouvelles Fonctionnalités - Exemple de Décisions:")
            
            # Trouver une décision intéressante (avec récompense finale)
            final_decisions = [d for d in all_decisions if d.get('dealer_final_sum') is not None]
            
            if final_decisions:
                decision = final_decisions[0]
                
                print(f"\n  Episode {decision['episode']}, Étape {decision['step']}:")
                print(f"    ┌─ État du Joueur:")
                print(f"    │  • Somme: {decision['player_sum']}")
                print(f"    │  • As utilisable: {'Oui' if decision['usable_ace'] else 'Non'}")
                
                if decision.get('player_cards'):
                    print(f"    │  • Cartes: {decision['player_cards']}")
                
                if decision.get('true_count') is not None:
                    print(f"    │  • True Count: {decision['true_count']:.2f}")
                
                print(f"    ├─ État du Dealer:")
                print(f"    │  • Carte visible: {decision['dealer_visible_card']}")
                
                if decision.get('dealer_cards'):
                    print(f"    │  • Toutes les cartes: {decision['dealer_cards']}")
                
                if decision.get('dealer_final_sum'):
                    print(f"    │  • Somme finale: {decision['dealer_final_sum']}")
                
                print(f"    └─ Décision:")
                print(f"       • Action: {decision['action_name']}")
                print(f"       • Récompense: {decision['reward']:+.1f}")
        
        # Statistiques par carte du dealer
        dealer_dist = summary.get('dealer_distribution', {})
        
        if dealer_dist:
            print(f"\n📊 Distribution par Carte du Dealer (Top 3):")
            
            # Trier par nombre total de décisions
            sorted_cards = sorted(
                dealer_dist.items(),
                key=lambda x: x[1]['total'],
                reverse=True
            )[:3]
            
            for card, stats in sorted_cards:
                hit_pct = (stats['hit'] / stats['total'] * 100) if stats['total'] > 0 else 0
                print(f"  • Carte {card:>2}: {stats['total']:3d} décisions ({hit_pct:5.1f}% HIT)")
        
        # Fichier size
        size_mb = report_path.stat().st_size / (1024 * 1024)
        print(f"\n💾 Taille du rapport: {size_mb:.2f} MB")
    
    # Afficher les commandes utiles
    print(f"\n{'='*70}")
    print("🔧 COMMANDES UTILES")
    print(f"{'='*70}\n")
    
    print("1. Visualiser un rapport de manière lisible:")
    print("   python scripts/view_report.py data/reports/qlearning_report.json --show-stats")
    
    print("\n2. Afficher les décisions d'un épisode spécifique:")
    print("   python scripts/view_report.py data/reports/qlearning_report.json --episode 5")
    
    print("\n3. Analyser programmatiquement:")
    print("""   python -c "
import json
report = json.load(open('data/reports/qlearning_report.json'))

# Décisions contre dealer avec 10
decisions_vs_10 = [
    d for d in report['all_decisions'] 
    if d['dealer_visible_card'] == 10
]

print(f'Décisions vs 10: {len(decisions_vs_10)}')
hit_ratio = sum(1 for d in decisions_vs_10 if d['action'] == 1) / len(decisions_vs_10)
print(f'HIT ratio vs 10: {hit_ratio*100:.1f}%')
   "\n""")
    
    print("\n4. Entraîner de nouveaux agents:")
    print("   python scripts/train_naive_enhanced.py --config config/agents_naive/qlearning.yaml")
    print("   python scripts/train_counting_enhanced.py --config config/agents_counting/qlearning_count.yaml")
    
    print(f"\n{'='*70}")
    print("✅ Système de tracking des cartes opérationnel!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
