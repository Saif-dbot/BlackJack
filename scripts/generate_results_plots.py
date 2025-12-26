"""Script pour générer les graphiques des résultats pour le rapport et la présentation."""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import numpy as np

# Configuration matplotlib pour LaTeX
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'text.usetex': False,  # Désactiver pour compatibilité
    'figure.figsize': (10, 6),
    'figure.dpi': 300
})

def load_results():
    """Charge tous les résultats d'entraînement."""
    logs_dir = Path("data/logs")
    results = {}
    
    # Charger les résultats d'entraînement
    for summary_file in logs_dir.glob("*_training_summary.json"):
        agent_name = summary_file.stem.replace("_training_summary", "")
        with open(summary_file, 'r') as f:
            results[agent_name] = json.load(f)
    
    return results

def plot_win_rates(results, output_dir):
    """Graphique des win rates par agent."""
    agents = []
    win_rates = []
    agent_types = []
    
    for name, data in results.items():
        agents.append(name.replace('_', ' ').title())
        win_rates.append(data.get('test_win_rate', 0) * 100)
        
        if 'count' in name:
            agent_types.append('Counting')
        elif 'optimized' in name:
            agent_types.append('Optimisé')
        else:
            agent_types.append('Naïf')
    
    # Créer le DataFrame
    df = pd.DataFrame({
        'Agent': agents,
        'Win Rate (%)': win_rates,
        'Type': agent_types
    })
    df = df.sort_values('Win Rate (%)', ascending=False)
    
    # Graphique
    plt.figure(figsize=(12, 6))
    colors = {'Naïf': '#3498db', 'Counting': '#e74c3c', 'Optimisé': '#2ecc71'}
    bars = plt.bar(range(len(df)), df['Win Rate (%)'], 
                   color=[colors[t] for t in df['Type']])
    
    plt.xlabel('Agents', fontsize=14, fontweight='bold')
    plt.ylabel('Taux de Victoire (%)', fontsize=14, fontweight='bold')
    plt.title('Comparaison des Taux de Victoire par Agent', fontsize=16, fontweight='bold')
    plt.xticks(range(len(df)), df['Agent'].tolist(), rotation=45, ha='right')
    plt.ylim(0, 50)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Légende
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[t], label=t) for t in colors.keys()]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Ajouter les valeurs sur les barres
    for i, v in enumerate(df['Win Rate (%)']):
        plt.text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'win_rates_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Graphique des win rates sauvegardé")

def plot_returns(results, output_dir):
    """Graphique des retours moyens."""
    agents = []
    returns = []
    
    for name, data in results.items():
        agents.append(name.replace('_', ' ').title())
        returns.append(data.get('test_avg_return', 0))
    
    df = pd.DataFrame({
        'Agent': agents,
        'Return': returns
    })
    df = df.sort_values('Return', ascending=False)
    
    plt.figure(figsize=(12, 6))
    colors = ['#2ecc71' if r >= 0 else '#e74c3c' for r in df['Return']]
    plt.bar(range(len(df)), df['Return'], color=colors)
    
    plt.xlabel('Agents', fontsize=14, fontweight='bold')
    plt.ylabel('Retour Moyen', fontsize=14, fontweight='bold')
    plt.title('Comparaison des Retours Moyens par Agent', fontsize=16, fontweight='bold')
    plt.xticks(range(len(df)), df['Agent'].tolist(), rotation=45, ha='right')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    for i, v in enumerate(df['Return']):
        plt.text(i, v + 0.005 if v >= 0 else v - 0.005, 
                f'{v:.4f}', ha='center', 
                va='bottom' if v >= 0 else 'top', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'returns_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Graphique des retours sauvegardé")

def plot_comparison_table(results, output_dir):
    """Tableau de comparaison détaillé."""
    data = []
    for name, res in results.items():
        data.append({
            'Agent': name.replace('_', ' ').title(),
            'Win Rate (%)': f"{res.get('test_win_rate', 0)*100:.2f}",
            'Avg Return': f"{res.get('test_avg_return', 0):.4f}",
            'Episodes': res.get('num_episodes', 'N/A')
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values('Win Rate (%)', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values.tolist(), colLabels=df.columns.tolist(), 
                     cellLoc='center', loc='center',
                     colWidths=[0.4, 0.2, 0.2, 0.2])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style rows
    for i in range(1, len(df) + 1):
        if i % 2 == 0:
            for j in range(len(df.columns)):
                table[(i, j)].set_facecolor('#ecf0f1')
    
    plt.title('Tableau Récapitulatif des Performances', 
              fontsize=16, fontweight='bold', pad=20)
    plt.savefig(output_dir / 'results_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Tableau des résultats sauvegardé")

def plot_best_agents(results, output_dir):
    """Top 5 meilleurs agents."""
    sorted_results = sorted(results.items(), 
                           key=lambda x: x[1].get('test_win_rate', 0), 
                           reverse=True)[:5]
    
    agents = [name.replace('_', ' ').title() for name, _ in sorted_results]
    win_rates = [data.get('test_win_rate', 0) * 100 for _, data in sorted_results]
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.get_cmap('viridis')(np.linspace(0.3, 0.9, len(agents)))
    bars = plt.barh(agents, win_rates, color=colors)
    
    plt.xlabel('Taux de Victoire (%)', fontsize=14, fontweight='bold')
    plt.title('Top 5 des Meilleurs Agents', fontsize=16, fontweight='bold')
    plt.xlim(0, 50)
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    
    for i, (bar, rate) in enumerate(zip(bars, win_rates)):
        plt.text(rate + 0.5, i, f'{rate:.2f}%', 
                va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'top5_agents.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Graphique Top 5 sauvegardé")

def plot_by_category(results, output_dir):
    """Comparaison par catégorie d'agents."""
    categories = {'Naïf': [], 'Counting': [], 'Optimisé': []}
    
    for name, data in results.items():
        wr = data.get('test_win_rate', 0) * 100
        if 'optimized' in name:
            categories['Optimisé'].append(wr)
        elif 'count' in name:
            categories['Counting'].append(wr)
        else:
            categories['Naïf'].append(wr)
    
    # Box plot
    plt.figure(figsize=(10, 6))
    box_data = [rates for rates in categories.values() if rates]
    box_labels = [cat for cat, rates in categories.items() if rates]
    
    bp = plt.boxplot(box_data, patch_artist=True,
                     notch=True, showmeans=True)
    plt.xticks(range(1, len(box_labels) + 1), box_labels)
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.ylabel('Taux de Victoire (%)', fontsize=14, fontweight='bold')
    plt.title('Distribution des Performances par Catégorie', 
              fontsize=16, fontweight='bold')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'category_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Graphique par catégorie sauvegardé")

def main():
    """Génère tous les graphiques."""
    print("Génération des graphiques pour le rapport et la présentation...")
    
    output_dir = Path("data/plots")
    output_dir.mkdir(exist_ok=True)
    
    results = load_results()
    
    if not results:
        print("❌ Aucun résultat trouvé dans data/logs/")
        return
    
    print(f"\n📊 {len(results)} agents trouvés\n")
    
    # Générer tous les graphiques
    plot_win_rates(results, output_dir)
    plot_returns(results, output_dir)
    plot_comparison_table(results, output_dir)
    plot_best_agents(results, output_dir)
    plot_by_category(results, output_dir)
    
    print(f"\n✅ Tous les graphiques générés dans {output_dir}/")
    print(f"   - win_rates_comparison.png")
    print(f"   - returns_comparison.png")
    print(f"   - results_table.png")
    print(f"   - top5_agents.png")
    print(f"   - category_comparison.png")

if __name__ == "__main__":
    main()
