# Blackjack Reinforcement Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Système complet d'apprentissage par renforcement pour comparer des stratégies naïves et avec comptage de cartes au Blackjack.

## 📊 Résultats Clés

- **12 agents RL implémentés** : Q-Learning, SARSA, Monte Carlo, DQN
- **Meilleur agent** : Monte Carlo Count (46.5% de taux de victoire)
- **Meilleur optimisé** : SARSA (43.5%, proche de la stratégie de base optimale à 42.68%)
- **234 configurations testées** via recherche de grille hyperparamètres
- **Interface Streamlit interactive** pour jouer contre les agents

## 📚 Documentation

- **[Rapport détaillé](rapport.pdf)** : Analyse complète du projet (~40 pages)
- **[Présentation](presentation.pdf)** : Slides de présentation (~25 slides)
- **[Documentation ReadTheDocs](docs/)** : Guide utilisateur et documentation technique
- **[Guide LaTeX](LATEX_README.md)** : Instructions de compilation des documents

## 🎯 Objectif

Comparer deux approches de jeu au Blackjack :
1. **Stratégie Naïve** : Agents apprennent sans information sur le comptage des cartes
2. **Stratégie avec Comptage** : Agents utilisent le système Hi-Lo pour optimiser leurs décisions

## 🚀 Quick Start

### Installation

```bash
# Cloner le projet
git clone https://github.com/votre-username/P3_Blackjack_RL.git
cd P3_Blackjack_RL

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirements.txt

# Installer le package en mode développement
pip install -e .
```

### Lancer l'Interface Streamlit

```bash
streamlit run streamlit_app/main.py
```

L'application offre 3 pages :
- **Vue d'ensemble** : Présentation du projet et résultats
- **Comparaison** : Comparaison détaillée des agents
- **Jouer** : Jouer au Blackjack avec recommandations des agents

### Entraînement d'un Agent

```bash
# Entraîner un agent Q-Learning naïf
python scripts/train_naive.py --config config/agents_naive/qlearning.yaml

# Entraîner un agent SARSA naïf
python scripts/train_naive.py --config config/agents_naive/sarsa.yaml

# Entraîner un agent Monte Carlo naïf
python scripts/train_naive.py --config config/agents_naive/mc.yaml
```

### Générer les Graphiques

```bash
python scripts/generate_results_plots.py
```

### Compiler la Documentation LaTeX

```bash
# Windows PowerShell
.\compile_latex.ps1

# Ou manuellement
pdflatex rapport.tex
pdflatex presentation.tex
```

## 📊 Résultats Attendus

| Agent | Type | Win Rate (Naïf) | Win Rate (Comptage) | Amélioration |
|-------|------|-----------------|---------------------|--------------|
| **Monte Carlo** | Tabular | ≥42% | ≥44% | +2-4% |
| **Q-Learning** | Tabular | ≥42% | ≥45% | +3-5% |
| **SARSA** | Tabular | ≥42% | ≥45% | +3-5% |
| **DQN** | Deep RL | ≥38% | ≥42% | +4-6% |

## 🏗️ Architecture

```
P3_Blackjack_RL/
├── src/
│   ├── environment/       # Environnement Blackjack + comptage cartes
│   ├── agents/
│   │   ├── naive/        # Agents sans comptage (MC, Q-Learning, SARSA, DQN)
│   │   └── counting/     # Agents avec comptage
│   ├── training/         # Pipeline d'entraînement
│   ├── evaluation/       # Évaluation et comparaison
│   └── utils/            # Utilitaires (logging, config)
├── streamlit_app/        # Interface web Streamlit
├── config/               # Configurations YAML
├── data/                 # Modèles entraînés et résultats
├── tests/                # Tests unitaires
└── scripts/              # Scripts d'entraînement CLI
```

## 🎓 Algorithmes Implémentés

### Agents Naïfs

1. **Monte Carlo** : Apprentissage par épisodes complets
   - First-visit MC avec moyennage des returns
   - État : (player_sum, dealer_card, usable_ace)
   
2. **Q-Learning** : TD control off-policy
   - Update : Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
   - Exploration : ε-greedy avec decay
   
3. **SARSA** : TD control on-policy
   - Update : Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]
   - Plus conservateur que Q-Learning
   
4. **DQN** : Deep Q-Network avec replay buffer
   - Neural network pour approximation Q-values
   - Target network pour stabilité

### Système de Comptage Hi-Lo

```python
# Valeurs des cartes
Low cards (2-6):  +1
Neutral (7-9):     0
High cards (10-A): -1

# True Count = Running Count / Decks Remaining
```

## 📈 Interface Streamlit (En développement)

```bash
# Lancer l'interface
streamlit run streamlit_app/app.py
```

**Pages disponibles** :
1. 🏠 **Accueil** : Documentation et guide
2. 🎓 **Training** : Entraînement interactif
3. 📊 **Comparison** : Comparaison agents
4. 🎮 **Simulation** : Jouer contre l'agent
5. 🃏 **Card Counting** : Analyse du comptage
6. 📈 **Dashboard** : Vue d'ensemble

## 🧪 Tests

Le projet maintient une couverture de tests ≥80% :

```bash
# Tests rapides
pytest tests/ -v --tb=short

# Tests avec rapport détaillé
pytest tests/ -v --cov=src --cov-report=term-missing

# Tests d'un module spécifique
pytest tests/test_card_counting.py -v -s
```

## 📚 Configuration

Les agents sont configurés via fichiers YAML :

```yaml
# config/agents_naive/qlearning.yaml
agent:
  type: qlearning
  state_dim: 3

hyperparameters:
  alpha: 0.01
  gamma: 0.99
  epsilon_start: 1.0
  epsilon_decay: 0.9995

training:
  episodes: 250000
  eval_frequency: 5000
```

## 🔬 Développement

### Standards de Code

- **Formatage** : Black (line length 100)
- **Type hints** : mypy --strict
- **Docstrings** : Google format
- **Tests** : pytest avec ≥80% coverage

### Pré-commit

```bash
# Formatter le code
black src/ streamlit_app/ tests/
isort src/ streamlit_app/ tests/

# Linting
flake8 src/ streamlit_app/ tests/ --max-line-length 100

# Type checking
mypy src/ --strict

# Tests
pytest tests/ --cov=src
```

## 📊 Résultats Scientifiques

Le comptage de cartes Hi-Lo améliore significativement la performance :
- **Amélioration moyenne** : +3-5% win rate
- **P-value** : < 0.05 (différence significative)
- **Cohen's d** : > 0.5 (effet modéré)

## 📄 License

MIT License - Voir [LICENSE](LICENSE)

## 🤝 Contributions

Ce projet suit strictement le [PROJECT_GUIDE.md](PROJECT_GUIDE.md) pour toutes les implémentations.

## 📞 Contact

Pour questions ou suggestions, créer une issue sur le repository.

---

**Status** : ✅ Phase 1-2 complètes (Environnement + Agents naïfs fonctionnels)  
**Prochaines étapes** : Agents avec comptage, Interface Streamlit, Évaluation complète
