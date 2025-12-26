Résultats Expérimentaux
=======================

Cette section présente les résultats détaillés de l'entraînement et de l'évaluation des 12 agents.

Vue d'Ensemble
--------------

.. table:: Performances Globales
   :widths: auto

   +----------------------+-------------+--------------+---------------+
   | Agent                | Win Rate(%) | Avg Return   | Type          |
   +======================+=============+==============+===============+
   | Monte Carlo Count    | **46.5**    | 0.012        | Counting      |
   +----------------------+-------------+--------------+---------------+
   | DQN                  | 45.3        | -0.009       | Naïf          |
   +----------------------+-------------+--------------+---------------+
   | Monte Carlo          | 44.7        | -0.023       | Naïf          |
   +----------------------+-------------+--------------+---------------+
   | SARSA Optimized      | **43.5**    | -0.014       | Optimisé      |
   +----------------------+-------------+--------------+---------------+
   | DQN Count            | 43.5        | -0.024       | Counting      |
   +----------------------+-------------+--------------+---------------+
   | Qlearning Count      | 42.5        | -0.045       | Counting      |
   +----------------------+-------------+--------------+---------------+
   | **Stratégie de Base**| **42.68**   | **-0.005**   | **Optimal**   |
   +----------------------+-------------+--------------+---------------+

Graphiques Comparatifs
----------------------

.. figure:: ../data/plots/win_rates_comparison.png
   :width: 80%
   :align: center
   
   Comparaison des taux de victoire par agent

.. figure:: ../data/plots/category_comparison.png
   :width: 70%
   :align: center
   
   Distribution des performances par catégorie

Analyse par Algorithme
-----------------------

Q-Learning
~~~~~~~~~~

**Baseline**: 39.3% win rate

**Optimized**: 41.4% win rate (+2.1%)

**Meilleurs hyperparamètres**:
- α = 0.2
- γ = 1.0
- ε_min = 0.1

SARSA
~~~~~

**Baseline**: 42.4% win rate

**Optimized**: 43.5% win rate (+1.1%)

**Meilleurs hyperparamètres**:
- α = 0.05
- γ = 1.0
- ε_min = 0.1

**Accord avec Basic Strategy**: 92.3%

Monte Carlo
~~~~~~~~~~~

**Baseline**: 44.7% win rate

**With Counting**: 46.5% win rate (+1.8%)

Meilleur agent global du projet!

DQN
~~~

**Baseline**: 45.3% win rate

**Optimized**: 42.6% win rate (-2.7%)

Architecture:
- Hidden layers: 2 × 128 neurons
- Learning rate: 0.001
- Batch size: 64

Comparaison Naïf vs Counting
-----------------------------

.. table:: Impact du Comptage de Cartes
   :widths: auto

   +----------------+-------------+-----------------+-------------+
   | Algorithme     | Naïf (%)    | Counting (%)    | Différence  |
   +================+=============+=================+=============+
   | Q-Learning     | 39.3        | 42.5            | +3.2        |
   +----------------+-------------+-----------------+-------------+
   | SARSA          | 42.4        | 42.0            | -0.4        |
   +----------------+-------------+-----------------+-------------+
   | Monte Carlo    | 44.7        | 46.5            | +1.8        |
   +----------------+-------------+-----------------+-------------+
   | DQN            | 45.3        | 43.5            | -1.8        |
   +----------------+-------------+-----------------+-------------+

**Observations**:
- Amélioration moyenne: +0.7%
- Monte Carlo bénéficie le plus du counting
- DQN/SARSA pénalisés par l'espace d'états élargi

Optimisation des Hyperparamètres
---------------------------------

Grid Search
~~~~~~~~~~~

**234 configurations testées** en 1h44min

**Résultats par algorithme**:

Q-Learning (108 configs):
  - Meilleure config: α=0.2, γ=1.0, ε_min=0.1 → **41.4%**
  - Pire config: α=0.3, γ=0.95, ε_min=0.2 → 37.2%
  - Écart: 4.2 points

SARSA (108 configs):
  - Meilleure config: α=0.05, γ=1.0, ε_min=0.1 → **43.5%**
  - Pire config: α=0.3, γ=0.95, ε_min=0.05 → 40.1%
  - Écart: 3.4 points

DQN (18 configs):
  - Meilleure config: lr=0.001, hidden=128, batch=64 → **42.6%**
  - Pire config: lr=0.01, hidden=64, batch=32 → 38.9%
  - Écart: 3.7 points

Insights
~~~~~~~~

1. **γ = 1.0 est optimal** pour tous les algorithmes
   - Épisodes courts (3-10 étapes)
   - Récompense finale la plus importante

2. **ε_min élevé (0.1-0.2) améliore les performances**
   - Maintient l'exploration
   - Évite la convergence prématurée

3. **Learning rate faible (α=0.05-0.1) pour SARSA**
   - Convergence plus stable
   - Moins de variance

Comparaison avec Stratégie de Base
-----------------------------------

La stratégie de base est la politique optimale mathématiquement prouvée:

**Win rate théorique**: 42.68%

**Accord des agents**:

.. table:: Accord avec Basic Strategy
   :widths: auto

   +-------------------+-----------+----------+
   | Agent             | WR (%)    | Accord(%)| 
   +===================+===========+==========+
   | SARSA Optimized   | 43.5      | 92.3     |
   +-------------------+-----------+----------+
   | Q-Learning Opt    | 41.4      | 89.7     |
   +-------------------+-----------+----------+
   | DQN               | 45.3      | 87.2     |
   +-------------------+-----------+----------+

**Analyse**:
- SARSA Optimized dépasse légèrement (+0.82%) la stratégie de base
- Accord élevé (92.3%) confirme l'apprentissage correct
- Possible variance statistique sur les 10k épisodes de test

Temps d'Entraînement
---------------------

.. table:: Durée d'Entraînement
   :widths: auto

   +----------------+------------+-------------+
   | Agent          | Épisodes   | Temps (min) |
   +================+============+=============+
   | Q-Learning     | 250,000    | 15          |
   +----------------+------------+-------------+
   | SARSA          | 250,000    | 16          |
   +----------------+------------+-------------+
   | Monte Carlo    | 200,000    | 18          |
   +----------------+------------+-------------+
   | DQN            | 300,000    | 35          |
   +----------------+------------+-------------+

**Observations**:
- Méthodes tabulaires: ~15-20 min
- DQN: 2× plus lent (réseaux de neurones)
- Total pour 12 agents: ~6 heures

Reproductibilité
----------------

Pour reproduire les résultats:

.. code-block:: bash

   # Entraîner tous les agents naïfs
   python scripts/train_naive.py
   
   # Entraîner les agents counting
   python scripts/train_counting.py
   
   # Optimisation des hyperparamètres
   python scripts/optimize_hyperparams.py
   
   # Réentraînement avec hyperparamètres optimaux
   python scripts/retrain_optimized.py
   
   # Comparaison avec Basic Strategy
   python scripts/compare_with_basic_strategy.py

**Note**: Les résultats peuvent varier légèrement (±1%) en raison de l'aléatoire.
