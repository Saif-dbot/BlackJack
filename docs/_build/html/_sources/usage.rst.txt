Utilisation
===========

Ce guide explique comment utiliser les différentes fonctionnalités du projet.

Interface Streamlit
-------------------

Lancement de l'Application
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   streamlit run streamlit_app/main.py

L'application s'ouvre dans votre navigateur à ``http://localhost:8501``

Pages Disponibles
~~~~~~~~~~~~~~~~~

**Vue d'ensemble**
   - Présentation du projet
   - Tableaux de performances
   - Meilleur agent

**Comparaison**
   - Graphiques interactifs
   - Sélection multi-agents
   - Analyse comparative

**Jouer**
   - Mode jeu interactif
   - Recommandations en temps réel
   - Comptage de cartes (agents counting)

Entraînement des Agents
------------------------

Agents Naïfs
~~~~~~~~~~~~

Entraîner tous les agents naïfs:

.. code-block:: bash

   python scripts/train_naive.py

Avec configuration personnalisée:

.. code-block:: bash

   python scripts/train_naive.py --agent qlearning --episodes 100000

Agents avec Counting
~~~~~~~~~~~~~~~~~~~~

Entraîner les agents avec comptage de cartes:

.. code-block:: bash

   python scripts/train_counting.py

Optimisation des Hyperparamètres
---------------------------------

Grid Search
~~~~~~~~~~~

Recherche exhaustive des meilleurs hyperparamètres:

.. code-block:: bash

   python scripts/optimize_hyperparams.py

Résultats sauvegardés dans ``data/logs/hyperparameter_optimization.json``

Réentraînement avec Hyperparamètres Optimaux
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python scripts/retrain_optimized.py

Analyse et Comparaison
-----------------------

Comparer avec Basic Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from scripts.compare_with_basic_strategy import compare_agents
   
   results = compare_agents()
   print(results)

Générer les Graphiques
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python scripts/generate_results_plots.py

Graphiques sauvegardés dans ``data/plots/``

Utilisation Programmatique
---------------------------

Charger un Agent
~~~~~~~~~~~~~~~~

.. code-block:: python

   from src.agents.naive.qlearning import QLearningAgent
   import pickle
   
   # Charger l'agent
   with open('data/models/naive/qlearning_final.pkl', 'rb') as f:
       data = pickle.load(f)
   
   agent = QLearningAgent(data['config'])
   agent.q_table = data['q_table']

Jouer une Partie
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from src.environment.blackjack_env import BlackjackEnv
   from src.environment.deck_config import DeckConfig
   
   # Créer l'environnement
   deck_config = DeckConfig(deck_type="finite", num_decks=6)
   env = BlackjackEnv(deck_config=deck_config)
   
   # Jouer
   obs, info = env.reset()
   done = False
   
   while not done:
       action = agent.act(obs, explore=False)
       obs, reward, terminated, truncated, info = env.step(action)
       done = terminated or truncated
   
   print(f"Reward: {reward}")

Évaluer un Agent
~~~~~~~~~~~~~~~~

.. code-block:: python

   def evaluate_agent(agent, env, episodes=1000):
       wins = 0
       total_return = 0
       
       for _ in range(episodes):
           obs, _ = env.reset()
           done = False
           episode_return = 0
           
           while not done:
               action = agent.act(obs, explore=False)
               obs, reward, terminated, truncated, _ = env.step(action)
               episode_return += reward
               done = terminated or truncated
           
           if episode_return > 0:
               wins += 1
           total_return += episode_return
       
       return {
           'win_rate': wins / episodes,
           'avg_return': total_return / episodes
       }

Configuration
-------------

Fichiers de Configuration YAML
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Les agents utilisent des fichiers de configuration dans ``config/``:

.. code-block:: yaml

   # config/agents_naive/qlearning.yaml
   agent_type: qlearning
   num_episodes: 250000
   alpha: 0.2           # Learning rate
   gamma: 1.0           # Discount factor
   epsilon_start: 1.0   # Initial exploration
   epsilon_min: 0.05    # Minimum exploration
   epsilon_decay: 0.99995

Modifier les Paramètres
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from src.utils.config_loader import load_config
   
   config = load_config('config/agents_naive/qlearning.yaml')
   config['alpha'] = 0.3  # Changer learning rate
   
   agent = QLearningAgent(config)

Exemples Avancés
----------------

Entraînement Personnalisé
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from src.agents.naive.sarsa import SARSAAgent
   from src.environment.blackjack_env import BlackjackEnv
   
   config = {
       'alpha': 0.1,
       'gamma': 0.99,
       'epsilon_start': 1.0,
       'epsilon_min': 0.01,
       'epsilon_decay': 0.9999
   }
   
   agent = SARSAAgent(config)
   env = BlackjackEnv()
   
   # Boucle d'entraînement
   for episode in range(10000):
       obs, _ = env.reset()
       done = False
       
       while not done:
           action = agent.act(obs)
           next_obs, reward, terminated, truncated, _ = env.step(action)
           agent.update(obs, action, reward, next_obs)
           obs = next_obs
           done = terminated or truncated
       
       agent.decay_epsilon()
       
       if (episode + 1) % 1000 == 0:
           print(f"Episode {episode + 1}/10000")

Comparaison Multi-Agents
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   
   agents = {
       'Q-Learning': qlearning_agent,
       'SARSA': sarsa_agent,
       'DQN': dqn_agent
   }
   
   results = []
   for name, agent in agents.items():
       perf = evaluate_agent(agent, env, episodes=1000)
       results.append({
           'Agent': name,
           'Win Rate': perf['win_rate'],
           'Avg Return': perf['avg_return']
       })
   
   df = pd.DataFrame(results)
   print(df)
