Blackjack RL Agents Documentation
===================================

.. image:: https://img.shields.io/badge/Python-3.8%2B-blue
   :alt: Python Version
   
.. image:: https://img.shields.io/badge/License-MIT-green
   :alt: License

Bienvenue dans la documentation du projet **Blackjack RL Agents**!

Ce projet implémente et compare **12 agents de Reinforcement Learning** pour jouer au Blackjack,
utilisant différentes approches d'apprentissage (Q-Learning, SARSA, DQN, Monte Carlo) avec 
et sans comptage de cartes Hi-Lo.

**Développé par:** FATEN Saif Eddine  
**Version:** 1.0.0

Objectifs du Projet
-------------------

- Entraîner des agents avec différents algorithmes RL
- Comparer les performances avec et sans card counting
- Optimiser les hyperparamètres pour maximiser le taux de victoire
- Comparer avec la stratégie de base optimale (Basic Strategy)

Résultats Clés
--------------

- **12 agents entraînés** (5 naïfs, 4 counting, 3 optimisés)
- **Meilleur agent:** SARSA Optimisé avec 43.5% de taux de victoire
- **Performance proche de l'optimal théorique** (42.68%)
- **Interface interactive Streamlit** pour visualisation et jeu

Table des Matières
==================

.. toctree::
   :maxdepth: 2
   :caption: Guide Utilisateur

   installation
   usage
   results

.. toctree::
   :maxdepth: 2
   :caption: Documentation Technique

   architecture
   agents
   environment
   api

.. toctree::
   :maxdepth: 1
   :caption: Ressources

   references
   contributing

Liens Rapides
=============

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
