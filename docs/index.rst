Blackjack RL Agents Documentation
===================================

.. image:: https://img.shields.io/badge/Python-3.8%2B-blue
   :alt: Python Version

Bienvenue dans la documentation du projet Blackjack RL Agents.

Ce projet a été développé à l'École Nationale des Arts et Métiers (ENSAM) de Meknès dans le cadre d'un projet académique d'apprentissage par renforcement. Il implémente et compare 12 agents de Reinforcement Learning pour jouer au Blackjack, utilisant différentes approches d'apprentissage (Q-Learning, SARSA, DQN, Monte Carlo) avec et sans comptage de cartes Hi-Lo.

Développé par FATEN Saif Eddine
Encadré par M. Tawfik Masrour (Enseignant à l'ENSAM Meknès)
Version 2.0.0

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

Liens Rapides
=============

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
