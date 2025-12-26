"""Application Streamlit interactive pour visualiser et comparer les agents de Blackjack."""

import streamlit as st
import json
import pickle
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ajouter le chemin du projet
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.blackjack_env import BlackjackEnv
from src.environment.deck_config import DeckConfig
from src.agents.naive.qlearning import QLearningAgent
from src.agents.naive.sarsa import SARSAAgent
from src.agents.naive.dqn import DQNAgent
from src.agents.naive.monte_carlo import MonteCarloAgent
from src.agents.naive.double_dqn import DoubleDQNAgent
from src.agents.counting.qlearning_count import QLearningCountAgent
from src.agents.counting.sarsa_count import SARSACountAgent
from src.agents.counting.monte_carlo_count import MonteCarloCountAgent

# Configuration de la page
st.set_page_config(
    page_title="Blackjack RL Agents",
    page_icon="♠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .agent-name {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_agent_summaries():
    """Charge tous les résumés d'entraînement des agents."""
    data_dir = Path("data/logs")
    summaries = {}
    
    for summary_file in data_dir.glob("*_training_summary.json"):
        agent_name = summary_file.stem.replace("_training_summary", "")
        with open(summary_file, 'r') as f:
            summaries[agent_name] = json.load(f)
    
    return summaries


@st.cache_data
def load_reports():
    """Charge tous les rapports de décisions."""
    reports_dir = Path("data/reports")
    reports = {}
    
    for report_file in reports_dir.glob("*_report.json"):
        agent_name = report_file.stem.replace("_report", "")
        with open(report_file, 'r') as f:
            reports[agent_name] = json.load(f)
    
    return reports


@st.cache_resource
def load_agent_model(agent_type, model_path):
    """Charge un modèle d'agent."""
    agent_classes = {
        'qlearning': QLearningAgent,
        'sarsa': SARSAAgent,
        'dqn': DQNAgent,
        'monte_carlo': MonteCarloAgent,
        'qlearning_count': QLearningCountAgent,
        'sarsa_count': SARSACountAgent,
        'monte_carlo_count': MonteCarloCountAgent
    }
    
    if agent_type not in agent_classes:
        return None
    
    # Pour les agents DQN, essayer de charger avec torch
    if agent_type in ['dqn', 'double_dqn']:
        try:
            import torch
            data = torch.load(model_path, map_location='cpu')
            if data is not None:
                st.success("[OK] Chargé avec torch.load")
        except Exception as e:
            data = None
            st.warning(f"torch.load échoué: {e}")
    else:
        data = None
    
    # Méthodes de chargement multiples avec réouverture du fichier
    errors = []
    
    # Méthode 1: Pickle standard
    if data is None:
        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            errors.append(f"Standard: {e}")
    
    # Méthode 2: Encoding latin1
    if data is None:
        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
        except Exception as e:
            errors.append(f"Latin1: {e}")
    
    # Méthode 3: Encoding bytes
    if data is None:
        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f, encoding='bytes')
        except Exception as e:
            errors.append(f"Bytes: {e}")
    
    # Méthode 4: Unpickler avec gestion custom
    if data is None:
        try:
            with open(model_path, 'rb') as f:
                unpickler = pickle.Unpickler(f, encoding='latin1')
                data = unpickler.load()
        except Exception as e:
            errors.append(f"Unpickler: {e}")
    
    if data is None:
        st.error(f"[ERREUR] Impossible de charger le fichier avec toutes les méthodes ({len(errors)} tentatives)")
        with st.expander("Voir les détails des erreurs"):
            for i, err in enumerate(errors, 1):
                st.code(f"{i}. {err}")
        st.info("Note: Le fichier pickle est peut-être corrompu ou utilise un ancien format. Réentraînement recommandé.")
        return None
    
    try:
        # Si c'est un dictionnaire, recréer l'agent
        if isinstance(data, dict):
            agent_class = agent_classes[agent_type]
            agent = agent_class(data['config'])
            
            # Restaurer l'état de l'agent
            if agent_type in ['qlearning', 'sarsa', 'qlearning_count', 'sarsa_count']:
                agent.q_table = data['q_table']
                agent.epsilon = data.get('epsilon', 0.0)
            elif agent_type in ['monte_carlo', 'monte_carlo_count']:
                agent.returns = data.get('returns', {})
                agent.q_table = data.get('q_table', {})
                agent.epsilon = data.get('epsilon', 0.0)
            elif agent_type in ['dqn']:
                if 'q_network_state' in data:
                    agent.q_network.load_state_dict(data['q_network_state'])
                    agent.q_network.eval()  # Mode évaluation
                if 'target_network_state' in data:
                    agent.target_network.load_state_dict(data['target_network_state'])
                    agent.target_network.eval()  # Mode évaluation
                agent.epsilon = 0.0  # Pas d'exploration pour l'interface
            
            return agent
        else:
            # Déjà un objet agent
            return data
            
    except Exception as e:
        st.error(f"Erreur lors de la restauration de l'agent: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None


def create_comparison_chart(summaries):
    """Crée un graphique de comparaison des agents."""
    df = pd.DataFrame([
        {
            'Agent': name.replace('_', ' ').title(),
            'Win Rate (%)': summary.get('test_win_rate', 0) * 100,
            'Avg Return': summary.get('test_avg_return', 0),
            'Type': 'Counting' if 'count' in name else 'Naive'
        }
        for name, summary in summaries.items()
    ])
    
    df = df.sort_values('Win Rate (%)', ascending=False)
    
    fig = px.bar(
        df,
        x='Agent',
        y='Win Rate (%)',
        color='Type',
        title='Comparaison des Win Rates par Agent',
        color_discrete_map={'Naive': '#1f77b4', 'Counting': '#ff7f0e'},
        text='Win Rate (%)'
    )
    
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(
        xaxis_tickangle=-45,
        yaxis_range=[0, 50],
        height=500
    )
    
    return fig


def create_return_chart(summaries):
    """Crée un graphique des retours moyens."""
    df = pd.DataFrame([
        {
            'Agent': name.replace('_', ' ').title(),
            'Avg Return': summary.get('test_avg_return', 0),
            'Type': 'Counting' if 'count' in name else 'Naive'
        }
        for name, summary in summaries.items()
    ])
    
    df = df.sort_values('Avg Return', ascending=False)
    
    fig = px.bar(
        df,
        x='Agent',
        y='Avg Return',
        color='Type',
        title='Comparaison des Retours Moyens',
        color_discrete_map={'Naive': '#2ca02c', 'Counting': '#d62728'},
        text='Avg Return'
    )
    
    fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    fig.update_layout(xaxis_tickangle=-45, height=500)
    
    return fig


def play_game_interactive(agent, agent_name):
    """Interface pour jouer une partie contre un agent."""
    st.subheader(f"Jouer contre {agent_name}")
    
    # Déterminer si c'est un agent de counting
    is_counting_agent = 'count' in agent_name.lower()
    
    if 'game_state' not in st.session_state:
        st.session_state.game_state = None
    
    if 'session_stats' not in st.session_state:
        st.session_state.session_stats = {'wins': 0, 'losses': 0, 'draws': 0, 'hands_played': 0}
    
    if 'current_agent' not in st.session_state:
        st.session_state.current_agent = None
    
    # Réinitialiser la session si l'agent a changé
    if st.session_state.current_agent != agent_name and st.session_state.game_state is not None:
        st.session_state.game_state = None
        st.session_state.session_stats = {'wins': 0, 'losses': 0, 'draws': 0, 'hands_played': 0}
        st.session_state.current_agent = agent_name
        st.rerun()
    
    # Bouton pour démarrer une session
    if st.session_state.game_state is None:
        if st.button("Démarrer Session", key=f"new_session_{agent_name}"):
            # Utiliser deck fini avec card counting pour supporter tous les agents
            deck_config = DeckConfig(deck_type="finite", num_decks=6, natural=True, sab=False)
            enable_counting = is_counting_agent
            env = BlackjackEnv(deck_config=deck_config, enable_counting=enable_counting)
            obs, info = env.reset()
            
            st.session_state.game_state = {
                'env': env,
                'obs': obs,
                'done': False,
                'history': [],
                'is_counting': is_counting_agent
            }
            st.session_state.session_stats = {'wins': 0, 'losses': 0, 'draws': 0, 'hands_played': 0}
            st.session_state.current_agent = agent_name
            st.rerun()
    
    if st.session_state.game_state:
        game = st.session_state.game_state
        
        # Afficher les statistiques de session pour les agents de counting
        if is_counting_agent:
            st.markdown("### Statistiques de Session")
            stats = st.session_state.session_stats
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Mains jouées", stats['hands_played'])
            col2.metric("Victoires", stats['wins'], f"{stats['wins']/max(stats['hands_played'], 1)*100:.1f}%")
            col3.metric("Défaites", stats['losses'])
            col4.metric("Égalités", stats['draws'])
            st.markdown("---")
        
        if not game['done']:
            obs = game['obs']
            
            # Gérer les observations avec et sans card counting
            if len(obs) == 4:
                player_sum, dealer_card, usable_ace, true_count = obs
            else:
                player_sum, dealer_card, usable_ace = obs
                true_count = None
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### Votre Main")
                st.metric("Total", player_sum)
                st.metric("As utilisable", "Oui" if usable_ace else "Non")
            
            with col2:
                st.markdown("### Carte Visible du Dealer")
                st.metric("Carte", dealer_card)
            
            with col3:
                st.markdown("### Informations")
                if true_count is not None:
                    st.metric("True Count", f"{true_count:.1f}")
                else:
                    st.caption("Sans card counting")
                
                try:
                    if hasattr(game['env'], 'deck') and hasattr(game['env'].deck, 'cards_left'):
                        remaining_cards = game['env'].deck.cards_left()
                        st.metric("Cartes restantes", remaining_cards)
                    elif hasattr(game['env'], '_deck') and hasattr(game['env']._deck, 'cards_left'):
                        remaining_cards = game['env']._deck.cards_left()
                        st.metric("Cartes restantes", remaining_cards)
                except (AttributeError, Exception):
                    pass
            
            # Décision de l'agent
            agent_error = None
            agent_action = None
            
            # Préparer l'observation pour l'agent
            if is_counting_agent:
                # Agents counting ont besoin de 4 valeurs (avec true_count)
                if true_count is not None:
                    agent_obs = (player_sum, dealer_card, usable_ace, true_count)
                else:
                    agent_obs = obs  # Utiliser l'observation originale si déjà complète
            else:
                # Agents naïfs n'utilisent que 3 valeurs
                agent_obs = (player_sum, dealer_card, usable_ace)
            
            try:
                agent_action = agent.act(agent_obs, explore=False)
            except (KeyError, RuntimeError, IndexError, AttributeError, Exception) as e:
                agent_error = str(e)[:150]
                # État non appris - utiliser stratégie de base simple
                if player_sum < 17:
                    agent_action = 1
                else:
                    agent_action = 0
            
            # Afficher la décision
            agent_decision = "Tirer (Hit)" if agent_action == 1 else "Rester (Stand)"
            
            if agent_error is None:
                st.info(f"Recommandation de l'agent: **{agent_decision}**")
            else:
                st.warning(f"[AVERTISSEMENT] Agent ne peut pas décider (erreur). Stratégie de base: **{agent_decision}**")
                with st.expander("Détails de l'erreur"):
                    st.code(f"Type: KeyError\nMessage: {agent_error}")
                    st.caption("Note: L'agent n'a pas appris cet état pendant l'entraînement.")
            
            # Debug information
            with st.expander("Informations de débogage"):
                st.write(f"État environnement: {obs}")
                st.write(f"État envoyé à l'agent: {agent_obs}")
                st.write(f"Action retournée: {agent_action}")
                st.write(f"Type d'agent: {type(agent).__name__}")
                st.write(f"Agent counting: {is_counting_agent}")
                if hasattr(agent, 'epsilon'):
                    st.write(f"Epsilon: {agent.epsilon}")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                if st.button("Rester (Stand)", key=f"stand_{agent_name}"):
                    next_obs, reward, terminated, truncated, info = game['env'].step(0)
                    st.session_state.game_state['obs'] = next_obs
                    st.session_state.game_state['done'] = terminated or truncated
                    st.session_state.game_state['last_reward'] = reward
                    st.session_state.game_state['history'].append(('Stand', reward, terminated or truncated))
                    st.rerun()
            
            with col2:
                if st.button("Tirer (Hit)", key=f"hit_{agent_name}"):
                    next_obs, reward, terminated, truncated, info = game['env'].step(1)
                    st.session_state.game_state['obs'] = next_obs
                    st.session_state.game_state['done'] = terminated or truncated
                    st.session_state.game_state['last_reward'] = reward
                    st.session_state.game_state['history'].append(('Hit', reward, terminated or truncated))
                    st.rerun()
            
            with col3:
                if st.button("Quitter Session", key=f"leave_playing_{agent_name}"):
                    st.session_state.game_state = None
                    st.session_state.session_stats = {'wins': 0, 'losses': 0, 'draws': 0, 'hands_played': 0}
                    st.rerun()
        
        else:
            # Main terminée - afficher résultat
            reward = game.get('last_reward', 0)
            
            # Mettre à jour les statistiques
            stats = st.session_state.session_stats
            stats['hands_played'] += 1
            if reward > 0:
                stats['wins'] += 1
                result_msg = "**Vous avez GAGNÉ!**"
                result_type = "success"
            elif reward < 0:
                stats['losses'] += 1
                result_msg = "**Vous avez PERDU!**"
                result_type = "error"
            else:
                stats['draws'] += 1
                result_msg = "**ÉGALITÉ!**"
                result_type = "warning"
            
            # Afficher le résultat
            st.markdown("---")
            if result_type == "success":
                st.success(result_msg)
            elif result_type == "error":
                st.error(result_msg)
            else:
                st.warning(result_msg)
            
            # Pour les agents de counting, continuer automatiquement
            if is_counting_agent:
                import time
                time.sleep(1)  # Petit délai pour voir le résultat
                
                # Relancer une nouvelle main automatiquement
                obs, info = game['env'].reset()
                game['obs'] = obs
                game['done'] = False
                game['last_reward'] = None
                st.rerun()
            else:
                # Pour les agents normaux, attendre le clic
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Nouvelle Main", key=f"new_hand_{agent_name}"):
                        obs, info = game['env'].reset()
                        game['obs'] = obs
                        game['done'] = False
                        game['last_reward'] = None
                        st.rerun()
                
                with col2:
                    if st.button("Quitter Session", key=f"leave_final_{agent_name}"):
                        st.session_state.game_state = None
                        st.session_state.session_stats = {'wins': 0, 'losses': 0, 'draws': 0, 'hands_played': 0}
                        st.rerun()


def show_decision_analysis(reports, agent_name):
    """Affiche l'analyse des décisions d'un agent."""
    if agent_name not in reports:
        st.warning("Rapport non disponible pour cet agent")
        return
    
    report = reports[agent_name]
    decisions = report.get('decisions', [])
    
    if not decisions:
        st.warning("Aucune décision enregistrée")
        return
    
    st.subheader(f"Analyse des Décisions - {agent_name}")
    
    # Statistiques globales
    total_decisions = len(decisions)
    hits = sum(1 for d in decisions if d.get('action') == 1)
    stands = total_decisions - hits
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Décisions", total_decisions)
    col2.metric("Hits", hits, f"{hits/total_decisions*100:.1f}%")
    col3.metric("Stands", stands, f"{stands/total_decisions*100:.1f}%")
    
    # Distribution des décisions par état
    df_decisions = pd.DataFrame([
        {
            'Player Sum': d['state'][0],
            'Dealer Card': d['state'][1],
            'Action': 'Hit' if d['action'] == 1 else 'Stand',
            'Reward': d['reward']
        }
        for d in decisions[:1000]  # Limiter pour la performance
    ])
    
    # Heatmap des décisions
    st.markdown("#### Carte de Décisions (Player Sum vs Dealer Card)")
    
    pivot_table = df_decisions.pivot_table(
        values='Action',
        index='Player Sum',
        columns='Dealer Card',
        aggfunc=lambda x: (x == 'Hit').sum() / len(x)
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_table.values,
        x=pivot_table.columns,
        y=pivot_table.index,
        colorscale='RdYlGn_r',
        colorbar_title="% Hit"
    ))
    
    fig.update_layout(
        title="Probabilité de Tirer (Hit) par État",
        xaxis_title="Carte Visible du Dealer",
        yaxis_title="Somme du Joueur",
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)


def main():
    """Fonction principale de l'application."""
    
    # En-tête
    st.markdown('<h1 class="main-header">Blackjack RL Agents Dashboard</h1>', unsafe_allow_html=True)
    
    # Charger les données
    summaries = load_agent_summaries()
    reports = load_reports()
    
    if not summaries:
        st.error("Aucun agent trouvé! Entraînez d'abord les agents.")
        return
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choisir une page:",
        ["Vue d'ensemble", "Comparaison", "Jouer"]
    )
    
    # Pages
    if page == "Vue d'ensemble":
        st.header("Vue d'ensemble des Agents")
        
        # Section Présentation du Projet
        st.markdown("---")
        st.markdown("### À Propos du Projet")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            **Blackjack RL Agents Dashboard - Version 1.0**
            
            Ce projet implémente et compare **12 agents de Reinforcement Learning** pour jouer au Blackjack.
            Les agents utilisent différentes approches d'apprentissage par renforcement, certains avec 
            le système de comptage de cartes **Hi-Lo**.
            
            **Objectifs:**
            - Entraîner des agents avec différents algorithmes (Q-Learning, SARSA, DQN, Monte Carlo)
            - Comparer les performances avec et sans card counting
            - Optimiser les hyperparamètres pour maximiser le win rate
            - Comparer avec la stratégie de base optimale (Basic Strategy)
            
            **Résultats:**
            - 12 agents entraînés (5 naïfs, 4 counting, 3 optimisés)
            - Meilleur agent: **SARSA Optimized** avec 43.5% de win rate
            - Win rate proche de l'optimal théorique (42.68%)
            """)
            
            st.info("**Développé par:** FATEN Saif Eddine")
        
        with col2:
            st.markdown("### Guide d'Utilisation")
            st.markdown("""
            **Navigation:**
            
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
            - Card counting (agents count)
            """)
        
        st.markdown("---")
        
        # Tableaux des performances
        st.subheader("Performance des Agents")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Agents Naïfs")
            naive_agents = {k: v for k, v in summaries.items() if 'count' not in k}
            df_naive = pd.DataFrame([
                {
                    'Agent': name.replace('_', ' ').title(),
                    'Win Rate': f"{summary.get('test_win_rate', 0)*100:.1f}%",
                    'Avg Return': f"{summary.get('test_avg_return', 0):.4f}",
                    'Episodes': summary.get('num_episodes', 0)
                }
                for name, summary in naive_agents.items()
            ]).sort_values('Win Rate', ascending=False)
            st.dataframe(df_naive, use_container_width=True)
        
        with col2:
            st.markdown("#### Agents avec Card Counting")
            counting_agents = {k: v for k, v in summaries.items() if 'count' in k}
            if counting_agents:
                df_counting = pd.DataFrame([
                    {
                        'Agent': name.replace('_', ' ').title(),
                        'Win Rate': f"{summary.get('test_win_rate', 0)*100:.1f}%",
                        'Avg Return': f"{summary.get('test_avg_return', 0):.4f}",
                        'Episodes': summary.get('num_episodes', 0)
                    }
                    for name, summary in counting_agents.items()
                ]).sort_values('Win Rate', ascending=False)
                st.dataframe(df_counting, use_container_width=True)
            else:
                st.info("Aucun agent avec card counting trouvé")
        
        # Meilleur agent
        best_agent = max(summaries.items(), key=lambda x: x[1].get('test_win_rate', 0))
        st.success(f"**Meilleur Agent**: {best_agent[0].replace('_', ' ').title()} avec {best_agent[1].get('test_win_rate', 0)*100:.1f}% de win rate")
    
    elif page == "Comparaison":
        st.header("Comparaison des Agents")
        
        # Graphiques
        fig1 = create_comparison_chart(summaries)
        st.plotly_chart(fig1, use_container_width=True)
        
        fig2 = create_return_chart(summaries)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Comparaison détaillée
        st.subheader("Comparaison Détaillée")
        selected_agents = st.multiselect(
            "Sélectionner les agents à comparer:",
            list(summaries.keys()),
            default=list(summaries.keys())[:3]
        )
        
        if selected_agents:
            comparison_data = []
            for agent_name in selected_agents:
                summary = summaries[agent_name]
                # Déterminer le type d'agent
                if 'optimized' in agent_name:
                    agent_type = 'Optimisé'
                elif 'count' in agent_name:
                    agent_type = 'Counting'
                else:
                    agent_type = 'Naïf'
                
                comparison_data.append({
                    'Agent': agent_name.replace('_', ' ').title(),
                    'Win Rate (%)': summary.get('test_win_rate', 0) * 100,
                    'Avg Return': summary.get('test_avg_return', 0),
                    'Type': agent_type
                })
            
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison, use_container_width=True)
    
    elif page == "Jouer":
        st.header("Jouer contre un Agent")
        
        # Maintenant tous les agents sont jouables avec deck fini
        if not summaries:
            st.warning("Aucun agent disponible")
            return
        
        # Filtrer les agents indisponibles
        available_agents = [name for name in summaries.keys() 
                           if name not in ['double_dqn', 'dqn_count']]
        
        agent_name = st.selectbox(
            "Choisir un agent:",
            available_agents,
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        if agent_name is None:
            st.warning("Aucun agent disponible")
            return
        
        # Info sur le mode de jeu
        agent_type_display = "avec Card Counting" if 'count' in agent_name.lower() else "Naïf/Optimisé"
        st.info(f"Mode de jeu: Deck fini (6 decks) | Type d'agent: {agent_type_display}")
        
        # Essayer différents chemins possibles
        possible_paths = [
            Path(f"data/models/naive/{agent_name}_final.pkl"),
            Path(f"data/models/counting/{agent_name}_final.pkl"),
            Path(f"data/models/optimized/{agent_name}_optimized.pkl"),
            Path(f"data/models/naive/{agent_name}.pkl"),
            Path(f"data/models/counting/{agent_name}.pkl"),
        ]
        
        model_path = None
        for path in possible_paths:
            if path.exists():
                model_path = path
                break
        
        if model_path:
            agent_type = agent_name.replace('_count', '').replace('_final', '').replace('_optimized', '')
            agent = load_agent_model(agent_type, model_path)
            
            if agent:
                play_game_interactive(agent, agent_name)
            else:
                st.error("Impossible de charger l'agent")
        else:
            st.error(f"Modèle non trouvé pour {agent_name}")
            st.info(f"Chemins cherchés : {[str(p) for p in possible_paths]}")
    

if __name__ == "__main__":
    main()
