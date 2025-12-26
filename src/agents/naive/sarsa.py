"""SARSA agent for Blackjack."""

import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..base import AgentBase


class SARSAAgent(AgentBase):
    """SARSA agent with epsilon-greedy exploration.
    
    On-policy TD control algorithm that updates Q-values based on
    the action actually taken (not the best action).
    
    Attributes:
        q_table: Q-value table (state -> action -> value)
        alpha: Learning rate
        gamma: Discount factor
        epsilon: Exploration rate
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize SARSA agent.
        
        Args:
            config: Configuration with alpha, gamma, epsilon parameters
        """
        super().__init__(config)
        
        # Hyperparameters
        self.alpha: float = self.config.get("alpha", 0.01)
        self.gamma: float = self.config.get("gamma", 0.99)
        self.epsilon: float = self.config.get("epsilon_start", 1.0)
        self.epsilon_min: float = self.config.get("epsilon_min", 0.01)
        self.epsilon_decay: float = self.config.get("epsilon_decay", 0.9995)
        
        # Q-table
        self.q_table: Dict[Tuple, np.ndarray] = defaultdict(lambda: np.zeros(self.action_dim))
        
    def act(self, state: Tuple, explore: bool = True) -> int:
        """Select action using epsilon-greedy policy.
        
        Args:
            state: Current state (player_sum, dealer_card, usable_ace)
            explore: Whether to explore
            
        Returns:
            Action (0=STAND, 1=HIT)
        """
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            q_values = self.q_table[state]
            return int(np.argmax(q_values))
    
    def update(
        self,
        state: Tuple,
        action: int,
        reward: float,
        next_state: Optional[Tuple] = None,
        done: bool = False,
        next_action: Optional[int] = None,
        **kwargs: Any
    ) -> Optional[float]:
        """Update Q-value using SARSA update rule.
        
        Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            next_action: Next action (for SARSA)
            
        Returns:
            None (no loss for tabular methods)
        """
        # Current Q-value
        current_q = self.q_table[state][action]
        
        # Target Q-value
        if done or next_state is None:
            target_q = reward
        else:
            # Use next action Q-value (on-policy)
            if next_action is None:
                # If next_action not provided, use greedy action
                next_action = self.act(next_state, explore=True)
            next_q = self.q_table[next_state][next_action]
            target_q = reward + self.gamma * next_q
        
        # SARSA update
        self.q_table[state][action] += self.alpha * (target_q - current_q)
        
        # Update stats
        self.steps_trained += 1
        if done:
            self.episodes_trained += 1
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return None
    
    def save(self, path: Path) -> None:
        """Save agent to disk.
        
        Args:
            path: Path to save agent
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "q_table": dict(self.q_table),
            "epsilon": self.epsilon,
            "config": self.config,
            "episodes_trained": self.episodes_trained,
            "steps_trained": self.steps_trained,
        }
        
        with open(path, "wb") as f:
            pickle.dump(state, f)
            
    def load(self, path: Path) -> None:
        """Load agent from disk.
        
        Args:
            path: Path to load agent from
        """
        with open(path, "rb") as f:
            state = pickle.load(f)
            
        self.q_table = defaultdict(lambda: np.zeros(self.action_dim), state["q_table"])
        self.epsilon = state["epsilon"]
        self.config = state["config"]
        self.episodes_trained = state["episodes_trained"]
        self.steps_trained = state["steps_trained"]
        
        # Restore hyperparameters
        self.alpha = self.config.get("alpha", 0.01)
        self.gamma = self.config.get("gamma", 0.99)
        self.epsilon_min = self.config.get("epsilon_min", 0.01)
        self.epsilon_decay = self.config.get("epsilon_decay", 0.9995)
