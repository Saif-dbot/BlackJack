"""SARSA agent with card counting."""

import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..base import AgentBase


class SARSACountAgent(AgentBase):
    """SARSA agent with Hi-Lo card counting.
    
    On-policy TD control with true count observation.
    State space: (player_sum, dealer_card, usable_ace, true_count_bin)
    
    Attributes:
        q_table: Q-value table (state -> action -> value)
        epsilon: Exploration rate
        alpha: Learning rate
        gamma: Discount factor
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize SARSA agent with counting.
        
        Args:
            config: Configuration with alpha, gamma, epsilon_start, epsilon_min, epsilon_decay
        """
        super().__init__(config)
        
        # Hyperparameters
        self.alpha: float = self.config.get("alpha", 0.1)
        self.gamma: float = self.config.get("gamma", 1.0)
        self.epsilon: float = self.config.get("epsilon_start", 1.0)
        self.epsilon_min: float = self.config.get("epsilon_min", 0.05)
        self.epsilon_decay: float = self.config.get("epsilon_decay", 0.99995)
        
        # Q-table with extended state (includes count)
        self.q_table: Dict[Tuple, np.ndarray] = defaultdict(lambda: np.zeros(self.action_dim))
        
    def act(self, state: Tuple, explore: bool = True) -> int:
        """Select action using epsilon-greedy policy.
        
        Args:
            state: Current state (player_sum, dealer_card, usable_ace, true_count_bin)
            explore: Whether to use exploration
            
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
        **kwargs: Any
    ) -> Optional[float]:
        """Update Q-value using SARSA with card counting.
        
        Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]
        Uses actual next action (not max), making it on-policy.
        
        Args:
            state: Current state with count
            action: Action taken
            reward: Reward received
            next_state: Next state with count
            done: Whether episode is done
            **kwargs: Must contain 'next_action' for SARSA
            
        Returns:
            None (no loss for tabular methods)
        """
        # Current Q-value
        current_q = self.q_table[state][action]
        
        # Target Q-value (uses actual next action)
        if done or next_state is None:
            target_q = reward
        else:
            next_action = kwargs.get("next_action")
            if next_action is None:
                raise ValueError("SARSA requires next_action in kwargs")
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
        self.alpha = self.config.get("alpha", 0.1)
        self.gamma = self.config.get("gamma", 1.0)
        self.epsilon_min = self.config.get("epsilon_min", 0.05)
        self.epsilon_decay = self.config.get("epsilon_decay", 0.99995)
