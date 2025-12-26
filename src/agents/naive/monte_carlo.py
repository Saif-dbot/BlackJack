"""Monte Carlo agent for Blackjack."""

import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..base import AgentBase


class MonteCarloAgent(AgentBase):
    """First-visit Monte Carlo agent with epsilon-greedy exploration.
    
    Learns Q-values from complete episodes using first-visit MC updates.
    Uses epsilon-greedy policy for exploration.
    
    Attributes:
        q_table: Q-value table (state -> action -> value)
        returns: Returns for each state-action pair
        epsilon: Exploration rate
        gamma: Discount factor
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize Monte Carlo agent.
        
        Args:
            config: Configuration with gamma, epsilon_start, epsilon_min, epsilon_decay
        """
        super().__init__(config)
        
        # Hyperparameters
        self.gamma: float = self.config.get("gamma", 0.99)
        self.epsilon: float = self.config.get("epsilon_start", 1.0)
        self.epsilon_min: float = self.config.get("epsilon_min", 0.01)
        self.epsilon_decay: float = self.config.get("epsilon_decay", 0.9995)
        
        # Q-table and returns tracking
        self.q_table: Dict[Tuple, np.ndarray] = defaultdict(lambda: np.zeros(self.action_dim))
        self.returns: Dict[Tuple[Tuple, int], List[float]] = defaultdict(list)
        self.max_returns: int = 100  # Limit returns memory to prevent slowdown
        
        # Episode buffer
        self.episode_buffer: List[Tuple[Tuple, int, float]] = []
        
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
    
    def store_transition(self, state: Tuple, action: int, reward: float) -> None:
        """Store transition in episode buffer.
        
        Args:
            state: State
            action: Action taken
            reward: Reward received
        """
        self.episode_buffer.append((state, action, reward))
        
    def update_from_episode(self) -> None:
        """Update Q-values from complete episode using first-visit MC."""
        if not self.episode_buffer:
            return
            
        # Calculate returns
        G = 0
        visited_state_actions = set()
        
        # Process episode in reverse
        for t in reversed(range(len(self.episode_buffer))):
            state, action, reward = self.episode_buffer[t]
            G = reward + self.gamma * G
            
            # First-visit MC: only update if this is first occurrence
            if (state, action) not in visited_state_actions:
                visited_state_actions.add((state, action))
                # Keep only recent returns to prevent memory issues
                returns_list = self.returns[(state, action)]
                returns_list.append(G)
                if len(returns_list) > self.max_returns:
                    returns_list.pop(0)
                # Update Q-value as average of returns
                self.q_table[state][action] = np.mean(returns_list)
        
        # Clear episode buffer
        self.episode_buffer = []
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Update stats
        self.episodes_trained += 1
        
    def update(
        self,
        state: Tuple,
        action: int,
        reward: float,
        next_state: Optional[Tuple] = None,
        done: bool = False,
        **kwargs: Any
    ) -> Optional[float]:
        """Store transition and update at episode end.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state (unused)
            done: Whether episode is done
            
        Returns:
            None (no loss for tabular methods)
        """
        self.store_transition(state, action, reward)
        self.steps_trained += 1
        
        if done:
            self.update_from_episode()
            
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
            "returns": dict(self.returns),
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
        self.returns = defaultdict(list, state["returns"])
        self.epsilon = state["epsilon"]
        self.config = state["config"]
        self.episodes_trained = state["episodes_trained"]
        self.steps_trained = state["steps_trained"]
        
        # Restore hyperparameters
        self.gamma = self.config.get("gamma", 0.99)
        self.epsilon_min = self.config.get("epsilon_min", 0.01)
        self.epsilon_decay = self.config.get("epsilon_decay", 0.9995)
