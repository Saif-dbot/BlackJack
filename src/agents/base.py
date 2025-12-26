"""Base class for all reinforcement learning agents."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


class AgentBase(ABC):
    """Abstract base class for RL agents.
    
    All agents must implement:
    - act(): Select action given state
    - update(): Update agent from experience
    - save(): Save agent to disk
    - load(): Load agent from disk
    
    Attributes:
        state_dim: Dimension of state space
        action_dim: Dimension of action space (always 2 for Blackjack: HIT/STAND)
        config: Configuration dictionary
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize agent.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.state_dim: int = 3  # Default: (player_sum, dealer_card, usable_ace)
        self.action_dim: int = 2  # HIT or STAND
        
        # Training statistics
        self.episodes_trained: int = 0
        self.steps_trained: int = 0
        
    @abstractmethod
    def act(self, state: Tuple, explore: bool = True) -> int:
        """Select action for given state.
        
        Args:
            state: Current state
            explore: Whether to explore (epsilon-greedy) or exploit
            
        Returns:
            Action to take (0=STAND, 1=HIT)
        """
        pass
    
    @abstractmethod
    def update(
        self,
        state: Tuple,
        action: int,
        reward: float,
        next_state: Optional[Tuple] = None,
        done: bool = False,
        **kwargs: Any
    ) -> Optional[float]:
        """Update agent from experience.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state (if applicable)
            done: Whether episode is done
            **kwargs: Additional algorithm-specific args
            
        Returns:
            Loss or None (for tabular methods)
        """
        pass
    
    @abstractmethod
    def save(self, path: Path) -> None:
        """Save agent to disk.
        
        Args:
            path: Path to save agent
        """
        pass
    
    @abstractmethod
    def load(self, path: Path) -> None:
        """Load agent from disk.
        
        Args:
            path: Path to load agent from
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Get agent configuration.
        
        Returns:
            Configuration dictionary
        """
        return self.config.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics.
        
        Returns:
            Dictionary with training stats
        """
        return {
            "episodes_trained": self.episodes_trained,
            "steps_trained": self.steps_trained,
        }
    
    def __repr__(self) -> str:
        """String representation of agent."""
        return f"{self.__class__.__name__}(episodes={self.episodes_trained})"
