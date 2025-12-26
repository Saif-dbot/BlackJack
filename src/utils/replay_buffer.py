"""Experience replay buffer for DQN."""

from collections import deque
from typing import Tuple, List, Optional
import random
import numpy as np


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples.
    
    Stores (state, action, reward, next_state, done) transitions
    and provides random sampling for experience replay.
    
    Attributes:
        buffer: Deque storing transitions
        batch_size: Number of samples per batch
        max_size: Maximum buffer capacity
    """
    
    def __init__(self, max_size: int = 100000, batch_size: int = 64) -> None:
        """Initialize replay buffer.
        
        Args:
            max_size: Maximum number of transitions to store
            batch_size: Number of samples per batch
        """
        self.buffer = deque(maxlen=max_size)
        self.batch_size = batch_size
        self.max_size = max_size
        
    def add(
        self,
        state: Tuple,
        action: int,
        reward: float,
        next_state: Optional[Tuple],
        done: bool
    ) -> None:
        """Add a new transition to buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state (None if episode ended)
            done: Whether episode ended
        """
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch of transitions.
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as numpy arrays
        """
        batch = random.sample(self.buffer, self.batch_size)
        
        states = np.array([t[0] for t in batch], dtype=np.float32)
        actions = np.array([t[1] for t in batch], dtype=np.int64)
        rewards = np.array([t[2] for t in batch], dtype=np.float32)
        next_states = np.array([t[3] for t in batch], dtype=np.float32)
        dones = np.array([t[4] for t in batch], dtype=np.float32)
        
        return states, actions, rewards, next_states, dones
        
    def __len__(self) -> int:
        """Return current buffer size.
        
        Returns:
            Number of transitions in buffer
        """
        return len(self.buffer)
        
    def __repr__(self) -> str:
        """String representation.
        
        Returns:
            Description of buffer state
        """
        return f"ReplayBuffer(size={len(self)}/{self.max_size}, batch_size={self.batch_size})"
