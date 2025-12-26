"""Double DQN agent - Advanced deep learning architecture."""

import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..base import AgentBase
from ...utils.replay_buffer import ReplayBuffer


class DoubleDQNAgent(AgentBase):
    """Double DQN agent with decoupled action selection and evaluation.
    
    Implements Double DQN algorithm to reduce overestimation bias:
    - Uses online network to SELECT best action
    - Uses target network to EVALUATE that action
    - More stable learning than vanilla DQN
    
    Attributes:
        q_network: Main Q-network (online)
        target_network: Target Q-network
        optimizer: Adam optimizer
        replay_buffer: Experience replay buffer
        epsilon: Exploration rate
        device: CPU or CUDA device
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize Double DQN agent.
        
        Args:
            config: Configuration with hyperparameters
        """
        super().__init__(config)
        
        # Hyperparameters
        self.gamma: float = self.config.get("gamma", 1.0)
        self.epsilon: float = self.config.get("epsilon_start", 1.0)
        self.epsilon_min: float = self.config.get("epsilon_min", 0.05)
        self.epsilon_decay: float = self.config.get("epsilon_decay", 0.99995)
        self.learning_rate: float = self.config.get("learning_rate", 0.001)
        self.batch_size: int = self.config.get("batch_size", 64)
        self.buffer_size: int = self.config.get("buffer_size", 100000)
        self.target_update_freq: int = self.config.get("target_update_freq", 1000)
        self.hidden_dim: int = self.config.get("hidden_dim", 128)
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks (same architecture as DQN)
        from ..naive.dqn import QNetwork
        self.q_network = QNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.target_network = QNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(max_size=self.buffer_size, batch_size=self.batch_size)
        
        # Training counter for target network updates
        self.update_counter = 0
        
    def act(self, state: Tuple, explore: bool = True) -> int:
        """Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            explore: Whether to use exploration
            
        Returns:
            Action (0=STAND, 1=HIT)
        """
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return int(q_values.argmax().item())
    
    def update(
        self,
        state: Tuple,
        action: int,
        reward: float,
        next_state: Optional[Tuple] = None,
        done: bool = False,
        **kwargs: Any
    ) -> Optional[float]:
        """Store transition and update network using Double DQN.
        
        Double DQN update:
        - Select action: a* = argmax_a Q_online(s', a)
        - Evaluate action: Q_target(s', a*)
        - This decoupling reduces overestimation bias
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            
        Returns:
            Loss value if update performed, None otherwise
        """
        # Store transition
        self.replay_buffer.add(state, action, reward, next_state, done)
        self.steps_trained += 1
        
        # Only update if we have enough samples
        if len(self.replay_buffer) < self.batch_size:
            return None
            
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN target calculation
        with torch.no_grad():
            # Use online network to SELECT best actions
            next_actions = self.q_network(next_states).argmax(1)
            # Use target network to EVALUATE those actions
            next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss (Huber loss for robustness)
        loss = nn.functional.smooth_l1_loss(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Update stats
        if done:
            self.episodes_trained += 1
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def save(self, path: Path) -> None:
        """Save agent to disk.
        
        Args:
            path: Path to save agent
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "config": self.config,
            "episodes_trained": self.episodes_trained,
            "steps_trained": self.steps_trained,
            "update_counter": self.update_counter,
        }
        
        torch.save(state, path)
            
    def load(self, path: Path) -> None:
        """Load agent from disk.
        
        Args:
            path: Path to load agent from
        """
        state = torch.load(path, map_location=self.device)
        
        self.q_network.load_state_dict(state["q_network"])
        self.target_network.load_state_dict(state["target_network"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.epsilon = state["epsilon"]
        self.config = state["config"]
        self.episodes_trained = state["episodes_trained"]
        self.steps_trained = state["steps_trained"]
        self.update_counter = state.get("update_counter", 0)
        
        # Restore hyperparameters
        self.gamma = self.config.get("gamma", 1.0)
        self.epsilon_min = self.config.get("epsilon_min", 0.05)
        self.epsilon_decay = self.config.get("epsilon_decay", 0.99995)
        self.learning_rate = self.config.get("learning_rate", 0.001)
        self.batch_size = self.config.get("batch_size", 64)
        self.target_update_freq = self.config.get("target_update_freq", 1000)
