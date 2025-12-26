"""Blackjack environment wrapper with card counting support."""

from typing import Optional, Tuple, List

import gymnasium as gym
import numpy as np

from .card_counting import CardCounter
from .deck_config import DeckConfig


class DetailedBlackjackEnv(gym.Wrapper):
    """Enhanced Blackjack environment that tracks all cards.
    
    This wrapper intercepts the Blackjack environment to track:
    - Player's cards
    - Dealer's cards (including hidden card)
    - Game outcomes
    
    Unlike standard Gymnasium Blackjack, this provides full card visibility.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_player_cards = []
        self.current_dealer_cards = []
        self.dealer_final_sum = None
        
    def reset(self, **kwargs):
        """Reset and track initial cards."""
        obs, info = self.env.reset(**kwargs)
        
        # Extract initial cards from observation
        player_sum, dealer_card, usable_ace = obs
        
        # Initialize tracking
        self.current_player_cards = [player_sum]  # Approximate initial cards
        self.current_dealer_cards = [dealer_card]
        self.dealer_final_sum = None
        
        # Add to info
        info['player_cards'] = self.current_player_cards.copy()
        info['dealer_cards'] = self.current_dealer_cards.copy()
        
        return obs, info
    
    def step(self, action):
        """Step and track cards."""
        prev_sum = getattr(self.unwrapped, 'player', [None])[0] if hasattr(self.unwrapped, 'player') else None
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        player_sum, dealer_card, usable_ace = obs
        
        # Track new card if HIT
        if action == 1 and prev_sum is not None:
            new_card = player_sum - prev_sum
            if new_card > 0:
                self.current_player_cards.append(new_card)
        
        # After game ends, get dealer's final cards
        if terminated:
            # Gymnasium Blackjack doesn't expose dealer cards, use observation
            self.dealer_final_sum = dealer_card
        
        # Update info
        info['player_cards'] = self.current_player_cards.copy()
        info['dealer_cards'] = self.current_dealer_cards.copy()
        info['dealer_final_sum'] = self.dealer_final_sum
        
        return obs, reward, terminated, truncated, info


class BlackjackEnv(gym.Wrapper):
    """Blackjack environment wrapper with card counting.
    
    Wraps Gymnasium's Blackjack-v1 environment and adds:
    - Card counting functionality (Hi-Lo system)
    - Extended observation space with count features
    - Deck configuration (finite/infinite)
    
    Args:
        deck_config: Configuration for deck type and parameters
        enable_counting: Whether to enable card counting features
        
    Attributes:
        counter: CardCounter instance for tracking count
        enable_counting: Whether counting is enabled
        deck_config: Deck configuration
    """
    
    def __init__(
        self,
        deck_config: Optional[DeckConfig] = None,
        enable_counting: bool = False,
    ) -> None:
        """Initialize Blackjack environment.
        
        Args:
            deck_config: Deck configuration (default: finite 6 decks)
            enable_counting: Enable card counting features
        """
        self.deck_config = deck_config or DeckConfig()
        self.enable_counting = enable_counting
        
        # Create base Gymnasium environment
        env_kwargs = self.deck_config.to_gym_kwargs()
        base_env = gym.make("Blackjack-v1", **env_kwargs)
        super().__init__(base_env)
        
        # Initialize card counter
        self.counter = CardCounter(num_decks=self.deck_config.num_decks)
        
        # Track episode statistics
        self.episode_cards = []
        self.player_cards = []
        self.dealer_cards = []
        
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[tuple, dict]:
        """Reset environment and counter.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            observation: Initial observation
            info: Additional information
        """
        obs, info = self.env.reset(seed=seed, options=options)
        
        # Reset counter
        self.counter.reset()
        self.episode_cards = []
        self.player_cards = []
        self.dealer_cards = []
        
        # Update counter with visible cards
        player_sum, dealer_card, usable_ace = obs
        # Note: Gymnasium doesn't give exact cards, only sums
        # Track dealer's visible card
        self.counter.update(dealer_card)
        self.episode_cards.append(dealer_card)
        self.dealer_cards.append(dealer_card)
        
        # Store initial info
        info['player_cards'] = self.player_cards.copy()
        info['dealer_cards'] = self.dealer_cards.copy()
        
        # Extend observation with counting features if enabled
        if self.enable_counting:
            obs_extended = self._extend_observation(obs)
            return obs_extended, info
        
        return obs, info
    
    def step(self, action: int) -> Tuple[tuple, float, bool, bool, dict]:
        """Execute action and update counter.
        
        Args:
            action: Action to take (0=stand, 1=hit)
            
        Returns:
            observation: Next observation
            reward: Reward received
            terminated: Whether episode ended
            truncated: Whether episode was truncated
            info: Additional information
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Add card tracking info
        # Note: Gymnasium Blackjack doesn't provide actual cards,
        # but we can infer some information from the state
        player_sum, dealer_card, usable_ace = obs[:3] if isinstance(obs, tuple) else obs
        
        # Track player cards (approximate based on action and sum change)
        if action == 1:  # HIT
            # A new card was added, but we don't know which one exactly
            # Store the sum instead for approximation
            if len(self.player_cards) == 0:
                self.player_cards.append(player_sum)
        
        # After game ends, try to get dealer's final cards from the environment
        if terminated:
            # The dealer's final sum is in the observation
            # We can't get exact cards from Gymnasium, so store what we have
            info['dealer_final_sum'] = dealer_card
        
        # Update info with card tracking
        info['player_cards'] = self.player_cards.copy()
        info['dealer_cards'] = self.dealer_cards.copy()
        
        # Extend observation with counting features if enabled
        if self.enable_counting:
            obs_extended = self._extend_observation(obs)
            return obs_extended, float(reward), terminated, truncated, info
        
        return obs, float(reward), terminated, truncated, info
    
    def _extend_observation(self, obs: tuple) -> tuple:
        """Extend observation with counting features.
        
        Args:
            obs: Base observation (player_sum, dealer_card, usable_ace)
            
        Returns:
            Extended observation with count features (4-dimensional)
        """
        player_sum, dealer_card, usable_ace = obs
        
        # Add true count (continuous value)
        true_count = self.counter.get_true_count()
        
        # Return extended observation
        # Format: (player_sum, dealer_card, usable_ace, true_count)
        return (player_sum, dealer_card, usable_ace, true_count)
    
    def get_state_naive(self, obs: tuple) -> tuple:
        """Extract naive state (without counting) from observation.
        
        Args:
            obs: Observation (may be extended or base)
            
        Returns:
            Naive state (player_sum, dealer_card, usable_ace)
        """
        # Always return first 3 elements
        return obs[:3] if isinstance(obs, tuple) else obs
    
    def get_count_features(self, obs: tuple) -> dict:
        """Extract count features from extended observation.
        
        Args:
            obs: Extended observation
            
        Returns:
            Dictionary with count features
        """
        if not self.enable_counting or len(obs) < 4:
            return {
                "true_count": 0.0,
            }
        
        _, _, _, true_count = obs
        return {
            "true_count": true_count,
        }
    
    def __repr__(self) -> str:
        """String representation of environment."""
        count_status = "enabled" if self.enable_counting else "disabled"
        return f"BlackjackEnv({self.deck_config}, counting={count_status})"
