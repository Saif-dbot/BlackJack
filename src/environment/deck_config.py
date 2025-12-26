"""Deck configuration for Blackjack environment."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class DeckConfig:
    """Configuration for Blackjack deck.
    
    Attributes:
        deck_type: Type of deck ('finite' or 'infinite')
        num_decks: Number of decks in shoe (for finite deck)
        natural: Whether natural blackjack pays 1.5x
        sab: Whether to use "stand all 17" rule
    """
    
    deck_type: Literal["finite", "infinite"] = "finite"
    num_decks: int = 6
    natural: bool = True
    sab: bool = True
    
    @property
    def with_replacement(self) -> bool:
        """Whether to deal cards with replacement.
        
        Returns:
            True for infinite deck, False for finite deck
        """
        return self.deck_type == "infinite"
    
    def to_gym_kwargs(self) -> dict:
        """Convert to Gymnasium environment kwargs.
        
        Returns:
            Dictionary of kwargs for gym.make()
        """
        if self.deck_type == "infinite":
            return {
                "natural": self.natural,
                "sab": self.sab,
            }
        else:
            # For finite deck, we need custom wrapper
            return {
                "natural": self.natural,
                "sab": self.sab,
            }
    
    def __repr__(self) -> str:
        """String representation of deck config."""
        if self.deck_type == "finite":
            return f"DeckConfig(finite, {self.num_decks} decks)"
        else:
            return "DeckConfig(infinite)"
