"""Hi-Lo card counting system for Blackjack."""

from typing import Optional


class CardCounter:
    """Hi-Lo card counting system for Blackjack.
    
    The Hi-Lo system assigns values to cards:
    - Low cards (2-6): +1
    - Neutral (7-9): 0
    - High cards (10-A): -1
    
    Args:
        num_decks: Number of decks in the shoe
        
    Attributes:
        num_decks: Number of decks in play
        running_count: Current running count
        cards_seen: Total cards dealt since last reset
    """
    
    def __init__(self, num_decks: int = 6) -> None:
        """Initialize card counter.
        
        Args:
            num_decks: Number of decks in the shoe (default: 6)
        """
        self.num_decks = num_decks
        self.running_count: int = 0
        self.cards_seen: int = 0
        
    def update(self, card_value: int) -> None:
        """Update count with new card.
        
        Args:
            card_value: Card value (1=A, 2-10=face value, 11-13=J/Q/K)
        """
        # Convert face cards to 10
        if card_value > 10:
            card_value = 10
            
        # Hi-Lo counting system
        if 2 <= card_value <= 6:
            self.running_count += 1
        elif card_value >= 10 or card_value == 1:
            self.running_count -= 1
        # 7-9 are neutral (no change)
        
        self.cards_seen += 1
        
    def get_true_count(self) -> float:
        """Calculate true count (running count / decks remaining).
        
        The true count normalizes the running count by the number of
        decks remaining, providing a more accurate measure of advantage.
        
        Returns:
            True count as float, or 0.0 if no cards seen
        """
        if self.cards_seen == 0:
            return 0.0
            
        # Calculate decks remaining
        total_cards = self.num_decks * 52
        cards_remaining = total_cards - self.cards_seen
        decks_remaining = cards_remaining / 52
        
        # Avoid division by very small numbers
        decks_remaining = max(decks_remaining, 0.5)
        
        return self.running_count / decks_remaining
    
    def get_true_count_bin(self, min_bin: int = -3, max_bin: int = 3) -> int:
        """Get discretized true count for tabular agents.
        
        Args:
            min_bin: Minimum bin value (default: -3)
            max_bin: Maximum bin value (default: 3)
            
        Returns:
            Binned true count in range [min_bin, max_bin]
        """
        true_count = self.get_true_count()
        binned = int(round(true_count))
        return max(min_bin, min(max_bin, binned))
    
    def reset(self) -> None:
        """Reset the counter to initial state."""
        self.running_count = 0
        self.cards_seen = 0
        
    def __repr__(self) -> str:
        """String representation of counter state."""
        return (
            f"CardCounter(running_count={self.running_count}, "
            f"cards_seen={self.cards_seen}, "
            f"true_count={self.get_true_count():.2f})"
        )
