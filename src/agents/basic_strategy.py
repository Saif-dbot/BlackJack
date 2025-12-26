"""Implémentation de la Basic Strategy optimale pour le Blackjack."""

import numpy as np
from typing import Tuple


class BasicStrategyAgent:
    """Agent qui utilise la stratégie mathématiquement optimale pour le Blackjack."""
    
    def __init__(self):
        """Initialise la Basic Strategy avec les tables de décision optimales."""
        self.name = "Basic Strategy"
        
        # Table pour les mains dures (sans As utilisable)
        # Lignes: somme du joueur (4-21), Colonnes: carte visible du dealer (2-11)
        self.hard_strategy = self._init_hard_strategy()
        
        # Table pour les mains souples (avec As utilisable)
        # Lignes: somme du joueur (13-21), Colonnes: carte visible du dealer (2-11)
        self.soft_strategy = self._init_soft_strategy()
    
    def _init_hard_strategy(self):
        """Initialise la table de stratégie pour les mains dures.
        
        Returns:
            dict: {(player_sum, dealer_card): action}
                  action: 0=Stand, 1=Hit
        """
        strategy = {}
        
        # Règles pour les mains dures
        for player_sum in range(4, 22):
            for dealer_card in range(2, 12):
                # Toujours tirer si <= 11
                if player_sum <= 11:
                    strategy[(player_sum, dealer_card)] = 1  # Hit
                
                # 12: Tirer sauf si dealer a 4-6
                elif player_sum == 12:
                    if dealer_card in [4, 5, 6]:
                        strategy[(player_sum, dealer_card)] = 0  # Stand
                    else:
                        strategy[(player_sum, dealer_card)] = 1  # Hit
                
                # 13-16: Tirer si dealer a 7 ou plus
                elif 13 <= player_sum <= 16:
                    if dealer_card >= 7:
                        strategy[(player_sum, dealer_card)] = 1  # Hit
                    else:
                        strategy[(player_sum, dealer_card)] = 0  # Stand
                
                # 17+: Toujours rester
                else:
                    strategy[(player_sum, dealer_card)] = 0  # Stand
        
        return strategy
    
    def _init_soft_strategy(self):
        """Initialise la table de stratégie pour les mains souples (avec As).
        
        Returns:
            dict: {(player_sum, dealer_card): action}
        """
        strategy = {}
        
        # Règles pour les mains souples
        for player_sum in range(13, 22):
            for dealer_card in range(2, 12):
                # Soft 13-17: Tirer sauf si dealer a 5-6 (pour soft 17-18)
                if player_sum <= 17:
                    strategy[(player_sum, dealer_card)] = 1  # Hit
                
                # Soft 18: Rester si dealer a 2,7,8, tirer sinon
                elif player_sum == 18:
                    if dealer_card in [2, 7, 8]:
                        strategy[(player_sum, dealer_card)] = 0  # Stand
                    elif dealer_card in [9, 10, 11]:
                        strategy[(player_sum, dealer_card)] = 1  # Hit
                    else:  # 3-6: normalement double, mais on reste
                        strategy[(player_sum, dealer_card)] = 0  # Stand
                
                # Soft 19+: Toujours rester
                else:
                    strategy[(player_sum, dealer_card)] = 0  # Stand
        
        return strategy
    
    def act(self, state: Tuple[int, int, bool], explore: bool = False) -> int:
        """Prend une décision selon la Basic Strategy.
        
        Args:
            state: (player_sum, dealer_card, usable_ace)
            explore: Ignoré (pas d'exploration pour la stratégie optimale)
        
        Returns:
            action: 0 (Stand) ou 1 (Hit)
        """
        player_sum, dealer_card, usable_ace = state
        
        # Toujours tirer si on a moins de 12
        if player_sum < 12:
            return 1
        
        # Toujours rester si on a 21
        if player_sum == 21:
            return 0
        
        # Choisir la stratégie appropriée
        if usable_ace and player_sum >= 13:
            # Main souple (avec As utilisable)
            strategy = self.soft_strategy
        else:
            # Main dure
            strategy = self.hard_strategy
        
        # Récupérer l'action optimale
        action = strategy.get((player_sum, dealer_card), 0)
        
        return action
    
    def update(self, state, action, reward, next_state, done):
        """Pas de mise à jour pour la Basic Strategy (stratégie fixe)."""
        pass
    
    def save(self, path):
        """Pas besoin de sauvegarder (stratégie déterministe)."""
        pass
    
    def load(self, path):
        """Pas besoin de charger (stratégie déterministe)."""
        pass
    
    def get_policy_string(self):
        """Retourne une représentation textuelle de la stratégie."""
        lines = ["BASIC STRATEGY - HARD HANDS", "=" * 50]
        lines.append("Player\\Dealer  2   3   4   5   6   7   8   9  10   A")
        
        for player_sum in range(17, 7, -1):
            row = f"{player_sum:2d}          "
            for dealer_card in range(2, 12):
                action = self.hard_strategy.get((player_sum, dealer_card), 0)
                row += " S  " if action == 0 else " H  "
            lines.append(row)
        
        lines.append("")
        lines.append("BASIC STRATEGY - SOFT HANDS")
        lines.append("=" * 50)
        lines.append("Player\\Dealer  2   3   4   5   6   7   8   9  10   A")
        
        for player_sum in range(21, 12, -1):
            row = f"A,{player_sum-11}         "
            for dealer_card in range(2, 12):
                action = self.soft_strategy.get((player_sum, dealer_card), 1)
                row += " S  " if action == 0 else " H  "
            lines.append(row)
        
        return "\n".join(lines)


if __name__ == "__main__":
    # Test de la Basic Strategy
    agent = BasicStrategyAgent()
    
    print(agent.get_policy_string())
    
    # Tests de décisions
    print("\n" + "=" * 50)
    print("TESTS DE DÉCISIONS")
    print("=" * 50)
    
    test_cases = [
        ((16, 10, False), "Hard 16 vs 10"),
        ((16, 6, False), "Hard 16 vs 6"),
        ((12, 5, False), "Hard 12 vs 5"),
        ((12, 7, False), "Hard 12 vs 7"),
        ((18, 9, True), "Soft 18 vs 9"),
        ((18, 7, True), "Soft 18 vs 7"),
        ((20, 11, False), "Hard 20 vs A"),
        ((11, 5, False), "Hard 11 vs 5"),
    ]
    
    for state, description in test_cases:
        action = agent.act(state)
        action_str = "Stand" if action == 0 else "Hit"
        print(f"{description:20s} -> {action_str}")
