"""Report generation for agent decision analysis."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


class DecisionReporter:
    """Generate detailed reports of agent decisions."""
    
    def __init__(self, output_dir: Path | str = "data/reports"):
        """Initialize reporter.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for decisions
        self.decisions: List[Dict] = []
        self.state_action_counts: Dict[Tuple, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.rewards_by_state: Dict[Tuple, List[float]] = defaultdict(list)
    
    @staticmethod
    def _json_default(obj):
        """Handle non-serializable types in JSON."""
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    @staticmethod
    def _convert_to_serializable(obj):
        """Recursively convert numpy types to native Python types."""
        if isinstance(obj, dict):
            return {k: DecisionReporter._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [DecisionReporter._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    def record_decision(
        self,
        state: Tuple,
        action: int,
        reward: float,
        next_state: Optional[Tuple] = None,
        episode: int = 0,
        step: int = 0,
        player_cards: Optional[List[int]] = None,
        dealer_cards: Optional[List[int]] = None,
        dealer_final_sum: Optional[int] = None,
    ) -> None:
        """Record an agent decision.
        
        Args:
            state: State (player_sum, dealer_card, usable_ace, [true_count])
            action: Action taken (0=STAND, 1=HIT)
            reward: Reward received
            next_state: Next state
            episode: Episode number
            step: Step in episode
            player_cards: List of player's cards
            dealer_cards: List of dealer's cards (visible + hidden after game)
            dealer_final_sum: Dealer's final sum after game
        """
        player_sum, dealer_card, usable_ace = state[:3]
        true_count = state[3] if len(state) > 3 else None
        
        decision = {
            "episode": episode,
            "step": step,
            "player_sum": int(player_sum),
            "dealer_visible_card": int(dealer_card),
            "usable_ace": bool(usable_ace),
            "true_count": float(true_count) if true_count is not None else None,
            "action": int(action),
            "action_name": "HIT" if action == 1 else "STAND",
            "reward": float(reward),
            "next_player_sum": int(next_state[0]) if next_state else None,
            "player_cards": [int(c) for c in player_cards] if player_cards else None,
            "dealer_cards": [int(c) for c in dealer_cards] if dealer_cards else None,
            "dealer_final_sum": int(dealer_final_sum) if dealer_final_sum is not None else None,
        }
        
        self.decisions.append(decision)
        self.state_action_counts[state][action] += 1
        self.rewards_by_state[state].append(reward)
    
    def generate_summary_report(self, agent_name: str) -> Dict:
        """Generate summary report of all decisions.
        
        Args:
            agent_name: Name of agent
            
        Returns:
            Summary report dictionary
        """
        if not self.decisions:
            return {}
        
        # Basic statistics
        total_decisions = len(self.decisions)
        total_reward = sum(d["reward"] for d in self.decisions)
        
        hit_count = sum(1 for d in self.decisions if d["action"] == 1)
        stand_count = total_decisions - hit_count
        
        # Analyze decisions by state
        state_stats = {}
        for state, action_counts in self.state_action_counts.items():
            player_sum, dealer_card, usable_ace = state[:3]
            true_count = state[3] if len(state) > 3 else None
            
            state_key = f"P{player_sum}_D{dealer_card}_A{int(usable_ace)}"
            if true_count is not None:
                state_key += f"_TC{true_count:.1f}"
            
            total_actions = sum(action_counts.values())
            
            state_stats[state_key] = {
                "total_visits": total_actions,
                "hit_count": action_counts.get(1, 0),
                "stand_count": action_counts.get(0, 0),
                "hit_rate": action_counts.get(1, 0) / total_actions if total_actions > 0 else 0,
                "avg_reward": np.mean(self.rewards_by_state[state]),
                "std_reward": np.std(self.rewards_by_state[state]),
                "min_reward": np.min(self.rewards_by_state[state]),
                "max_reward": np.max(self.rewards_by_state[state]),
            }
        
        # Decision distribution by sum
        sum_distribution = defaultdict(lambda: {"hit": 0, "stand": 0, "total": 0})
        for decision in self.decisions:
            player_sum = decision["player_sum"]
            action = decision["action"]
            sum_distribution[player_sum]["total"] += 1
            if action == 1:
                sum_distribution[player_sum]["hit"] += 1
            else:
                sum_distribution[player_sum]["stand"] += 1
        
        # Convert to standard dict
        sum_distribution = dict(sum_distribution)
        
        # Decision distribution by dealer card
        dealer_distribution = defaultdict(lambda: {"hit": 0, "stand": 0, "total": 0})
        for decision in self.decisions:
            dealer_card = decision.get("dealer_visible_card", decision.get("dealer_card", 0))
            action = decision["action"]
            dealer_distribution[dealer_card]["total"] += 1
            if action == 1:
                dealer_distribution[dealer_card]["hit"] += 1
            else:
                dealer_distribution[dealer_card]["stand"] += 1
        
        dealer_distribution = dict(dealer_distribution)
        
        report = {
            "agent_name": agent_name,
            "total_decisions": total_decisions,
            "total_reward": float(total_reward),
            "avg_reward_per_decision": float(total_reward / total_decisions if total_decisions > 0 else 0),
            "hit_count": hit_count,
            "stand_count": stand_count,
            "hit_ratio": float(hit_count / total_decisions if total_decisions > 0 else 0),
            "stand_ratio": float(stand_count / total_decisions if total_decisions > 0 else 0),
            "state_stats": state_stats,
            "sum_distribution": sum_distribution,
            "dealer_distribution": dealer_distribution,
        }
        
        return report
    
    def generate_policy_table(self, agent_name: str) -> Dict:
        """Generate policy table (best action for each state).
        
        Args:
            agent_name: Name of agent
            
        Returns:
            Policy table
        """
        policy = {}
        
        for state, action_counts in self.state_action_counts.items():
            if not action_counts:
                continue
            
            player_sum = state[0]
            dealer_card = state[1]
            usable_ace = state[2]
            true_count = state[3] if len(state) > 3 else None
            
            # Determine best action (most frequent)
            best_action = max(action_counts.items(), key=lambda x: x[1])[0]
            hit_count = action_counts.get(1, 0)
            stand_count = action_counts.get(0, 0)
            total = hit_count + stand_count
            
            # Create state key
            if true_count is not None:
                state_key = f"({int(player_sum)},{int(dealer_card)},{int(usable_ace)},{true_count:.1f})"
            else:
                state_key = f"({int(player_sum)},{int(dealer_card)},{int(usable_ace)})"
            
            policy[state_key] = {
                "best_action": "HIT" if best_action == 1 else "STAND",
                "hit_percentage": float(hit_count / total * 100) if total > 0 else 0,
                "stand_percentage": float(stand_count / total * 100) if total > 0 else 0,
                "visit_count": total,
            }
        
        return policy
    
    def save_report(self, agent_name: str) -> None:
        """Save complete report to JSON.
        
        Args:
            agent_name: Name of agent
        """
        report = {
            "summary": self._convert_to_serializable(self.generate_summary_report(agent_name)),
            "policy": self._convert_to_serializable(self.generate_policy_table(agent_name)),
            "all_decisions": self._convert_to_serializable(self.decisions[-1000:]),  # Last 1000 decisions
        }
        
        save_path = self.output_dir / f"{agent_name}_report.json"
        with open(save_path, "w") as f:
            json.dump(report, f, indent=2, default=self._json_default)
        
        print(f"Saved report to {save_path}")
    
    def save_detailed_report(self, agent_name: str) -> None:
        """Save detailed report with all decisions.
        
        Args:
            agent_name: Name of agent
        """
        report = {
            "summary": self._convert_to_serializable(self.generate_summary_report(agent_name)),
            "policy": self._convert_to_serializable(self.generate_policy_table(agent_name)),
            "all_decisions": self._convert_to_serializable(self.decisions),
        }
        
        save_path = self.output_dir / f"{agent_name}_detailed_report.json"
        with open(save_path, "w") as f:
            json.dump(report, f, indent=2, default=self._json_default)
        
        print(f"Saved detailed report to {save_path}")
    
    def print_summary(self, agent_name: str) -> None:
        """Print summary to console.
        
        Args:
            agent_name: Name of agent
        """
        report = self.generate_summary_report(agent_name)
        
        print(f"\n{'='*60}")
        print(f"Decision Summary Report: {agent_name}")
        print(f"{'='*60}")
        print(f"Total Decisions: {report.get('total_decisions', 0)}")
        print(f"Total Reward: {report.get('total_reward', 0):.2f}")
        print(f"Avg Reward/Decision: {report.get('avg_reward_per_decision', 0):.4f}")
        print(f"HIT Rate: {report.get('hit_ratio', 0):.1%}")
        print(f"STAND Rate: {report.get('stand_ratio', 0):.1%}")
        print(f"{'='*60}\n")


class CardSequenceAnalyzer:
    """Analyze card sequences and outcomes."""
    
    def __init__(self, output_dir: Path | str = "data/reports"):
        """Initialize analyzer.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sequences: List[Dict] = []
    
    def record_game(
        self,
        player_cards: List[int],
        dealer_cards: List[int],
        outcome: str,  # "WIN", "LOSE", "DRAW"
        decisions: List[str],  # List of "HIT" or "STAND"
        reward: float,
    ) -> None:
        """Record a complete game.
        
        Args:
            player_cards: List of player cards dealt
            dealer_cards: List of dealer cards dealt
            outcome: Game outcome
            decisions: Agent decisions
            reward: Final reward
        """
        self.sequences.append({
            "player_cards": player_cards,
            "dealer_cards": dealer_cards,
            "outcome": outcome,
            "decisions": decisions,
            "reward": float(reward),
        })
    
    def analyze_patterns(self, agent_name: str) -> Dict:
        """Analyze patterns in card sequences.
        
        Args:
            agent_name: Name of agent
            
        Returns:
            Pattern analysis
        """
        if not self.sequences:
            return {}
        
        outcomes_count = {"WIN": 0, "LOSE": 0, "DRAW": 0}
        avg_reward_by_outcome = {"WIN": [], "LOSE": [], "DRAW": []}
        
        for seq in self.sequences:
            outcome = seq["outcome"]
            outcomes_count[outcome] += 1
            avg_reward_by_outcome[outcome].append(seq["reward"])
        
        analysis = {
            "total_games": len(self.sequences),
            "outcome_distribution": {
                "WIN": outcomes_count["WIN"],
                "LOSE": outcomes_count["LOSE"],
                "DRAW": outcomes_count["DRAW"],
            },
            "avg_reward_by_outcome": {
                outcome: float(np.mean(rewards)) if rewards else 0
                for outcome, rewards in avg_reward_by_outcome.items()
            },
        }
        
        return analysis
    
    def save_sequences(self, agent_name: str, limit: Optional[int] = None) -> None:
        """Save card sequences to JSON.
        
        Args:
            agent_name: Name of agent
            limit: Limit number of sequences (None for all)
        """
        sequences = self.sequences[:limit] if limit else self.sequences
        
        # Convert to serializable format
        sequences_serializable = []
        for seq in sequences:
            sequences_serializable.append({
                "player_cards": [int(c) for c in seq["player_cards"]],
                "dealer_cards": [int(c) for c in seq["dealer_cards"]],
                "outcome": seq["outcome"],
                "decisions": seq["decisions"],
                "reward": float(seq["reward"]),
            })
        
        data = {
            "agent_name": agent_name,
            "total_sequences": len(self.sequences),
            "analysis": self.analyze_patterns(agent_name),
            "sequences": sequences_serializable,
        }
        
        save_path = self.output_dir / f"{agent_name}_card_sequences.json"
        with open(save_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"Saved card sequences to {save_path}")
