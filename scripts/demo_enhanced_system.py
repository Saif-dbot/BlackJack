#!/usr/bin/env python3
"""Quick demo of enhanced training system."""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def print_header(text: str, char: str = "=") -> None:
    """Print a formatted header."""
    print(f"\n{char * 70}")
    print(f"  {text}")
    print(f"{char * 70}\n")


def check_generated_files(data_dir: Path) -> None:
    """Check what files were generated."""
    print_header("Generated Files Summary")
    
    models_naive = list((data_dir / "models" / "naive").glob("*_final.pkl"))
    models_counting = list((data_dir / "models" / "counting").glob("*_final.pkl"))
    reports = list((data_dir / "reports").glob("*_report.json"))
    plots = list((data_dir / "plots").glob("*.png"))
    summaries = list((data_dir / "logs").glob("*_training_summary.json"))
    
    print(f"✓ Trained Models (Naive): {len(models_naive)}")
    for model in sorted(models_naive):
        size_kb = model.stat().st_size / 1024
        print(f"  - {model.name} ({size_kb:.1f} KB)")
    
    print(f"\n✓ Trained Models (Counting): {len(models_counting)}")
    for model in sorted(models_counting):
        size_kb = model.stat().st_size / 1024
        print(f"  - {model.name} ({size_kb:.1f} KB)")
    
    print(f"\n✓ Reports Generated: {len(reports)}")
    for report in sorted(reports):
        size_kb = report.stat().st_size / 1024
        print(f"  - {report.name} ({size_kb:.1f} KB)")
    
    print(f"\n✓ Plots Generated: {len(plots)}")
    for plot in sorted(plots):
        size_kb = plot.stat().st_size / 1024
        print(f"  - {plot.name} ({size_kb:.1f} KB)")
    
    print(f"\n✓ Training Summaries: {len(summaries)}")
    for summary in sorted(summaries):
        with open(summary) as f:
            data = json.load(f)
        print(f"  - {summary.name}")
        print(f"    Win Rate: {data.get('final_win_rate', 0):.1%}")
        print(f"    Time: {data.get('training_time_seconds', 0):.1f}s")


def show_sample_report(data_dir: Path) -> None:
    """Show a sample report."""
    print_header("Sample Report Content")
    
    report_file = data_dir / "reports" / "qlearning_report.json"
    if not report_file.exists():
        print("No qlearning report found. Train an agent first!")
        return
    
    with open(report_file) as f:
        report = json.load(f)
    
    summary = report.get("summary", {})
    policy = report.get("policy", {})
    
    print("📊 Decision Summary:")
    print(f"  Total Decisions: {summary.get('total_decisions', 0)}")
    print(f"  Total Reward: {summary.get('total_reward', 0):.2f}")
    print(f"  HIT Ratio: {summary.get('hit_ratio', 0):.1%}")
    print(f"  STAND Ratio: {summary.get('stand_ratio', 0):.1%}")
    
    print(f"\n🎯 Sample Policy (first 5 states):")
    for i, (state, action_info) in enumerate(list(policy.items())[:5]):
        print(f"  {state}: {action_info['best_action']} " + 
              f"({action_info['hit_percentage']:.0f}% HIT)")


def show_comparison(data_dir: Path) -> None:
    """Show agent comparison."""
    print_header("Agent Performance Comparison")
    
    summaries_dir = data_dir / "logs"
    if not summaries_dir.exists():
        print("No summaries found. Train agents first!")
        return
    
    summaries = {}
    for summary_file in sorted(summaries_dir.glob("*_training_summary.json")):
        with open(summary_file) as f:
            data = json.load(f)
        agent_name = summary_file.stem.replace("_training_summary", "")
        summaries[agent_name] = data
    
    if not summaries:
        print("No training summaries found!")
        return
    
    # Sort by win rate
    sorted_agents = sorted(
        summaries.items(),
        key=lambda x: x[1].get("final_win_rate", 0),
        reverse=True
    )
    
    print(f"{'Agent':<20} {'Win Rate':<12} {'Avg Return':<14} {'Time (s)':<10}")
    print("-" * 56)
    
    for agent_name, summary in sorted_agents:
        wr = summary.get("final_win_rate", 0)
        ar = summary.get("final_avg_return", 0)
        t = summary.get("training_time_seconds", 0)
        print(f"{agent_name:<20} {wr:>10.1%}  {ar:>12.4f}  {t:>8.1f}s")


def show_next_steps() -> None:
    """Show what to do next."""
    print_header("Next Steps", "→")
    
    print("""
1. TRAIN AGENTS
   python scripts/train_naive_enhanced.py --config config/agents_naive/qlearning.yaml
   python scripts/train_naive_enhanced.py --config config/agents_naive/sarsa.yaml
   python scripts/train_naive_enhanced.py --config config/agents_naive/monte_carlo.yaml
   
2. TRAIN COUNTING AGENTS
   python scripts/train_counting_enhanced.py --config config/agents_counting/qlearning_count.yaml
   python scripts/train_counting_enhanced.py --config config/agents_counting/sarsa_count.yaml
   
3. ANALYZE RESULTS
   python scripts/analyze_agents.py --data-dir data --generate-html
   
4. CLEANUP OLD FILES (optional)
   python scripts/cleanup_models.py --all

📊 Generated files will be in:
   - data/models/        : Trained models (.pkl)
   - data/plots/         : Visualizations (.png, .json, .html)
   - data/reports/       : Decision reports (.json)
   - data/logs/          : Training summaries (.json)
    """)


def main():
    """Main demo function."""
    data_dir = Path("data")
    
    print("\n" + "=" * 70)
    print("  🎰 BLACKJACK RL - ENHANCED TRAINING SYSTEM DEMO")
    print("=" * 70)
    
    # Check if data exists
    if not data_dir.exists():
        print("\n⚠️  No data directory found. This is normal for first run!")
        show_next_steps()
        return
    
    # Show generated files
    check_generated_files(data_dir)
    
    # Show sample report
    show_sample_report(data_dir)
    
    # Show comparison
    show_comparison(data_dir)
    
    # Show next steps
    show_next_steps()
    
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
