#!/usr/bin/env python3
"""Comprehensive analysis and optimization script for all agents."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.visualization import compare_agents
from src.evaluation.optimizer import HyperparameterOptimizer, FineTuner, AdaptiveScheduler
from src.utils import setup_logger


def load_all_summaries(logs_dir: Path) -> Dict[str, Dict]:
    """Load all training summaries.
    
    Args:
        logs_dir: Directory containing training summaries
        
    Returns:
        Dictionary of agent summaries
    """
    summaries = {}
    
    for summary_file in logs_dir.glob("*_training_summary.json"):
        agent_name = summary_file.stem.replace("_training_summary", "")
        with open(summary_file, "r") as f:
            summaries[agent_name] = json.load(f)
    
    return summaries


def load_all_reports(reports_dir: Path) -> Dict[str, Dict]:
    """Load all decision reports.
    
    Args:
        reports_dir: Directory containing decision reports
        
    Returns:
        Dictionary of agent reports
    """
    reports = {}
    
    for report_file in reports_dir.glob("*_report.json"):
        agent_name = report_file.stem.replace("_report", "")
        with open(report_file, "r") as f:
            reports[agent_name] = json.load(f)
    
    return reports


def generate_comparison_report(
    summaries: Dict[str, Dict],
    reports: Dict[str, Dict],
    output_dir: Path,
) -> None:
    """Generate comprehensive comparison report.
    
    Args:
        summaries: Training summaries
        reports: Decision reports
        output_dir: Output directory
    """
    comparison = {
        "timestamp": str(Path(__file__).stat()),
        "agents": {},
    }
    
    # Collect all metrics
    max_win_rate = 0
    max_avg_return = 0
    
    for agent_name, summary in summaries.items():
        comparison["agents"][agent_name] = {
            "training": summary,
            "decisions": reports.get(agent_name, {}).get("summary", {}),
        }
        
        if summary.get("final_win_rate", 0) > max_win_rate:
            max_win_rate = summary["final_win_rate"]
        if summary.get("final_avg_return", 0) > max_avg_return:
            max_avg_return = summary["final_avg_return"]
    
    # Add rankings
    comparison["rankings"] = {
        "by_win_rate": sorted(
            summaries.items(),
            key=lambda x: x[1].get("final_win_rate", 0),
            reverse=True
        ),
        "by_avg_return": sorted(
            summaries.items(),
            key=lambda x: x[1].get("final_avg_return", 0),
            reverse=True
        ),
        "by_training_time": sorted(
            summaries.items(),
            key=lambda x: x[1].get("training_time_seconds", float("inf"))
        ),
    }
    
    # Save comparison report
    report_path = output_dir / "comprehensive_comparison.json"
    with open(report_path, "w") as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n✓ Saved comprehensive comparison to {report_path}")
    
    # Print rankings
    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE AGENT COMPARISON")
    print(f"{'='*70}")
    
    print(f"\n🏆 RANKING BY WIN RATE:")
    for i, (agent_name, summary) in enumerate(comparison["rankings"]["by_win_rate"], 1):
        wr = summary.get("final_win_rate", 0)
        print(f"  {i}. {agent_name:20} → {wr:.1%}")
    
    print(f"\n💰 RANKING BY AVERAGE RETURN:")
    for i, (agent_name, summary) in enumerate(comparison["rankings"]["by_avg_return"], 1):
        ar = summary.get("final_avg_return", 0)
        print(f"  {i}. {agent_name:20} → {ar:+.4f}")
    
    print(f"\n⚡ RANKING BY TRAINING SPEED:")
    for i, (agent_name, summary) in enumerate(comparison["rankings"]["by_training_time"], 1):
        tt = summary.get("training_time_seconds", 0)
        eps_per_sec = summary.get("total_episodes", 0) / max(tt, 1)
        print(f"  {i}. {agent_name:20} → {tt:7.1f}s ({eps_per_sec:.0f} eps/s)")
    
    print(f"\n{'='*70}\n")


def generate_visual_comparison(
    summaries: Dict[str, Dict],
    output_dir: Path,
) -> None:
    """Generate visual comparison plots.
    
    Args:
        summaries: Training summaries
        output_dir: Output directory
    """
    print(f"\nGenerating visual comparison plots...")
    
    # Prepare data for comparison
    agents_metrics = {}
    for agent_name, summary in summaries.items():
        agents_metrics[agent_name] = {
            "final_win_rate": summary.get("final_win_rate", 0),
            "final_avg_return": summary.get("final_avg_return", 0),
            "training_time": summary.get("training_time_seconds", 0),
        }
    
    # Generate comparison plot
    compare_agents(agents_metrics, output_dir)
    
    # Generate additional plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Comprehensive Agent Analysis", fontsize=16, fontweight="bold")
    
    agent_names = list(summaries.keys())
    win_rates = [summaries[a].get("final_win_rate", 0) for a in agent_names]
    avg_returns = [summaries[a].get("final_avg_return", 0) for a in agent_names]
    test_win_rates = [summaries[a].get("test_win_rate", 0) for a in agent_names]
    training_times = [summaries[a].get("training_time_seconds", 0) for a in agent_names]
    
    # Win Rate comparison (train vs test)
    ax = axes[0, 0]
    x = np.arange(len(agent_names))
    width = 0.35
    ax.bar(x - width/2, win_rates, width, label="Train", alpha=0.8)
    ax.bar(x + width/2, test_win_rates, width, label="Test", alpha=0.8)
    ax.set_xlabel("Agent")
    ax.set_ylabel("Win Rate")
    ax.set_title("Training vs Test Win Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(agent_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    # Average Return
    ax = axes[0, 1]
    colors = ["green" if ar > 0 else "red" for ar in avg_returns]
    bars = ax.bar(agent_names, avg_returns, color=colors, alpha=0.7, edgecolor="black")
    ax.set_ylabel("Average Return")
    ax.set_title("Final Average Return")
    ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
    ax.set_xticklabels(agent_names, rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")
    
    # Training Time
    ax = axes[1, 0]
    ax.barh(agent_names, training_times, color="skyblue", edgecolor="black")
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Training Time")
    ax.grid(True, alpha=0.3, axis="x")
    for i, (name, time) in enumerate(zip(agent_names, training_times)):
        ax.text(time, i, f" {time:.1f}s", va="center")
    
    # Efficiency (win rate per second)
    ax = axes[1, 1]
    efficiency = [wr / max(t, 1) for wr, t in zip(win_rates, training_times)]
    ax.barh(agent_names, efficiency, color="lightcoral", edgecolor="black")
    ax.set_xlabel("Win Rate / Second")
    ax.set_title("Training Efficiency")
    ax.grid(True, alpha=0.3, axis="x")
    
    plt.tight_layout()
    
    save_path = output_dir / "comprehensive_analysis.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved comprehensive analysis plot to {save_path}")
    
    plt.close()


def generate_html_report(
    summaries: Dict[str, Dict],
    reports: Dict[str, Dict],
    output_dir: Path,
) -> None:
    """Generate HTML report for easy viewing.
    
    Args:
        summaries: Training summaries
        reports: Decision reports
        output_dir: Output directory
    """
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Blackjack RL - Agent Comparison Report</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            h1 { color: #333; text-align: center; }
            h2 { color: #555; border-bottom: 2px solid #2c3e50; padding-bottom: 10px; }
            table {
                width: 100%;
                border-collapse: collapse;
                background-color: white;
                margin: 20px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            th, td {
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            th {
                background-color: #2c3e50;
                color: white;
                font-weight: bold;
            }
            tr:hover { background-color: #f0f0f0; }
            .metric-box {
                background-color: white;
                padding: 15px;
                margin: 10px 0;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .positive { color: green; font-weight: bold; }
            .negative { color: red; font-weight: bold; }
            .neutral { color: #666; }
            .progress-bar {
                width: 100%;
                height: 20px;
                background-color: #e0e0e0;
                border-radius: 10px;
                overflow: hidden;
            }
            .progress-fill {
                height: 100%;
                background-color: #4CAF50;
                width: 0%;
            }
        </style>
    </head>
    <body>
        <h1>🎰 Blackjack Reinforcement Learning - Agent Comparison Report</h1>
        
        <h2>📊 Performance Summary</h2>
    """
    
    # Create performance table
    html_content += """
        <table>
            <tr>
                <th>Agent</th>
                <th>Win Rate</th>
                <th>Test Win Rate</th>
                <th>Avg Return</th>
                <th>Training Time</th>
            </tr>
    """
    
    for agent_name, summary in summaries.items():
        wr = summary.get("final_win_rate", 0)
        twr = summary.get("test_win_rate", 0)
        ar = summary.get("final_avg_return", 0)
        tt = summary.get("training_time_seconds", 0)
        
        wr_class = "positive" if wr > 0.45 else "neutral"
        ar_class = "positive" if ar > 0 else "negative"
        
        html_content += f"""
            <tr>
                <td><strong>{agent_name}</strong></td>
                <td class="{wr_class}">{wr:.1%}</td>
                <td class="{wr_class}">{twr:.1%}</td>
                <td class="{ar_class}">{ar:+.4f}</td>
                <td>{tt:.1f}s</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <h2>📈 Visualizations</h2>
        <div class="metric-box">
            <p><strong>Comparison Plot:</strong> <a href="agents_comparison.png">View</a></p>
            <p><strong>Comprehensive Analysis:</strong> <a href="comprehensive_analysis.png">View</a></p>
        </div>
        
        <h2>📋 Detailed Metrics</h2>
    """
    
    for agent_name, summary in summaries.items():
        decision_summary = reports.get(agent_name, {}).get("summary", {})
        
        html_content += f"""
        <div class="metric-box">
            <h3>{agent_name}</h3>
            <p><strong>Final Win Rate:</strong> {summary.get('final_win_rate', 0):.1%}</p>
            <p><strong>Test Win Rate:</strong> {summary.get('test_win_rate', 0):.1%}</p>
            <p><strong>Average Return:</strong> {summary.get('final_avg_return', 0):+.4f}</p>
            <p><strong>Training Time:</strong> {summary.get('training_time_seconds', 0):.1f}s</p>
            <p><strong>Total Decisions (test):</strong> {decision_summary.get('total_decisions', 'N/A')}</p>
            <p><strong>HIT Ratio:</strong> {decision_summary.get('hit_ratio', 0):.1%}</p>
        </div>
        """
    
    html_content += """
        <footer style="text-align: center; margin-top: 40px; color: #999;">
            Generated by Blackjack RL Analysis System
        </footer>
    </body>
    </html>
    """
    
    report_path = output_dir / "comparison_report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"✓ Saved HTML report to {report_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive analysis and comparison of all agents"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Data directory containing logs and reports (default: data)",
    )
    parser.add_argument(
        "--generate-html",
        action="store_true",
        help="Generate HTML report",
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    logs_dir = data_dir / "logs"
    reports_dir = data_dir / "reports"
    plots_dir = data_dir / "plots"
    
    # Load all data
    print(f"Loading training summaries from {logs_dir}...")
    summaries = load_all_summaries(logs_dir)
    
    print(f"Loading decision reports from {reports_dir}...")
    reports = load_all_reports(reports_dir)
    
    if not summaries:
        print("❌ No training summaries found!")
        return
    
    # Generate reports
    generate_comparison_report(summaries, reports, plots_dir)
    generate_visual_comparison(summaries, plots_dir)
    
    if args.generate_html:
        generate_html_report(summaries, reports, plots_dir)
        print(f"\n💡 Open '{plots_dir}/comparison_report.html' in a browser to view the report")
    
    print(f"\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
