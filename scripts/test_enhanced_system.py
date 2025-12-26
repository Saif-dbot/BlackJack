#!/usr/bin/env python3
"""Quick test script to validate enhanced training system."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("Testing enhanced training system...")
print("=" * 60)

# Test 1: Check visualization module
print("\n1. Testing visualization module...")
try:
    from src.evaluation.visualization import TrainingVisualizer, compare_agents
    visualizer = TrainingVisualizer()
    print("   ✓ TrainingVisualizer imported successfully")
    
    # Add some test data
    for i in range(1, 11):
        visualizer.add_metrics(
            episode=i*100,
            win_rate=0.40 + i*0.005,
            lose_rate=0.35 - i*0.003,
            draw_rate=0.25 - i*0.002,
            avg_return=-0.06 + i*0.001,
            epsilon=1.0 * (0.99995 ** (i*100))
        )
    print("   ✓ Added test metrics")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 2: Check reporter module
print("\n2. Testing reporter module...")
try:
    from src.evaluation.reporter import DecisionReporter, CardSequenceAnalyzer
    reporter = DecisionReporter()
    
    # Add test decisions
    for i in range(100):
        reporter.record_decision(
            state=(15, 5, 0),
            action=1,
            reward=-1,
            episode=i
        )
    
    summary = reporter.generate_summary_report("test_agent")
    print(f"   ✓ DecisionReporter working: {summary.get('total_decisions', 0)} decisions recorded")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 3: Check optimizer module
print("\n3. Testing optimizer module...")
try:
    from src.evaluation.optimizer import HyperparameterOptimizer, FineTuner, AdaptiveScheduler
    
    optimizer = HyperparameterOptimizer()
    print("   ✓ HyperparameterOptimizer imported")
    
    finetuner = FineTuner()
    print("   ✓ FineTuner imported")
    
    scheduler = AdaptiveScheduler()
    linear_sched = scheduler.linear_schedule(1.0, 0.0, 1000)
    print(f"   ✓ AdaptiveScheduler working: schedule(0)={linear_sched(0):.2f}, schedule(1000)={linear_sched(1000):.2f}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 4: Check agent imports
print("\n4. Testing agent imports...")
try:
    from src.agents.naive import MonteCarloAgent, QLearningAgent, SARSAAgent, DQNAgent, DoubleDQNAgent
    print("   ✓ All naive agents imported")
    
    from src.agents.counting import MonteCarloCountAgent, QLearningCountAgent, SARSACountAgent, DQNCountAgent
    print("   ✓ All counting agents imported")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 5: Check environment
print("\n5. Testing environment...")
try:
    from src.environment import BlackjackEnv, DeckConfig
    
    deck_config = DeckConfig(deck_type="infinite", num_decks=1)
    env = BlackjackEnv(deck_config=deck_config, enable_counting=False)
    obs, info = env.reset()
    print(f"   ✓ BlackjackEnv created: obs={obs}")
    
    env_count = BlackjackEnv(deck_config=deck_config, enable_counting=True)
    obs, info = env_count.reset()
    print(f"   ✓ BlackjackEnv with counting: obs={obs}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 6: Check matplotlib
print("\n6. Testing matplotlib...")
try:
    import matplotlib.pyplot as plt
    print("   ✓ Matplotlib imported successfully")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ All tests passed! Enhanced training system is ready.")
print("=" * 60)

print("\n🚀 Quick start:")
print("  python scripts/train_naive_enhanced.py --config config/agents_naive/qlearning.yaml")
print("\n📊 After training, analyze results:")
print("  python scripts/analyze_agents.py --data-dir data --generate-html")
