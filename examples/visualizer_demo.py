#!/usr/bin/env python3
"""
Demonstration of the ROMA-DSPy Hierarchical Execution Visualizer.

This script shows how to use the visualizer to understand the hierarchical
decomposition process of the RecursiveSolver.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import dspy
from roma_dspy import RecursiveSolver
from roma_dspy.visualizer import (
    HierarchicalVisualizer,
    RealTimeVisualizer,
    TreeVisualizer,
    TimelineVisualizer,
    StatisticsVisualizer
)


def demo_realtime_visualization():
    """Demonstrate real-time visualization during execution."""
    print("\n" + "="*80)
    print("🎬 DEMO: Real-Time Execution Visualization")
    print("="*80)

    # Configure DSPy (using a mock LM for demo)
    lm = dspy.LM('openai/gpt-4o-mini')
    dspy.configure(lm=lm)

    # Create visualizer in real-time mode
    visualizer = HierarchicalVisualizer(mode="realtime", use_colors=True, verbose=True)

    # Create solver with visualizer
    solver = RecursiveSolver(
        max_depth=2,
        lm=lm,
        visualizer=visualizer,
        enable_logging=False
    )

    # Example task that will decompose hierarchically
    task = "Write a comprehensive tutorial on machine learning that covers supervised learning, unsupervised learning, and reinforcement learning with practical examples"

    print(f"\n📋 Task: {task}\n")
    print("Watch as the solver decomposes this task hierarchically...\n")

    # Execute with visualization
    result = solver.solve(task)

    print(f"\n✅ Final Result Preview: {str(result.result)[:200]}...")

    return solver


def demo_tree_visualization():
    """Demonstrate tree visualization of completed execution."""
    print("\n" + "="*80)
    print("🌳 DEMO: Hierarchical Tree Visualization")
    print("="*80)

    # Configure DSPy
    lm = dspy.LM('openai/gpt-4o-mini')
    dspy.configure(lm=lm)

    # Create solver without real-time visualization
    solver = RecursiveSolver(
        max_depth=3,
        lm=lm,
        enable_logging=False
    )

    # Complex task for deep hierarchy
    task = "Plan and organize a 3-day tech conference including venue selection, speaker lineup, workshop schedules, and attendee logistics"

    print(f"\n📋 Task: {task}\n")
    print("Executing task (without real-time visualization)...")

    # Execute task
    result = solver.solve(task)

    # Now visualize the tree
    tree_viz = TreeVisualizer(use_colors=True, show_ids=False, show_timing=True)
    tree_output = tree_viz.visualize(solver)

    print("\n" + tree_output)

    return solver


def demo_timeline_visualization():
    """Demonstrate timeline visualization of execution flow."""
    print("\n" + "="*80)
    print("⏱️  DEMO: Execution Timeline Visualization")
    print("="*80)

    # Configure DSPy
    lm = dspy.LM('openai/gpt-4o-mini')
    dspy.configure(lm=lm)

    # Create solver
    solver = RecursiveSolver(
        max_depth=2,
        lm=lm,
        enable_logging=False
    )

    # Task for timeline demo
    task = "Develop a mobile app: design UI, implement backend API, setup database, and create deployment pipeline"

    print(f"\n📋 Task: {task}\n")
    print("Executing task...")

    # Execute task
    result = solver.solve(task)

    # Visualize timeline
    timeline_viz = TimelineVisualizer(width=100)
    timeline_output = timeline_viz.visualize(solver)

    print("\n" + timeline_output)

    return solver


def demo_statistics_visualization():
    """Demonstrate statistics and metrics visualization."""
    print("\n" + "="*80)
    print("📊 DEMO: Execution Statistics Visualization")
    print("="*80)

    # Configure DSPy
    lm = dspy.LM('openai/gpt-4o-mini')
    dspy.configure(lm=lm)

    # Create solver
    solver = RecursiveSolver(
        max_depth=2,
        lm=lm,
        enable_logging=False
    )

    # Task for statistics demo
    task = "Analyze market trends, competitor analysis, and customer feedback to develop a product strategy"

    print(f"\n📋 Task: {task}\n")
    print("Executing task...")

    # Execute task
    result = solver.solve(task)

    # Visualize statistics
    stats_viz = StatisticsVisualizer()
    stats_output = stats_viz.visualize(solver)

    print("\n" + stats_output)

    return solver


def demo_combined_visualization():
    """Demonstrate all visualization modes combined."""
    print("\n" + "="*80)
    print("🎯 DEMO: Combined Visualization (All Modes)")
    print("="*80)

    # Configure DSPy
    lm = dspy.LM('openai/gpt-4o-mini')
    dspy.configure(lm=lm)

    # Create visualizer with all modes
    visualizer = HierarchicalVisualizer(mode="all", use_colors=True, verbose=False)

    # Create solver with visualizer
    solver = RecursiveSolver(
        max_depth=2,
        lm=lm,
        visualizer=visualizer,
        enable_logging=False
    )

    # Task for combined demo
    task = "Research and write a detailed report on climate change impacts, mitigation strategies, and policy recommendations"

    print(f"\n📋 Task: {task}\n")

    # Execute with real-time visualization
    result = solver.solve(task)

    # Show all visualization modes
    print("\n" + visualizer.visualize_execution(solver))

    return solver


def demo_export_capabilities():
    """Demonstrate export capabilities of the visualizer."""
    print("\n" + "="*80)
    print("💾 DEMO: Export Capabilities")
    print("="*80)

    # Configure DSPy
    lm = dspy.LM('openai/gpt-4o-mini')
    dspy.configure(lm=lm)

    # Create visualizer
    visualizer = HierarchicalVisualizer(mode="all", use_colors=True)

    # Create solver
    solver = RecursiveSolver(
        max_depth=2,
        lm=lm,
        visualizer=visualizer,
        enable_logging=False
    )

    # Simple task for export demo
    task = "Create a weekly meal plan with healthy recipes"

    print(f"\n📋 Task: {task}\n")
    print("Executing task...")

    # Execute task
    result = solver.solve(task)

    # Export to different formats
    print("\n📁 Exporting visualization to different formats...")

    # Export to HTML
    html_file = "execution_visualization.html"
    visualizer.export_to_html(solver, html_file)
    print(f"   ✅ HTML export: {html_file}")

    # Export to JSON
    json_file = "execution_data.json"
    visualizer.export_to_json(solver, json_file)
    print(f"   ✅ JSON export: {json_file}")

    return solver


async def demo_async_visualization():
    """Demonstrate visualization with async execution."""
    print("\n" + "="*80)
    print("⚡ DEMO: Async Execution Visualization")
    print("="*80)

    # Configure DSPy
    lm = dspy.LM('openai/gpt-4o-mini')
    dspy.configure(lm=lm)

    # Create visualizer
    visualizer = HierarchicalVisualizer(mode="realtime", use_colors=True, verbose=True)

    # Create solver
    solver = RecursiveSolver(
        max_depth=2,
        lm=lm,
        visualizer=visualizer,
        enable_logging=False
    )

    # Tasks for parallel execution
    tasks = [
        "Design a website homepage",
        "Write a blog post about AI",
        "Create a marketing strategy"
    ]

    print("\n📋 Running multiple tasks asynchronously:")
    for i, task in enumerate(tasks, 1):
        print(f"   {i}. {task}")

    print("\nExecuting tasks in parallel...\n")

    # Execute tasks asynchronously
    results = await asyncio.gather(*[
        solver.async_solve(task) for task in tasks
    ])

    print("\n✅ All tasks completed!")
    for i, result in enumerate(results, 1):
        print(f"   Task {i} result: {str(result.result)[:50]}...")

    return solver


def main():
    """Main function to run all demos."""
    print("\n" + "="*80)
    print("🚀 ROMA-DSPy HIERARCHICAL EXECUTION VISUALIZER")
    print("="*80)
    print("\nThis demo showcases different visualization modes for understanding")
    print("how the RecursiveSolver decomposes and executes complex tasks.\n")

    # Check for OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set. Using mock mode for demonstration.")
        print("   Set your API key to see real task decomposition:\n")
        print("   export OPENAI_API_KEY='your-api-key-here'\n")

    demos = [
        ("1", "Real-Time Visualization", demo_realtime_visualization),
        ("2", "Tree Visualization", demo_tree_visualization),
        ("3", "Timeline Visualization", demo_timeline_visualization),
        ("4", "Statistics Visualization", demo_statistics_visualization),
        ("5", "Combined Visualization", demo_combined_visualization),
        ("6", "Export Capabilities", demo_export_capabilities),
        ("7", "Async Execution", lambda: asyncio.run(demo_async_visualization())),
        ("8", "Run All Demos", None)
    ]

    print("Available demos:")
    for num, name, _ in demos[:-1]:
        print(f"  {num}. {name}")
    print(f"  {demos[-1][0]}. {demos[-1][1]}")
    print("  0. Exit")

    while True:
        choice = input("\nSelect a demo (0-8): ").strip()

        if choice == "0":
            print("\nGoodbye! 👋")
            break
        elif choice == "8":
            # Run all demos
            for num, name, func in demos[:-1]:
                if func:
                    try:
                        func()
                    except Exception as e:
                        print(f"\n❌ Error in {name}: {e}")
                    input("\nPress Enter to continue to next demo...")
        elif choice in [d[0] for d in demos[:-1]]:
            # Run selected demo
            for num, name, func in demos:
                if num == choice and func:
                    try:
                        func()
                    except Exception as e:
                        print(f"\n❌ Error: {e}")
                        import traceback
                        traceback.print_exc()
                    break
        else:
            print("Invalid choice. Please try again.")

        if choice != "0":
            input("\nPress Enter to return to menu...")


if __name__ == "__main__":
    main()
