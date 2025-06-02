#!/usr/bin/env python3
"""
Example script demonstrating multi-frame D3Fields processing with scoring.
This script shows how to use the new all_frames functionality.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from main import main, get_frame_count, process_all_frames, compute_temporal_consistency
from d3fields_data_loader import D3FieldsDataLoader
import json


def example_single_task_multiframe():
    """Example: Process all frames for a single task with scoring."""
    print("🎬 Example: Multi-frame processing for OpenBottle task")
    print("=" * 60)
    
    # Process OpenBottle task with all frames
    main(
        output_dir="output/example_multiframe_results",
        max_envs_per_type=1,  # Process just 1 environment for demo
        env_types=['grasp'],   # Just grasp environments
        device='cuda:0',
        tasks=['OpenBottle'],
        use_gt_masks=True,
        process_all_frames_flag=True,
        max_frames=5,  # Limit to first 5 frames for demo
    )
    
    print("\n🎯 Results saved to: output/example_multiframe_results")
    print("📁 Check the all_frames/frame_x folders for individual frame results")
    print("📊 Check all_frames_results.json for scoring and temporal consistency")


def example_inspect_frame_counts():
    """Example: Inspect how many frames are available for different tasks/environments."""
    print("🔍 Inspecting frame counts across tasks and environments")
    print("=" * 60)
    
    loader = D3FieldsDataLoader()
    tasks = loader.get_tasks()[:3]  # First 3 tasks for demo
    
    frame_count_summary = {}
    
    for task in tasks:
        print(f"\n📋 Task: {task}")
        environments = loader.get_environments(task)
        task_summary = {}
        
        for env_type in ['grasp', 'manip']:
            if env_type in environments:
                envs = environments[env_type][:2]  # First 2 environments per type
                env_counts = {}
                
                for env in envs:
                    try:
                        frame_count = get_frame_count(task, env)
                        env_counts[env] = frame_count
                        print(f"  📁 {env_type}/{env}: {frame_count} frames")
                    except Exception as e:
                        print(f"  ❌ {env_type}/{env}: Error - {e}")
                        env_counts[env] = 0
                
                task_summary[env_type] = env_counts
        
        frame_count_summary[task] = task_summary
    
    # Save frame count summary
    output_path = Path("output/frame_count_summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(frame_count_summary, f, indent=2)
    
    print(f"\n💾 Frame count summary saved to: {output_path}")
    return frame_count_summary


def example_temporal_analysis():
    """Example: Analyze temporal consistency from previous results."""
    print("📈 Example: Temporal consistency analysis")
    print("=" * 60)
    
    # Look for existing results
    results_dir = Path("output/example_multiframe_results")
    
    if not results_dir.exists():
        print("❌ No previous results found. Run example_single_task_multiframe() first.")
        return
    
    # Find all_frames_results.json files
    results_files = list(results_dir.rglob("all_frames_results.json"))
    
    if not results_files:
        print("❌ No all_frames_results.json files found.")
        return
    
    print(f"📊 Found {len(results_files)} result files to analyze:")
    
    temporal_analysis = {}
    
    for results_file in results_files:
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            task = results['task']
            environment = results['environment']
            temporal_scores = results.get('temporal_scores', {})
            
            print(f"\n🎯 {task}/{environment}:")
            print(f"   📹 Total frames: {results['total_frames']}")
            print(f"   ✅ Successful: {results['successful_frames']}")
            print(f"   📊 Avg instances/frame: {results['avg_instances_per_frame']:.1f}")
            
            if temporal_scores:
                print(f"   🔄 Temporal consistency: {temporal_scores.get('temporal_consistency_score', 0):.3f}")
                print(f"   📈 Instance stability: {temporal_scores.get('instance_count_stability', 0):.3f}")
                print(f"   📊 Instance counts: {temporal_scores.get('instance_counts_over_time', [])}")
            
            temporal_analysis[f"{task}_{environment}"] = {
                'file_path': str(results_file),
                'temporal_scores': temporal_scores,
                'summary': {
                    'total_frames': results['total_frames'],
                    'successful_frames': results['successful_frames'],
                    'avg_instances_per_frame': results['avg_instances_per_frame']
                }
            }
            
        except Exception as e:
            print(f"❌ Error analyzing {results_file}: {e}")
    
    # Save temporal analysis
    analysis_path = Path("output/temporal_analysis_summary.json")
    with open(analysis_path, 'w') as f:
        json.dump(temporal_analysis, f, indent=2)
    
    print(f"\n💾 Temporal analysis saved to: {analysis_path}")
    return temporal_analysis


def example_compare_single_vs_multi():
    """Example: Compare single frame vs multi-frame processing."""
    print("⚖️  Example: Single frame vs Multi-frame comparison")
    print("=" * 60)
    
    # Single frame processing
    print("1️⃣  Running single frame processing...")
    main(
        output_dir="output/comparison_single_frame",
        max_envs_per_type=1,
        env_types=['grasp'],
        device='cuda:0',
        tasks=['OpenBottle'],
        use_gt_masks=True,
        process_all_frames_flag=False,  # Single frame
    )
    
    # Multi-frame processing
    print("\n2️⃣  Running multi-frame processing...")
    main(
        output_dir="output/comparison_multi_frame", 
        max_envs_per_type=1,
        env_types=['grasp'],
        device='cuda:0',
        tasks=['OpenBottle'],
        use_gt_masks=True,
        process_all_frames_flag=True,  # All frames
        max_frames=3,  # Limit for demo
    )
    
    print("\n📊 Comparison complete!")
    print("📁 Single frame results: output/comparison_single_frame")
    print("📁 Multi-frame results: output/comparison_multi_frame")


def main_example():
    """Main function to run all examples."""
    print("🚀 Multi-Frame D3Fields Processing Examples")
    print("=" * 80)
    
    examples = [
        ("1", "Inspect frame counts", example_inspect_frame_counts),
        ("2", "Single task multi-frame processing", example_single_task_multiframe),
        ("3", "Temporal consistency analysis", example_temporal_analysis),
        ("4", "Single vs Multi-frame comparison", example_compare_single_vs_multi),
    ]
    
    print("\nAvailable examples:")
    for num, desc, _ in examples:
        print(f"  {num}. {desc}")
    
    choice = input("\nEnter example number (1-4) or 'all' to run all: ").strip()
    
    if choice.lower() == 'all':
        for num, desc, func in examples:
            print(f"\n{'='*20} Running Example {num}: {desc} {'='*20}")
            try:
                func()
            except Exception as e:
                print(f"❌ Example {num} failed: {e}")
                import traceback
                traceback.print_exc()
    else:
        try:
            choice_num = int(choice)
            if 1 <= choice_num <= len(examples):
                num, desc, func = examples[choice_num - 1]
                print(f"\n{'='*20} Running Example {num}: {desc} {'='*20}")
                func()
            else:
                print("❌ Invalid choice. Please enter 1-4 or 'all'.")
        except ValueError:
            print("❌ Invalid input. Please enter a number 1-4 or 'all'.")


if __name__ == "__main__":
    main_example() 