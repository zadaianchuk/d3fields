#!/usr/bin/env python3
"""
Example usage of the Grounding SAM Demo Script

This script demonstrates various ways to use the grounding_sam_demo.py script
for processing the d3fields adamanip dataset, including the new dual camera
and all scenes functionality.
"""

import os
import subprocess
import sys


def run_command(cmd, description):
    """Run a command and handle the output"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
        if result.returncode != 0:
            print(f"Command failed with return code: {result.returncode}")
        return result.returncode == 0
    except Exception as e:
        print(f"Error running command: {e}")
        return False


def main():
    """Demonstrate various usage patterns of the Grounding SAM demo"""
    
    print("Grounding SAM Demo - Example Usage (Updated with Dual Camera)")
    print("=" * 70)
    
    # Check if the demo script exists
    demo_script = "grounding_sam_demo.py"
    if not os.path.exists(demo_script):
        print(f"Error: {demo_script} not found!")
        print("Make sure you're running this from the correct directory.")
        return
    
    # Example 1: List available scenes
    print("\n1. Listing available scenes in the dataset...")
    cmd1 = f"python {demo_script} --list_scenes"
    run_command(cmd1, "List available scenes")
    
    # Example 2: Process ALL scenes by default (NEW DEFAULT BEHAVIOR)
    print("\n2. Processing ALL scenes with dual camera view (NEW DEFAULT)...")
    cmd2 = f"python {demo_script} --output_dir ./all_scenes_results --no_display --max_envs_per_scene 1"
    run_command(cmd2, "Process all scenes with dual camera view")
    
    # Example 3: Process specific scene with dual camera view
    print("\n3. Processing specific scene with dual camera view...")
    cmd3 = f"python {demo_script} --scene OpenBottle --env_type grasp_env --env_id 0 --frame_id 1 --dual_camera --output_dir ./dual_camera_results --no_display"
    run_command(cmd3, "Process OpenBottle with dual camera")
    
    # Example 4: Traditional single camera processing
    print("\n4. Traditional single camera processing...")
    cmd4 = f"python {demo_script} --scene OpenDoor --env_type grasp_env --env_id 0 --camera_id 0 --frame_id 1 --output_dir ./single_camera_results --no_display"
    run_command(cmd4, "Process single camera (traditional mode)")
    
    # Example 5: Scene overview for OpenCoffeeMachine
    print("\n5. Creating scene overview for OpenCoffeeMachine...")
    cmd5 = f"python {demo_script} --scene OpenCoffeeMachine --scene_overview --no_display"
    run_command(cmd5, "OpenCoffeeMachine scene overview")
    
    # Example 6: Batch process with custom queries (single camera)
    print("\n6. Batch processing with custom queries...")
    cmd6 = f"python {demo_script} --scene OpenBottle --env_type grasp_env --env_id 0 --batch_process --queries bottle robotic_arm --max_frames 2 --output_dir ./batch_results --save_masks --no_display"
    run_command(cmd6, "Batch process with custom queries")
    
    # Example 7: Process all scenes with custom parameters
    print("\n7. Process all scenes with more environments per scene...")
    cmd7 = f"python {demo_script} --all_scenes --max_envs_per_scene 2 --frame_id 2 --output_dir ./extended_all_scenes --no_display"
    run_command(cmd7, "Process all scenes (extended)")
    
    print("\n" + "="*70)
    print("Example usage complete!")
    print("Check the following directories for output files:")
    print("  - ./all_scenes_results/     (Default: all scenes, dual camera)")
    print("  - ./dual_camera_results/    (Specific scene, dual camera)")
    print("  - ./single_camera_results/  (Traditional single camera)")
    print("  - ./batch_results/          (Batch processing)")
    print("  - ./extended_all_scenes/    (All scenes with more envs)")
    print("="*70)
    
    print("\nNEW FEATURES:")
    print("  🆕 Default behavior: Process ALL scenes with dual camera view")
    print("  🆕 --dual_camera: Show both cameras side by side for any scene")
    print("  🆕 --all_scenes: Explicitly process all scenes")
    print("  🆕 Dual camera visualizations show Camera 0 and Camera 1 in 2x4 grid")
    print("  🆕 Simplified default queries for better performance")


if __name__ == "__main__":
    main() 