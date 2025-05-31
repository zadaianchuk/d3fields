#!/usr/bin/env python3
"""
Simple utilities for working with D3Fields AdaManip data.
"""

from d3fields_data_loader import D3FieldsDataLoader, convert_to_fusion_format
from pathlib import Path
import numpy as np


def quick_load(task: str, environment: str = None, frame_idx: int = 0):
    """
    Quick load a datapoint.
    
    Args:
        task: Task name (e.g., 'OpenBottle')
        environment: Environment name (if None, uses first available grasp env)
        frame_idx: Frame index to load
    
    Returns:
        Datapoint dictionary
    """
    loader = D3FieldsDataLoader()
    
    if environment is None:
        environments = loader.get_environments(task)
        if environments['grasp']:
            environment = environments['grasp'][0]
        elif environments['manip']:
            environment = environments['manip'][0]
        else:
            raise ValueError(f"No environments found for task {task}")
    
    return loader.load_datapoint(task, environment, frame_idx)


def convert_and_save(task: str, environment: str, output_path: str, frame_idx: int = 0):
    """
    Convert and save a datapoint to D3Fields format.
    
    Args:
        task: Task name
        environment: Environment name
        output_path: Where to save the converted data
        frame_idx: Frame index to convert
    """
    loader = D3FieldsDataLoader()
    datapoint = loader.load_datapoint(task, environment, frame_idx)
    loader.save_d3fields_format(datapoint, output_path)
    print(f"Saved {task}/{environment} frame {frame_idx} to {output_path}")


def list_available():
    """List all available tasks and environments."""
    loader = D3FieldsDataLoader()
    tasks = loader.get_tasks()
    
    print("Available Tasks and Environments:")
    print("=" * 40)
    
    total_envs = 0
    for task in tasks:
        environments = loader.get_environments(task)
        grasp_count = len(environments['grasp'])
        manip_count = len(environments['manip'])
        total_count = grasp_count + manip_count
        total_envs += total_count
        
        print(f"{task}:")
        print(f"  Grasp: {grasp_count} environments")
        print(f"  Manip: {manip_count} environments")
        print(f"  Total: {total_count} environments")
        print()
    
    print(f"Total: {len(tasks)} tasks, {total_envs} environments")


def get_data_stats(task: str, environment: str = None):
    """Get statistics about the data."""
    datapoint = quick_load(task, environment)
    
    print(f"Data Statistics for {datapoint['task']}/{datapoint['environment']}:")
    print("=" * 50)
    print(f"Number of cameras: {len(datapoint['camera_params'])}")
    print(f"Color image shape: {datapoint['color_images'].shape}")
    print(f"Depth image shape: {datapoint['depth_images'].shape}")
    print(f"Color range: {datapoint['color_images'].min()}-{datapoint['color_images'].max()}")
    print(f"Depth range: {datapoint['depth_images'].min():.3f}-{datapoint['depth_images'].max():.3f}m")
    
    # Camera parameters
    for i, cam_params in enumerate(datapoint['camera_params']):
        params = cam_params['params']
        print(f"Camera {i}: fx={params[0]:.1f}, fy={params[1]:.1f}, cx={params[2]:.1f}, cy={params[3]:.1f}")


def batch_convert_all(output_base: str = "d3fields_all", frame_idx: int = 0):
    """Convert all available environments to D3Fields format."""
    loader = D3FieldsDataLoader()
    tasks = loader.get_tasks()
    
    output_base = Path(output_base)
    converted_count = 0
    failed_count = 0
    
    print(f"Converting all environments to: {output_base}")
    print("=" * 50)
    
    for task in tasks:
        print(f"\nProcessing {task}...")
        environments = loader.get_environments(task)
        
        for env_type in ['grasp', 'manip']:
            for env in environments.get(env_type, []):
                try:
                    datapoint = loader.load_datapoint(task, env, frame_idx)
                    output_path = output_base / task / env
                    loader.save_d3fields_format(datapoint, output_path)
                    converted_count += 1
                    print(f"  ✓ {env}")
                except Exception as e:
                    failed_count += 1
                    print(f"  ✗ {env}: {e}")
    
    print(f"\nConversion complete!")
    print(f"✓ Converted: {converted_count}")
    print(f"✗ Failed: {failed_count}")
    print(f"📁 Output: {output_base}")


def demo():
    """Run a quick demo."""
    print("D3Fields AdaManip Data Loader Demo")
    print("=" * 40)
    
    # List available data
    list_available()
    
    # Load and inspect a sample
    print("\nLoading sample data...")
    datapoint = quick_load('OpenBottle')
    get_data_stats('OpenBottle')
    
    # Convert to fusion format
    print("\nConverting to fusion format...")
    obs = convert_to_fusion_format(datapoint)
    print(f"✓ Fusion format: color {obs['color'].shape}, depth {obs['depth'].shape}")
    
    # Save sample
    print("\nSaving sample in D3Fields format...")
    convert_and_save('OpenBottle', 'grasp_env_0', 'demo_output')
    
    print("\n🎉 Demo complete!")


if __name__ == "__main__":
    demo() 