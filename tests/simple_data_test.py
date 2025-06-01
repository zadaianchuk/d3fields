#!/usr/bin/env python3
"""
Simple test of the AdaManip D3Fields data loader without PyTorch dependencies.
"""

import numpy as np
from data_loader import AdaManipD3FieldsLoader

def test_data_loading():
    """Test basic data loading functionality."""
    print("Testing AdaManip D3Fields Data Loader")
    print("=" * 50)
    
    # Initialize loader
    loader = AdaManipD3FieldsLoader()
    
    # Get available tasks
    tasks = loader.get_available_tasks()
    print(f"Available tasks: {tasks}")
    
    if not tasks:
        print("No tasks found!")
        return
    
    # Test with first available task
    task = tasks[0]
    print(f"\nTesting with task: {task}")
    
    # Get environments
    environments = loader.get_environments_for_task(task)
    print(f"Environments: {environments}")
    
    if environments['grasp']:
        env = environments['grasp'][0]
        print(f"\nLoading environment: {env}")
        
        # Load dataset info
        dataset_info = loader.load_dataset_info(task, env)
        print(f"Dataset info: {dataset_info}")
        
        # Load first frame images
        images = loader.load_images(task, env, frame_indices=0)
        print(f"\nImage shapes:")
        for key, data in images.items():
            print(f"  {key}: {data.shape}, dtype: {data.dtype}")
            if key == 'color':
                print(f"    Color range: {data.min()} - {data.max()}")
            elif key == 'depth':
                print(f"    Depth range: {data.min():.3f} - {data.max():.3f} meters")
        
        # Load camera parameters
        for cam_id in range(dataset_info['num_cameras']):
            cam_params = loader.load_camera_parameters(task, env, cam_id)
            print(f"\nCamera {cam_id} parameters:")
            print(f"  Intrinsics:\n{cam_params['intrinsics']}")
            print(f"  Extrinsics shape: {cam_params['extrinsics'].shape}")
            print(f"  Raw params [fx, fy, cx, cy]: {cam_params['params']}")
        
        # Load auxiliary data
        aux_data = loader.load_auxiliary_data(task, env)
        print(f"\nAuxiliary data:")
        for key, data in aux_data.items():
            print(f"  {key}: {data.shape}, dtype: {data.dtype}")
            if key == 'action':
                print(f"    Action range: {data.min():.3f} - {data.max():.3f}")
            elif key == 'point_clouds':
                print(f"    Point cloud range: {data.min():.3f} - {data.max():.3f}")
        
        # Test complete datapoint loading
        print(f"\nLoading complete datapoint...")
        datapoint = loader.load_complete_datapoint(task, env, frame_idx=0)
        print(f"Complete datapoint keys: {list(datapoint.keys())}")
        
        return datapoint
    
    return None

def analyze_dataset_statistics():
    """Analyze basic statistics across the dataset."""
    print("\n" + "=" * 50)
    print("Dataset Statistics Analysis")
    print("=" * 50)
    
    loader = AdaManipD3FieldsLoader()
    tasks = loader.get_available_tasks()
    
    total_envs = 0
    total_frames = 0
    
    for task in tasks:
        environments = loader.get_environments_for_task(task)
        task_envs = len(environments['grasp']) + len(environments['manip'])
        total_envs += task_envs
        
        print(f"\n{task}:")
        print(f"  Grasp environments: {len(environments['grasp'])}")
        print(f"  Manip environments: {len(environments['manip'])}")
        
        # Sample one environment to get frame count
        if environments['grasp']:
            sample_env = environments['grasp'][0]
            try:
                dataset_info = loader.load_dataset_info(task, sample_env)
                frames_per_env = dataset_info['num_frames']
                total_frames += frames_per_env * task_envs
                print(f"  Frames per environment: {frames_per_env}")
                print(f"  Total frames for task: {frames_per_env * task_envs}")
            except Exception as e:
                print(f"  Error loading sample environment: {e}")
    
    print(f"\nOverall Statistics:")
    print(f"  Total tasks: {len(tasks)}")
    print(f"  Total environments: {total_envs}")
    print(f"  Estimated total frames: {total_frames}")

def demonstrate_iteration():
    """Demonstrate different iteration patterns."""
    print("\n" + "=" * 50)
    print("Iteration Patterns Demo")
    print("=" * 50)
    
    loader = AdaManipD3FieldsLoader()
    
    # Episode-wise iteration (first 3)
    print("Episode-wise iteration (first 3):")
    count = 0
    for datapoint in loader.get_datapoint_iterator(frame_wise=False):
        count += 1
        task = datapoint['task']
        env = datapoint['environment']
        frames = datapoint['images']['color'].shape[0]
        print(f"  {count}. {task}/{env} - {frames} frames")
        
        if count >= 3:
            break
    
    # Frame-wise iteration (first 5)
    print(f"\nFrame-wise iteration (first 5):")
    count = 0
    for datapoint in loader.get_datapoint_iterator(frame_wise=True):
        count += 1
        task = datapoint['task']
        env = datapoint['environment']
        frame_idx = datapoint['frame_idx']
        print(f"  {count}. {task}/{env} - frame {frame_idx}")
        
        if count >= 5:
            break
    
    # Filtered iteration
    print(f"\nFiltered iteration (OpenBottle grasp only):")
    count = 0
    for datapoint in loader.get_datapoint_iterator(
        tasks=['OpenBottle'], 
        env_types=['grasp'], 
        frame_wise=False
    ):
        count += 1
        task = datapoint['task']
        env = datapoint['environment']
        print(f"  {count}. {task}/{env}")

def save_sample_data_info(datapoint, filename='sample_data_info.txt'):
    """Save detailed information about a sample datapoint."""
    if datapoint is None:
        return
    
    with open(filename, 'w') as f:
        f.write("AdaManip D3Fields Sample Datapoint Information\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Task: {datapoint['task']}\n")
        f.write(f"Environment: {datapoint['environment']}\n")
        f.write(f"Frame index: {datapoint['frame_idx']}\n\n")
        
        f.write("Dataset Info:\n")
        for key, value in datapoint['dataset_info'].items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        f.write("Image Data:\n")
        for key, data in datapoint['images'].items():
            f.write(f"  {key}: shape={data.shape}, dtype={data.dtype}\n")
            f.write(f"    min={data.min()}, max={data.max()}\n")
        f.write("\n")
        
        f.write("Camera Parameters:\n")
        for cam_key, cam_data in datapoint['camera_params'].items():
            f.write(f"  {cam_key}:\n")
            f.write(f"    Intrinsics shape: {cam_data['intrinsics'].shape}\n")
            f.write(f"    Extrinsics shape: {cam_data['extrinsics'].shape}\n")
            f.write(f"    Raw params: {cam_data['params']}\n")
        f.write("\n")
        
        f.write("Auxiliary Data:\n")
        for key, data in datapoint['auxiliary_data'].items():
            f.write(f"  {key}: shape={data.shape}, dtype={data.dtype}\n")
            f.write(f"    min={data.min()}, max={data.max()}\n")
        f.write("\n")
    
    print(f"Saved detailed datapoint information to {filename}")

def main():
    """Main test function."""
    try:
        # Test basic loading
        datapoint = test_data_loading()
        
        # Analyze dataset
        analyze_dataset_statistics()
        
        # Demonstrate iteration
        demonstrate_iteration()
        
        # Save sample info
        save_sample_data_info(datapoint)
        
        print("\n" + "=" * 50)
        print("✅ All tests completed successfully!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 