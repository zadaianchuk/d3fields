#!/usr/bin/env python3
"""
Concise D3Fields data loader for AdaManip dataset.
Loads data and saves in D3Fields format with proper testing.
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import shutil


class D3FieldsDataLoader:
    """Streamlined data loader for AdaManip D3Fields dataset."""
    
    def __init__(self, base_path: str = "/ssdstore/azadaia/project_snellius_sync/d3fields/output/adamanip_d3fields"):
        self.base_path = Path(base_path)
        assert self.base_path.exists(), f"Base path does not exist: {base_path}"
        self.tasks = [d.name for d in self.base_path.iterdir() if d.is_dir()]
    
    def get_tasks(self) -> List[str]:
        """Get available tasks."""
        return self.tasks.copy()
    
    def get_environments(self, task: str) -> Dict[str, List[str]]:
        """Get environments for a task."""
        task_path = self.base_path / task
        assert task_path.exists(), f"Task not found: {task}"
        
        envs = {'grasp': [], 'manip': []}
        for env_dir in task_path.iterdir():
            if env_dir.is_dir():
                if env_dir.name.startswith('grasp_env_'):
                    envs['grasp'].append(env_dir.name)
                elif env_dir.name.startswith('manip_env_'):
                    envs['manip'].append(env_dir.name)
        
        for env_type in envs:
            envs[env_type].sort(key=lambda x: int(x.split('_')[-1]))
        
        return envs
    

    def load_datapoint(self, task: str, environment: str, frame_idx: int = 0) -> Dict:
        """Load a single datapoint."""
        env_path = self.base_path / task / environment
        assert env_path.exists(), f"Environment not found: {env_path}"
        
        # Load dataset info
        info_file = env_path / "dataset_info.txt"
        assert info_file.exists(), f"Dataset info not found: {info_file}"
        
        dataset_info = {}
        with open(info_file, 'r') as f:
            for line in f:
                if ':' in line and not line.startswith('='):
                    key, value = line.strip().split(':', 1)
                    if key.strip() == 'num_frames':
                        dataset_info['num_frames'] = int(value.strip())
                    elif key.strip() == 'num_cameras':
                        dataset_info['num_cameras'] = int(value.strip())
        
        assert frame_idx < dataset_info['num_frames'], f"Frame {frame_idx} >= {dataset_info['num_frames']}"
        
        # Load images and camera parameters
        num_cameras = dataset_info['num_cameras']
        color_images = []
        depth_images = []
        camera_params = []
        
        for cam_id in range(num_cameras):
            cam_path = env_path / f"camera_{cam_id}"
            
            # Load color image
            color_path = cam_path / "color" / f"{frame_idx}.png"
            assert color_path.exists(), f"Color image not found: {color_path}"
            color_img = cv2.imread(str(color_path))
            color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            color_images.append(color_img)
            
            # Load depth image
            depth_path = cam_path / "depth" / f"{frame_idx}.png"
            assert depth_path.exists(), f"Depth image not found: {depth_path}"
            depth_img = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
            depth_img = depth_img.astype(np.float32) / 1000.0  # Convert to meters
            depth_images.append(depth_img)
            
            # Load camera parameters
            params_file = cam_path / "camera_params.npy"
            extrinsics_file = cam_path / "camera_extrinsics.npy"
            assert params_file.exists(), f"Camera params not found: {params_file}"
            assert extrinsics_file.exists(), f"Camera extrinsics not found: {extrinsics_file}"
            
            params = np.load(params_file)  # [fx, fy, cx, cy]
            extrinsics = np.load(extrinsics_file)  # 4x4 world-to-camera
            assert extrinsics.shape == (4, 4), f"Extrinsics shape wrong: {extrinsics.shape}"
            
            intrinsics = np.array([
                [params[0], 0, params[2]],
                [0, params[1], params[3]],
                [0, 0, 1]
            ])
            
            camera_params.append({
                'intrinsics': intrinsics,
                'extrinsics': extrinsics,
                'params': params
            })
        
        return {
            'task': task,
            'environment': environment,
            'frame_idx': frame_idx,
            'dataset_info': dataset_info,
            'color_images': np.array(color_images),  # (num_cameras, H, W, 3)
            'depth_images': np.array(depth_images),  # (num_cameras, H, W)
            'camera_params': camera_params
        }
    
    def save_d3fields_format(self, datapoint: Dict, output_path: str):
        """Save datapoint in D3Fields format."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        num_cameras = len(datapoint['camera_params'])
        
        for cam_id in range(num_cameras):
            cam_dir = output_path / f"camera_{cam_id}"
            color_dir = cam_dir / "color"
            depth_dir = cam_dir / "depth"
            
            color_dir.mkdir(parents=True, exist_ok=True)
            depth_dir.mkdir(parents=True, exist_ok=True)
            
            # Save color image
            color_img = datapoint['color_images'][cam_id]
            color_bgr = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(color_dir / "0.png"), color_bgr)
            
            # Save depth image
            depth_img = datapoint['depth_images'][cam_id]
            depth_mm = (depth_img * 1000).astype(np.uint16)  # Convert to millimeters
            cv2.imwrite(str(depth_dir / "0.png"), depth_mm)
            
            # Save camera parameters
            cam_params = datapoint['camera_params'][cam_id]
            np.save(cam_dir / "camera_params.npy", cam_params['params'])
            np.save(cam_dir / "camera_extrinsics.npy", cam_params['extrinsics'])
        
        # Save metadata
        metadata = {
            'task': datapoint['task'],
            'environment': datapoint['environment'],
            'frame_idx': datapoint['frame_idx'],
            'num_cameras': num_cameras,
            'num_frames': 1  # Single frame
        }
        
        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved D3Fields format data to: {output_path}")


def convert_to_fusion_format(datapoint: Dict) -> Dict:
    """Convert datapoint to format expected by Fusion class."""
    color = datapoint['color_images']  # (num_cameras, H, W, 3)
    depth = datapoint['depth_images']  # (num_cameras, H, W)
    
    K_matrices = []
    pose_matrices = []
    correct_pose_matrices = []
    for cam_params in datapoint['camera_params']:
        K_matrices.append(cam_params['intrinsics'])
        # Convert world-to-camera to camera-to-world and extract 3x4
        extrinsics_c2w = np.linalg.inv(cam_params['extrinsics']).T
        assert np.allclose(extrinsics_c2w[-1], np.array([0,0,0,1]), rtol=1e-5, atol=1e-5)
        pose_matrices.append(extrinsics_c2w[:3, :])
        correct_pose_matrices.append((cam_params['extrinsics'].T)[:3, :])
    
    return {
        'color': color,
        'depth': depth,
        'K': np.array(K_matrices),
        'pose': np.array(pose_matrices),
        "correct_pose": np.array(correct_pose_matrices)
    }


def run_tests():
    """Run comprehensive tests with assertions."""
    print("Running D3Fields Data Loader Tests")
    print("=" * 50)
    
    # Test initialization
    loader = D3FieldsDataLoader()
    tasks = loader.get_tasks()
    assert len(tasks) > 0, "No tasks found!"
    print(f"✓ Found {len(tasks)} tasks: {tasks}")
    
    # Test environment loading
    task = tasks[0]
    environments = loader.get_environments(task)
    assert 'grasp' in environments or 'manip' in environments, "No environments found!"
    print(f"✓ Task {task} has environments: {environments}")
    
    # Test datapoint loading
    if environments['grasp']:
        env = environments['grasp'][0]
    else:
        env = environments['manip'][0]
    
    datapoint = loader.load_datapoint(task, env, frame_idx=0)
    
    # Validate datapoint structure
    assert 'color_images' in datapoint, "Missing color images"
    assert 'depth_images' in datapoint, "Missing depth images"
    assert 'camera_params' in datapoint, "Missing camera parameters"
    
    color_shape = datapoint['color_images'].shape
    depth_shape = datapoint['depth_images'].shape
    num_cameras = len(datapoint['camera_params'])
    
    assert len(color_shape) == 4, f"Color shape should be 4D, got {color_shape}"
    assert len(depth_shape) == 3, f"Depth shape should be 3D, got {depth_shape}"
    assert color_shape[0] == num_cameras, f"Color cameras mismatch: {color_shape[0]} vs {num_cameras}"
    assert depth_shape[0] == num_cameras, f"Depth cameras mismatch: {depth_shape[0]} vs {num_cameras}"
    assert color_shape[1:3] == depth_shape[1:3], f"Image size mismatch: {color_shape[1:3]} vs {depth_shape[1:3]}"
    
    print(f"✓ Loaded datapoint: {color_shape} color, {depth_shape} depth, {num_cameras} cameras")
    
    # Test camera parameters
    for i, cam_params in enumerate(datapoint['camera_params']):
        assert 'intrinsics' in cam_params, f"Camera {i} missing intrinsics"
        assert 'extrinsics' in cam_params, f"Camera {i} missing extrinsics"
        assert 'params' in cam_params, f"Camera {i} missing params"
        
        intrinsics = cam_params['intrinsics']
        extrinsics = cam_params['extrinsics']
        params = cam_params['params']
        
        assert intrinsics.shape == (3, 3), f"Intrinsics shape wrong: {intrinsics.shape}"
        assert extrinsics.shape == (4, 4), f"Extrinsics shape wrong: {extrinsics.shape}"
        assert len(params) == 4, f"Params length wrong: {len(params)}"
        
        # Validate intrinsics matrix structure
        assert intrinsics[0, 1] == 0, "Intrinsics should have 0 skew"
        assert intrinsics[1, 0] == 0, "Intrinsics should have 0 skew"
        assert intrinsics[2, 2] == 1, "Intrinsics bottom-right should be 1"
        
    print(f"✓ Camera parameters validated for {num_cameras} cameras")
    
    # Test fusion format conversion
    obs = convert_to_fusion_format(datapoint)
    assert 'color' in obs and 'depth' in obs and 'K' in obs and 'pose' in obs, "Missing fusion format keys"
    assert obs['color'].shape == color_shape, "Fusion color shape mismatch"
    assert obs['depth'].shape == depth_shape, "Fusion depth shape mismatch"
    assert obs['K'].shape == (num_cameras, 3, 3), f"Fusion K shape wrong: {obs['K'].shape}"
    assert obs['pose'].shape == (num_cameras, 3, 4), f"Fusion pose shape wrong: {obs['pose'].shape}"
    assert obs['correct_pose'].shape == (3, 4), f"Fusion correct pose shape wrong: {obs['correct_pose'].shape}"
    
    print("✓ Fusion format conversion validated")
    
    # Test saving D3Fields format
    test_output = Path("test_d3fields_output")
    if test_output.exists():
        shutil.rmtree(test_output)
    
    loader.save_d3fields_format(datapoint, test_output)
    
    # Validate saved files
    assert test_output.exists(), "Output directory not created"
    assert (test_output / "metadata.json").exists(), "Metadata file not created"
    
    for cam_id in range(num_cameras):
        cam_dir = test_output / f"camera_{cam_id}"
        assert cam_dir.exists(), f"Camera {cam_id} directory not created"
        assert (cam_dir / "color" / "0.png").exists(), f"Camera {cam_id} color image not saved"
        assert (cam_dir / "depth" / "0.png").exists(), f"Camera {cam_id} depth image not saved"
        assert (cam_dir / "camera_params.npy").exists(), f"Camera {cam_id} params not saved"
        assert (cam_dir / "camera_extrinsics.npy").exists(), f"Camera {cam_id} extrinsics not saved"
    
    print("✓ D3Fields format saving validated")
    
    # Test loading saved data
    saved_color = cv2.imread(str(test_output / "camera_0" / "color" / "0.png"))
    saved_depth = cv2.imread(str(test_output / "camera_0" / "depth" / "0.png"), cv2.IMREAD_ANYDEPTH)
    saved_params = np.load(test_output / "camera_0" / "camera_params.npy")
    saved_extrinsics = np.load(test_output / "camera_0" / "camera_extrinsics.npy")
    
    assert saved_color is not None, "Failed to load saved color image"
    assert saved_depth is not None, "Failed to load saved depth image"
    assert saved_params.shape == (4,), f"Saved params shape wrong: {saved_params.shape}"
    assert saved_extrinsics.shape == (4, 4), f"Saved extrinsics shape wrong: {saved_extrinsics.shape}"
    
    print("✓ Saved data can be loaded back")
    
    # Clean up
    shutil.rmtree(test_output)
    
    print("\n" + "=" * 50)
    print("🎉 All tests passed!")
    print("=" * 50)
    
    return datapoint


def batch_convert_to_d3fields(output_base: str = "d3fields_converted", 
                             max_per_task: int = 2, 
                             tasks: Optional[List[str]] = None):
    """Convert multiple environments to D3Fields format."""
    print(f"Batch converting to D3Fields format in: {output_base}")
    
    loader = D3FieldsDataLoader()
    available_tasks = loader.get_tasks()
    
    if tasks is None:
        tasks = available_tasks
    else:
        tasks = [t for t in tasks if t in available_tasks]
    
    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    
    converted_count = 0
    
    for task in tasks:
        print(f"\nProcessing task: {task}")
        environments = loader.get_environments(task)
        
        task_count = 0
        for env_type in ['grasp', 'manip']:
            for env in environments.get(env_type, []):
                if task_count >= max_per_task:
                    break
                
                try:
                    datapoint = loader.load_datapoint(task, env, frame_idx=0)
                    output_path = output_base / task / env
                    loader.save_d3fields_format(datapoint, output_path)
                    converted_count += 1
                    task_count += 1
                    print(f"  ✓ Converted {env}")
                    
                except Exception as e:
                    print(f"  ✗ Failed to convert {env}: {e}")
    
    print(f"\nConversion complete! Converted {converted_count} environments")
    print(f"Output saved to: {output_base}")


if __name__ == "__main__":
    # Run tests
    datapoint = run_tests()
    
    # Optional: Batch convert some environments
    print("\nRunning batch conversion...")
    batch_convert_to_d3fields(max_per_task=1) 