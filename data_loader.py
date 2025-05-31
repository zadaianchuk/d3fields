import os
import json
import numpy as np
from PIL import Image
import cv2
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import glob

class AdaManipD3FieldsLoader:
    """
    Data loader for AdaManip D3Fields dataset stored in the output directory.
    Supports loading RGB images, depth images, camera parameters, and metadata
    for various manipulation tasks and environments.
    """
    
    def __init__(self, base_path: str = "/ssdstore/azadaia/project_snellius_sync/d3fields/output/adamanip_d3fields"):
        """
        Initialize the data loader.
        
        Args:
            base_path: Path to the adamanip_d3fields output directory
        """
        self.base_path = Path(base_path)
        if not self.base_path.exists():
            raise FileNotFoundError(f"Base path does not exist: {base_path}")
        
        self.tasks = self._discover_tasks()
        
    def _discover_tasks(self) -> List[str]:
        """Discover available tasks in the dataset."""
        tasks = []
        for task_dir in self.base_path.iterdir():
            if task_dir.is_dir():
                tasks.append(task_dir.name)
        return sorted(tasks)
    
    def get_available_tasks(self) -> List[str]:
        """Get list of available tasks."""
        return self.tasks.copy()
    
    def get_environments_for_task(self, task: str) -> Dict[str, List[str]]:
        """
        Get available environments for a specific task.
        
        Args:
            task: Task name (e.g., 'OpenBottle', 'OpenSafe')
            
        Returns:
            Dictionary with 'grasp' and 'manip' environment lists
        """
        task_path = self.base_path / task
        if not task_path.exists():
            raise ValueError(f"Task '{task}' not found. Available tasks: {self.tasks}")
        
        environments = {'grasp': [], 'manip': []}
        
        for env_dir in task_path.iterdir():
            if env_dir.is_dir():
                if env_dir.name.startswith('grasp_env_'):
                    environments['grasp'].append(env_dir.name)
                elif env_dir.name.startswith('manip_env_'):
                    environments['manip'].append(env_dir.name)
        
        # Sort by environment number
        for env_type in environments:
            environments[env_type].sort(key=lambda x: int(x.split('_')[-1]))
        
        return environments
    
    def load_dataset_info(self, task: str, environment: str) -> Dict:
        """
        Load dataset information for a specific task and environment.
        
        Args:
            task: Task name
            environment: Environment name (e.g., 'grasp_env_0')
            
        Returns:
            Dictionary containing dataset metadata
        """
        env_path = self.base_path / task / environment
        info_file = env_path / "dataset_info.txt"
        
        if not info_file.exists():
            raise FileNotFoundError(f"Dataset info not found: {info_file}")
        
        info = {}
        with open(info_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if ':' in line and not line.startswith('='):
                    key, value = line.strip().split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Parse specific values
                    if key == 'num_frames':
                        info[key] = int(value)
                    elif key == 'num_cameras':
                        info[key] = int(value)
                    elif key == 'fixed_cameras_only':
                        info[key] = value.lower() == 'true'
                    elif key == 'fixed_camera_indices':
                        info[key] = eval(value)  # Parse list
                    else:
                        info[key] = value
        
        return info
    
    def load_camera_info(self, task: str, environment: str) -> List[List[Dict]]:
        """
        Load camera information for each frame.
        
        Args:
            task: Task name
            environment: Environment name
            
        Returns:
            List of camera info for each frame
        """
        env_path = self.base_path / task / environment
        camera_info_file = env_path / "camera_info.json"
        
        if not camera_info_file.exists():
            raise FileNotFoundError(f"Camera info not found: {camera_info_file}")
        
        with open(camera_info_file, 'r') as f:
            return json.load(f)
    
    def load_camera_parameters(self, task: str, environment: str, camera_id: int) -> Dict[str, np.ndarray]:
        """
        Load camera parameters (intrinsics and extrinsics) for a specific camera.
        
        Args:
            task: Task name
            environment: Environment name
            camera_id: Camera ID (0, 1, etc.)
            
        Returns:
            Dictionary with 'intrinsics' and 'extrinsics' arrays
        """
        camera_path = self.base_path / task / environment / f"camera_{camera_id}"
        
        # Load camera parameters (intrinsics)
        params_file = camera_path / "camera_params.npy"
        if params_file.exists():
            params = np.load(params_file)  # [fx, fy, cx, cy]
            intrinsics = np.array([
                [params[0], 0, params[2]],
                [0, params[1], params[3]],
                [0, 0, 1]
            ])
        else:
            raise FileNotFoundError(f"Camera parameters not found: {params_file}")
        
        # Load camera extrinsics
        extrinsics_file = camera_path / "camera_extrinsics.npy"
        if extrinsics_file.exists():
            extrinsics = np.load(extrinsics_file)  # 4x4 world-to-camera matrix
        else:
            raise FileNotFoundError(f"Camera extrinsics not found: {extrinsics_file}")
        
        return {
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'params': params  # [fx, fy, cx, cy]
        }
    
    def load_global_camera_parameters(self, task: str, environment: str) -> Dict[str, np.ndarray]:
        """
        Load global camera parameters for all cameras and frames.
        
        Args:
            task: Task name
            environment: Environment name
            
        Returns:
            Dictionary with global camera data
        """
        env_path = self.base_path / task / environment
        
        result = {}
        
        # Load global intrinsics if available
        intrinsics_file = env_path / "camera_intrinsics_all_steps.npy"
        if intrinsics_file.exists():
            result['intrinsics_all_steps'] = np.load(intrinsics_file)
        
        # Load global extrinsics if available
        extrinsics_file = env_path / "camera_extrinsics_all_steps.npy"
        if extrinsics_file.exists():
            result['extrinsics_all_steps'] = np.load(extrinsics_file)
        
        return result
    
    def load_images(self, 
                   task: str, 
                   environment: str, 
                   frame_indices: Optional[Union[int, List[int]]] = None,
                   camera_ids: Optional[Union[int, List[int]]] = None,
                   load_depth: bool = True,
                   load_color: bool = True) -> Dict[str, np.ndarray]:
        """
        Load RGB and/or depth images for specified frames and cameras.
        
        Args:
            task: Task name
            environment: Environment name
            frame_indices: Frame index or list of frame indices. If None, load all frames.
            camera_ids: Camera ID or list of camera IDs. If None, load all cameras.
            load_depth: Whether to load depth images
            load_color: Whether to load color images
            
        Returns:
            Dictionary with 'color' and/or 'depth' arrays of shape (num_frames, num_cameras, H, W, 3/1)
        """
        env_path = self.base_path / task / environment
        
        # Get dataset info to determine available frames and cameras
        dataset_info = self.load_dataset_info(task, environment)
        num_frames = dataset_info['num_frames']
        num_cameras = dataset_info['num_cameras']
        
        # Handle frame indices
        if frame_indices is None:
            frame_indices = list(range(num_frames))
        elif isinstance(frame_indices, int):
            frame_indices = [frame_indices]
        
        # Handle camera IDs
        if camera_ids is None:
            camera_ids = list(range(num_cameras))
        elif isinstance(camera_ids, int):
            camera_ids = [camera_ids]
        
        result = {}
        
        # Load color images
        if load_color:
            color_images = []
            for frame_idx in frame_indices:
                frame_cameras = []
                for camera_id in camera_ids:
                    color_path = env_path / f"camera_{camera_id}" / "color" / f"{frame_idx}.png"
                    if color_path.exists():
                        img = cv2.imread(str(color_path))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        frame_cameras.append(img)
                    else:
                        raise FileNotFoundError(f"Color image not found: {color_path}")
                color_images.append(frame_cameras)
            result['color'] = np.array(color_images)
        
        # Load depth images
        if load_depth:
            depth_images = []
            for frame_idx in frame_indices:
                frame_cameras = []
                for camera_id in camera_ids:
                    depth_path = env_path / f"camera_{camera_id}" / "depth" / f"{frame_idx}.png"
                    if depth_path.exists():
                        # Load depth as 16-bit PNG and convert to meters
                        depth_img = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
                        # Convert from millimeters to meters (typical for depth data)
                        depth_img = depth_img.astype(np.float32) / 1000.0
                        frame_cameras.append(depth_img)
                    else:
                        raise FileNotFoundError(f"Depth image not found: {depth_path}")
                depth_images.append(frame_cameras)
            result['depth'] = np.array(depth_images)
        
        return result
    
    def load_auxiliary_data(self, task: str, environment: str) -> Dict[str, np.ndarray]:
        """
        Load auxiliary data like actions, environment state, point clouds, etc.
        
        Args:
            task: Task name
            environment: Environment name
            
        Returns:
            Dictionary with available auxiliary data
        """
        env_path = self.base_path / task / environment
        result = {}
        
        # Load action data
        action_file = env_path / "action.npy"
        if action_file.exists():
            result['action'] = np.load(action_file)
        
        # Load environment state
        env_state_file = env_path / "env_state.npy"
        if env_state_file.exists():
            result['env_state'] = np.load(env_state_file)
        
        # Load point clouds
        pcs_file = env_path / "pcs.npy"
        if pcs_file.exists():
            result['point_clouds'] = np.load(pcs_file)
        
        # Load episode ends
        episode_ends_file = env_path / "episode_ends.npy"
        if episode_ends_file.exists():
            result['episode_ends'] = np.load(episode_ends_file)
        
        return result
    
    def load_complete_datapoint(self, 
                               task: str, 
                               environment: str,
                               frame_idx: Optional[int] = None) -> Dict:
        """
        Load a complete datapoint including images, camera parameters, and auxiliary data.
        
        Args:
            task: Task name
            environment: Environment name
            frame_idx: Specific frame to load. If None, loads all frames.
            
        Returns:
            Complete datapoint dictionary
        """
        # Load dataset info
        dataset_info = self.load_dataset_info(task, environment)
        num_cameras = dataset_info['num_cameras']
        
        # Load images
        images = self.load_images(task, environment, frame_indices=frame_idx)
        
        # Load camera parameters for all cameras
        camera_params = {}
        for camera_id in range(num_cameras):
            camera_params[f'camera_{camera_id}'] = self.load_camera_parameters(task, environment, camera_id)
        
        # Load global camera parameters
        global_params = self.load_global_camera_parameters(task, environment)
        
        # Load auxiliary data
        aux_data = self.load_auxiliary_data(task, environment)
        
        # Load camera info
        camera_info = self.load_camera_info(task, environment)
        
        return {
            'task': task,
            'environment': environment,
            'frame_idx': frame_idx,
            'dataset_info': dataset_info,
            'images': images,
            'camera_params': camera_params,
            'global_camera_params': global_params,
            'auxiliary_data': aux_data,
            'camera_info': camera_info
        }
    
    def get_datapoint_iterator(self, 
                              tasks: Optional[List[str]] = None,
                              env_types: Optional[List[str]] = None,
                              frame_wise: bool = False) -> Dict:
        """
        Create an iterator over datapoints in the dataset.
        
        Args:
            tasks: List of tasks to include. If None, include all tasks.
            env_types: List of environment types ('grasp', 'manip'). If None, include all.
            frame_wise: If True, yield individual frames. If False, yield entire episodes.
            
        Yields:
            Datapoint dictionaries
        """
        if tasks is None:
            tasks = self.tasks
        
        if env_types is None:
            env_types = ['grasp', 'manip']
        
        for task in tasks:
            if task not in self.tasks:
                continue
                
            environments = self.get_environments_for_task(task)
            
            for env_type in env_types:
                for environment in environments.get(env_type, []):
                    try:
                        if frame_wise:
                            dataset_info = self.load_dataset_info(task, environment)
                            num_frames = dataset_info['num_frames']
                            for frame_idx in range(num_frames):
                                yield self.load_complete_datapoint(task, environment, frame_idx)
                        else:
                            yield self.load_complete_datapoint(task, environment)
                    except Exception as e:
                        print(f"Error loading {task}/{environment}: {e}")
                        continue

def load_adamanip_datapoints(base_path: str = "/ssdstore/azadaia/project_snellius_sync/d3fields/output/adamanip_d3fields",
                           tasks: Optional[List[str]] = None,
                           env_types: Optional[List[str]] = None,
                           frame_wise: bool = False) -> List[Dict]:
    """
    Convenience function to load all datapoints from the AdaManip D3Fields dataset.
    
    Args:
        base_path: Path to the adamanip_d3fields directory
        tasks: List of specific tasks to load (e.g., ['OpenBottle', 'OpenSafe'])
        env_types: List of environment types to load ('grasp', 'manip')
        frame_wise: If True, return individual frames. If False, return entire episodes.
        
    Returns:
        List of datapoint dictionaries
    """
    loader = AdaManipD3FieldsLoader(base_path)
    datapoints = []
    
    for datapoint in loader.get_datapoint_iterator(tasks=tasks, env_types=env_types, frame_wise=frame_wise):
        datapoints.append(datapoint)
    
    return datapoints

# Example usage and utility functions
def example_usage():
    """Example of how to use the data loader."""
    
    # Initialize loader
    loader = AdaManipD3FieldsLoader()
    
    # Get available tasks
    print("Available tasks:", loader.get_available_tasks())
    
    # Get environments for a specific task
    environments = loader.get_environments_for_task('OpenBottle')
    print("OpenBottle environments:", environments)
    
    # Load a specific datapoint
    datapoint = loader.load_complete_datapoint('OpenBottle', 'grasp_env_0')
    print("Datapoint keys:", datapoint.keys())
    print("Image shapes:", {k: v.shape for k, v in datapoint['images'].items()})
    
    # Load only specific frames and cameras
    images = loader.load_images('OpenBottle', 'grasp_env_0', 
                               frame_indices=[0, 1], 
                               camera_ids=[0])
    print("Specific images shapes:", {k: v.shape for k, v in images.items()})
    
    # Iterate over all datapoints
    count = 0
    for datapoint in loader.get_datapoint_iterator(tasks=['OpenBottle'], 
                                                  env_types=['grasp'], 
                                                  frame_wise=True):
        count += 1
        if count >= 5:  # Just show first 5
            break
        print(f"Datapoint {count}: {datapoint['task']}/{datapoint['environment']}, frame {datapoint['frame_idx']}")

if __name__ == "__main__":
    example_usage() 