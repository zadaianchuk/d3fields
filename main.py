#!/usr/bin/env python3
"""
Main script to compute D3Fields for all objects in the AdaManip dataset.
Processes each task with appropriate text queries and saves results.
"""

import os
from pathlib import Path
import json
import traceback
import cv2
import numpy as np
import pickle
import torch
import trimesh
from sklearn.decomposition import PCA
from PIL import Image
from d3fields_data_loader import D3FieldsDataLoader, convert_to_fusion_format
from fusion import Fusion, instance2onehot, create_init_grid


# Task-specific queries for each manipulation task
TASK_QUERIES = {
    'OpenBottle': ['bottle', 'robotic arm'],
    'OpenDoor': ['door', 'robotic arm'],
    'OpenSafe': ['safe',  'robotic arm', ],
    'OpenCoffeeMachine': ['coffee machine', 'robotic arm', ],
    'OpenWindow': ['window',  'robotic arm', ],
    'OpenPressureCooker': ['pressure cooker', 'robotic arm',]
}

# Object type mapping from task names to object categories
OBJECT_TYPE_MAPPING = {
    'OpenBottle': 'bottle',
    'OpenDoor': 'door', 
    'OpenSafe': 'safe',
    'OpenCoffeeMachine': 'coffee_machine',
    'OpenWindow': 'window',
    'OpenPressureCooker': 'pressure_cooker',
    'OpenPen': 'pen',
    'OpenLamp': 'lamp',
    'OpenMicroWave': 'microwave'
}

# Default boundaries for the workspace (adjust based on your setup)
DEFAULT_BOUNDARIES = {
    'x_lower': -1, 'x_upper': 1,
    'y_lower': -1, 'y_upper': 1,
    'z_lower': 0.1, 'z_upper': 1
}

# Default segmentation thresholds
DEFAULT_THRESHOLDS = [0.25, 0.25]

def infer_boundaries_and_step(point_clouds: dict, padding_ratio: float = 0.1, target_grid_points: int = 500000):
    """
    Infer workspace boundaries and step size from extracted point clouds.
    
    Args:
        point_clouds: Dictionary of extracted point clouds {instance_name: numpy_array}
        padding_ratio: Ratio of padding to add around the point cloud bounds
        target_grid_points: Target number of grid points for step size calculation
    
    Returns:
        tuple: (boundaries_dict, step_size)
    """
    if not point_clouds:
        print("    Warning: No point clouds provided, using default boundaries")
        return DEFAULT_BOUNDARIES, 0.05
    
    # Combine all point clouds
    all_points = []
    for pcd_np in point_clouds.values():
        if len(pcd_np) > 0:
            all_points.append(pcd_np)
    
    if not all_points:
        print("    Warning: All point clouds are empty, using default boundaries")
        return DEFAULT_BOUNDARIES, 0.05
    
    combined_points = np.concatenate(all_points, axis=0)
    
    # Compute bounds
    min_coords = combined_points.min(axis=0)
    max_coords = combined_points.max(axis=0)
    
    # Add padding
    ranges = max_coords - min_coords
    padding = ranges * padding_ratio
    
    boundaries = {
        'x_lower': float(min_coords[0] - padding[0]),
        'x_upper': float(max_coords[0] + padding[0]),
        'y_lower': float(min_coords[1] - padding[1]),
        'y_upper': float(max_coords[1] + padding[1]),
        'z_lower': float(min_coords[2] - padding[2]),
        'z_upper': float(max_coords[2] + padding[2])
    }
    
    # Calculate step size based on target grid points
    volume = (boundaries['x_upper'] - boundaries['x_lower']) * \
             (boundaries['y_upper'] - boundaries['y_lower']) * \
             (boundaries['z_upper'] - boundaries['z_lower'])
    
    step_size = (volume / target_grid_points) ** (1/3)
    step_size = max(0.01, min(0.1, step_size))  # Clamp between 1cm and 10cm
    
    print(f"    Inferred boundaries: x=[{boundaries['x_lower']:.3f}, {boundaries['x_upper']:.3f}], "
          f"y=[{boundaries['y_lower']:.3f}, {boundaries['y_upper']:.3f}], "
          f"z=[{boundaries['z_lower']:.3f}, {boundaries['z_upper']:.3f}]")
    print(f"    Inferred step size: {step_size:.3f}m")
    
    return boundaries, step_size

# Base path for GT masks
GT_MASK_BASE_PATH = "/ssdstore/azadaia/project_snellius_sync/d3fields/output/adamanip_d3fields"


def load_gt_masks(task: str, environment: str, frame_idx: int, num_cameras: int):
    """
    Load GT masks from the adamanip_d3fields directory.
    
    Args:
        task: Task name (e.g., 'OpenBottle')
        environment: Environment name (e.g., 'grasp_env_0')
        frame_idx: Frame index to load
        num_cameras: Number of cameras
    
    Returns:
        dict: Contains 'masks' (list of per-camera masks) and 'labels' (list of labels)
    """
    masks = []
    labels = []
    
    for cam_id in range(num_cameras):
        mask_path = Path(GT_MASK_BASE_PATH) / task / environment / f"camera_{cam_id}" / "masks" / f"{frame_idx}.png"
        
        if not mask_path.exists():
            print(f"    Warning: GT mask not found at {mask_path}")
            # Create empty mask
            mask = np.zeros((512, 512), dtype=np.uint8)
        else:
            # Load mask
            mask = np.array(Image.open(mask_path))
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]  # Take first channel if RGB
            mask = mask.astype(np.uint8)
        
        # Get unique values (instance IDs)
        unique_vals = np.unique(mask)
        unique_vals = unique_vals[unique_vals > 0]  # Remove background (0)
        
        # Convert instance mask to list of binary masks for each instance
        instance_masks = []
        instance_labels = []
        
        for inst_id in unique_vals:
            binary_mask = (mask == inst_id).astype(bool)
            if binary_mask.sum() > 10:  # Only keep masks with sufficient pixels
                instance_masks.append(binary_mask)
                instance_labels.append(f'object_{inst_id}')
        
        masks.append(np.array(instance_masks) if instance_masks else np.zeros((0, mask.shape[0], mask.shape[1]), dtype=bool))
        labels.append(instance_labels)
    
    print(f"    Loaded GT masks: {[len(cam_labels) for cam_labels in labels]} instances per camera")
    
    return {
        'masks': masks,  # list of [num_instances, H, W] per camera
        'labels': labels  # list of label lists per camera
    }


def apply_gt_masks_to_fusion_simple(fusion, gt_masks):
    """
    Apply GT masks to the fusion system in a simplified way.
    
    Args:
        fusion: Fusion object
        gt_masks: GT mask data from load_gt_masks
    """
    import torch  # Import torch here when needed
    
    # Set the mask data in fusion's curr_obs_torch
    fusion.curr_obs_torch['mask_gs'] = gt_masks['masks']  # list of [num_obj, H, W]
    
    # Fix labels to include 'background' as first label
    fixed_labels = []
    for cam_labels in gt_masks['labels']:
        cam_fixed_labels = ['background'] + cam_labels  # Add background as first label
        fixed_labels.append(cam_fixed_labels)
    
    fusion.curr_obs_torch['mask_label'] = fixed_labels  # [num_cam, ] list of list
    
    # Create dummy confidence scores
    mask_confs = []
    for cam_masks in gt_masks['masks']:
        if len(cam_masks) > 0:
            mask_confs.append(np.ones(len(cam_masks)))  # High confidence for GT masks
        else:
            mask_confs.append(np.array([]))
    fusion.curr_obs_torch['mask_conf'] = mask_confs
    
    # Set semantic labels (take from first camera that has labels, including background)
    semantic_labels = []
    for cam_labels in fixed_labels:
        if len(cam_labels) > 1:  # More than just background
            semantic_labels = cam_labels
            break
    if not semantic_labels:
        semantic_labels = ['background']  # Fallback to just background
    
    fusion.curr_obs_torch['semantic_label'] = semantic_labels
    fusion.curr_obs_torch['consensus_mask_label'] = semantic_labels  # Set consensus labels directly
    
    print(f"    Applied GT masks with labels: {semantic_labels}")
    
    # Create the mask tensor directly instead of going through alignment
    # Convert GT masks to the expected format: [num_cam, H, W, num_instance]
    num_instances = len(semantic_labels)
    H, W = 512, 512  # Assuming 512x512 images
    
    mask_tensor = torch.zeros((fusion.num_cam, H, W, num_instances), dtype=torch.uint8, device=fusion.device)
    
    # Set background mask (instance 0) to areas not covered by other instances
    for cam_id in range(fusion.num_cam):
        background_mask = torch.ones((H, W), dtype=torch.bool, device=fusion.device)
        
        if len(gt_masks['masks'][cam_id]) > 0:
            for inst_id, inst_mask in enumerate(gt_masks['masks'][cam_id]):
                # Convert numpy mask to torch tensor
                inst_mask_torch = torch.from_numpy(inst_mask.astype(np.uint8)).to(fusion.device)
                mask_tensor[cam_id, :, :, inst_id + 1] = inst_mask_torch  # inst_id + 1 because background is 0
                background_mask = background_mask & (~inst_mask_torch.bool())
        
        # Set background mask
        mask_tensor[cam_id, :, :, 0] = background_mask.to(torch.uint8)
    
    # Convert to one-hot format and set the dtype
    fusion.curr_obs_torch['mask'] = mask_tensor.to(dtype=fusion.dtype)
    fusion.curr_obs_torch['mask_gs'] = mask_tensor.to(dtype=fusion.dtype).cpu().numpy()
    
    print(f"    Set mask tensor with shape: {mask_tensor.shape}")


def create_meshes_and_save(fusion, output_dir: Path, task: str, environment: str, point_clouds: dict = None, boundaries: dict = None, step_size: float = None):
    """
    Create and save 3D meshes (mask, feature, color) similar to vis_repr.py
    
    Args:
        fusion: Fusion object
        output_dir: Output directory
        task: Task name (empty string to use output_dir directly)
        environment: Environment name (empty string to use output_dir directly)
        point_clouds: Dictionary of extracted point clouds for boundary inference
        boundaries: Workspace boundaries (None to infer from data)
        step_size: Grid step size for mesh generation (None to infer from data)
    """
    print(f"    Creating 3D meshes...")
    
    # Infer boundaries and step size if not provided
    if boundaries is None or step_size is None:
        print("    Inferring boundaries and step size from point cloud data...")
        inferred_boundaries, inferred_step = infer_boundaries_and_step(point_clouds or {})
        boundaries = boundaries or inferred_boundaries
        step_size = step_size or inferred_step
    
    device = fusion.device
    
    # Create initial grid
    init_grid, grid_shape = create_init_grid(boundaries, step_size)
    init_grid = init_grid.to(device=device, dtype=torch.float32)
    
    print(f"    Evaluating grid with {len(init_grid)} points, grid shape: {grid_shape}")
    
    # Evaluate initial grid
    with torch.no_grad():
        out = fusion.batch_eval(init_grid, return_names=['dino_feats', 'mask', 'color_tensor'])
    
    print(f"    Grid evaluation output keys: {list(out.keys()) if isinstance(out, dict) else 'not a dict'}")
    
    # Extract mesh
    print("    Extracting mesh...")
    vertices, triangles = fusion.extract_mesh(init_grid, out, grid_shape)
    
    print(f"    Extracted {len(vertices)} vertices and {len(triangles)} triangles")
    
    if len(vertices) == 0 or len(triangles) == 0:
        raise ValueError(f"No mesh vertices/triangles extracted for {task}/{environment}" if task and environment else "No mesh vertices/triangles extracted")
    
    # Evaluate mesh vertices for features, masks, and colors
    vertices_tensor = torch.from_numpy(vertices).to(device, dtype=torch.float32)
    print(f"    Evaluating {len(vertices)} mesh vertices...")
    
    with torch.no_grad():
        vertex_out = fusion.batch_eval(vertices_tensor, return_names=['dino_feats', 'mask', 'color_tensor'])
    
    # Create output directory for meshes - if task/environment are empty, use output_dir directly
    if task and environment:
        mesh_output_dir = output_dir / task / environment
    else:
        mesh_output_dir = output_dir
    mesh_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create and save mask meshes (one for each instance)
    print("    Creating mask meshes...")
    mask_meshes = fusion.create_instance_mask_mesh(vertices, triangles, vertex_out)
    for i, mask_mesh in enumerate(mask_meshes):
        task_suffix = f"_{task.lower()}" if task else ""
        mask_path = mesh_output_dir / f"mask_mesh_{i}{task_suffix}.ply"
        mask_mesh.export(str(mask_path))
        print(f"    Saved mask mesh {i} to {mask_path}")

    # Create and save feature mesh
    print("    Creating feature mesh...")

    # Load or create a simple PCA model for features
    object_name = OBJECT_TYPE_MAPPING.get(task, task.lower()) if task else None
    pca_model = create_simple_pca_model(object_name)
    #save only vertices as ply
    # vertices_path = mesh_output_dir / f"vertices_{task.lower()}.ply"
    # o3d.io.write_point_cloud(str(vertices_path), o3d.geometry.PointCloud(vertices))
    
    # print(f"    Saved vertices to {vertices_path}")

    feature_mesh = fusion.create_descriptor_mesh(vertices, triangles, vertex_out, {'pca': pca_model}, mask_out_bg=True)
    task_suffix = f"_{task.lower()}" if task else ""
    feature_path = mesh_output_dir / f"feature_mesh{task_suffix}.ply"
    feature_mesh.export(str(feature_path))
    print(f"    Saved feature mesh to {feature_path}")
    
    color_mesh = fusion.create_color_mesh(vertices, triangles, vertex_out)
    color_path = mesh_output_dir / f"color_mesh{task_suffix}.ply"
    color_mesh.export(str(color_path))
    print(f"    Saved color mesh to {color_path}")
    
    print(f"    ✓ 3D mesh generation complete for {task}/{environment}" if task and environment else f"    ✓ 3D mesh generation complete")
    
    return boundaries, step_size  # Return the used boundaries and step size


def create_point_cloud_ply_files(fusion, output_dir: Path, task: str, environment: str, boundaries: dict):
    """
    Alternative method: create PLY files from point clouds when mesh extraction fails
    """
    print("    Creating point cloud PLY files...")
    
    mesh_output_dir = output_dir / task / environment
    mesh_output_dir.mkdir(parents=True, exist_ok=True)
    
    num_instances = fusion.get_inst_num()
    
    for inst_id in range(1, num_instances):
        try:
            # Extract point cloud for this instance
            pcd = fusion.extract_masked_pcd_in_views([inst_id], [0], boundaries)
            
            if pcd.shape[0] > 0:
                pcd_np = pcd.cpu().numpy() if hasattr(pcd, 'cpu') else pcd
                
                # Create Open3D point cloud
                import open3d as o3d
                o3d_pcd = o3d.geometry.PointCloud()
                o3d_pcd.points = o3d.utility.Vector3dVector(pcd_np)
                
                # Save as PLY
                pcd_path = mesh_output_dir / f"pointcloud_instance_{inst_id}_{task.lower()}.ply"
                o3d.io.write_point_cloud(str(pcd_path), o3d_pcd)
                print(f"    Saved point cloud for instance {inst_id} to {pcd_path}")
                
        except Exception as e:
            print(f"    Warning: Failed to create point cloud PLY for instance {inst_id}: {e}")
    
    print(f"    ✓ Point cloud PLY generation complete for {task}/{environment}")


def create_simple_pca_model(object_name: str = None, model_name: str = "dinov2_vitl14"):
    """
    Load a proper PCA model for the given object type.
    Falls back to a dummy PCA model if the proper one is not found.
    
    Args:
        object_name: Name of the object (e.g., 'bottle', 'door')
        model_name: DINOv2 model name used for PCA
    
    Returns:
        PCA model for feature visualization
    """
    from sklearn.decomposition import PCA
    import pickle
    from pathlib import Path
    
    # Try to load proper PCA model if object_name is provided
    if object_name:
        pca_model_path = Path("pca_model") / f"{object_name}_{model_name}.pkl"
        
        if pca_model_path.exists():
            try:
                with open(pca_model_path, 'rb') as f:
                    pca_model_dict = pickle.load(f)
                    print(f"    Loaded proper PCA model for {object_name} from {pca_model_path}")
                    return pca_model_dict['pca']
            except Exception as e:
                print(f"    Warning: Failed to load PCA model for {object_name}: {e}")
        else:
            print(f"    Warning: PCA model not found at {pca_model_path}")
    
    # Fallback to dummy PCA model
    print(f"    Using dummy PCA model for {object_name or 'unknown object'}")
    pca = PCA(n_components=3)
    # Fit with dummy data (in practice, this should be fitted on real features)
    dummy_features = np.random.randn(1000, 1024)  # Assuming 1024-dim DINOv2 features
    pca.fit(dummy_features)
    
    return pca


def save_point_clouds_as_ply(point_clouds: dict, output_dir: Path, task: str, environment: str):
    """
    Save extracted point clouds as PLY files
    
    Args:
        point_clouds: Dictionary of point clouds {instance_name: numpy_array}
        output_dir: Output directory
        task: Task name (empty string to use output_dir directly)
        environment: Environment name (empty string to use output_dir directly)
    """
    if not point_clouds:
        print("    No point clouds to save")
        return
    
    print(f"    Saving {len(point_clouds)} point clouds as PLY files...")
    
    # Create output directory - if task/environment are empty, use output_dir directly
    if task and environment:
        mesh_output_dir = output_dir / task / environment
    else:
        mesh_output_dir = output_dir
    mesh_output_dir.mkdir(parents=True, exist_ok=True)
    
    import open3d as o3d
    
    for inst_name, pcd_np in point_clouds.items():
        try:
            # Create Open3D point cloud
            o3d_pcd = o3d.geometry.PointCloud()
            o3d_pcd.points = o3d.utility.Vector3dVector(pcd_np)
            
            # Add some color for visualization (you could also extract actual colors)
            colors = np.random.rand(len(pcd_np), 3)  # Random colors for now
            o3d_pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # Save as PLY - use task name if provided, otherwise use generic name
            task_suffix = f"_{task.lower()}" if task else ""
            pcd_path = mesh_output_dir / f"{inst_name}{task_suffix}.ply"
            o3d.io.write_point_cloud(str(pcd_path), o3d_pcd)
            print(f"    Saved {inst_name} point cloud ({len(pcd_np)} points) to {pcd_path}")
            
        except Exception as e:
            print(f"    Warning: Failed to save {inst_name} point cloud: {e}")
    
    print(f"    ✓ Point cloud PLY files saved for {task}/{environment}" if task and environment else f"    ✓ Point cloud PLY files saved")


def create_pointcloud_from_depth_and_mask(fusion, inst_id):
    """
    Fallback method to create point cloud directly from depth and mask data
    when fusion.extract_masked_pcd fails
    """
    try:
        # Get depth, mask, and camera parameters
        depth = fusion.curr_obs_torch['depth']  # [num_cam, H, W]
        mask = fusion.curr_obs_torch['mask']    # [num_cam, H, W, num_inst]
        K = fusion.curr_obs_torch['K']          # [num_cam, 3, 3]
        pose = fusion.curr_obs_torch['pose']    # [num_cam, 3, 4]
        
        all_points = []
        
        for cam_id in range(fusion.num_cam):
            if inst_id >= mask.shape[-1]:
                continue
                
            # Get mask for this instance and camera
            inst_mask = mask[cam_id, :, :, inst_id] > 0.5
            inst_depth = depth[cam_id]
            
            if inst_mask.sum() == 0:
                continue
            
            # Get masked depth values
            masked_depth = inst_depth[inst_mask]
            
            # Filter out zero/invalid depth values
            valid_depth = masked_depth > 0.01  # Minimum depth threshold
            if valid_depth.sum() == 0:
                continue
            
            # Get pixel coordinates
            H, W = inst_mask.shape
            v, u = torch.where(inst_mask)
            
            # Filter by valid depth
            v = v[valid_depth]
            u = u[valid_depth]
            z = masked_depth[valid_depth]
            
            if len(z) == 0:
                continue
            
            # Back-project to 3D (camera coordinates)
            fx = K[cam_id, 0, 0]
            fy = K[cam_id, 1, 1]
            cx = K[cam_id, 0, 2]
            cy = K[cam_id, 1, 2]
            
            x = (u.float() - cx) * z / fx
            y = (v.float() - cy) * z / fy
            
            # Stack to get points in camera coordinates
            cam_points = torch.stack([x, y, z], dim=1)  # [N, 3]
            
            # Transform to world coordinates
            R = pose[cam_id, :3, :3]  # [3, 3]
            t = pose[cam_id, :3, 3]   # [3]
            
            # Apply transformation: world_points = R @ cam_points.T + t
            world_points = (R @ cam_points.T).T + t
            
            all_points.append(world_points)
        
        if all_points:
            # Concatenate points from all cameras
            combined_points = torch.cat(all_points, dim=0)
            print(f"    Fallback: Created {len(combined_points)} points for instance {inst_id}")
            return combined_points
        else:
            print(f"    Fallback: No valid points found for instance {inst_id}")
            return torch.empty((0, 3), device=fusion.device)
            
    except Exception as e:
        print(f"    Fallback failed for instance {inst_id}: {e}")
        return torch.empty((0, 3), device=fusion.device)


def get_frame_count(task: str, environment: str) -> int:
    """Get the number of frames available for a given task/environment."""
    loader = D3FieldsDataLoader()
    
    # Load any frame to get dataset info
    try:
        datapoint = loader.load_datapoint(task, environment, frame_idx=0)
        return datapoint['dataset_info']['num_frames']
    except Exception as e:
        print(f"    Warning: Could not determine frame count for {task}/{environment}: {e}")
        return 1  # Default to 1 frame


def compute_frame_scores(frame_results: dict, output_dir: Path, task: str, environment: str) -> dict:
    """
    Compute scores for a single frame's results.
    
    Args:
        frame_results: Results from process_single_frame
        output_dir: Output directory
        task: Task name
        environment: Environment name
    
    Returns:
        dict: Scoring metrics
    """
    scores = {
        'instance_count': frame_results.get('num_instances', 0),
        'point_cloud_sizes': {},
        'mesh_quality': {},
        'processing_success': frame_results.get('status') == 'success'
    }
    
    # Load point clouds and compute sizes
    frame_dir = Path(frame_results.get('output_dir', ''))
    if frame_dir.exists():
        for file in frame_dir.glob("pointcloud_*.npy"):
            try:
                pcd = np.load(file)
                instance_name = file.stem
                scores['point_cloud_sizes'][instance_name] = len(pcd)
            except Exception as e:
                print(f"    Warning: Could not load point cloud {file}: {e}")
        
        # Check for mesh files
        mesh_files = list(frame_dir.glob("*_mesh_*.ply"))
        scores['mesh_quality']['num_meshes'] = len(mesh_files)
        scores['mesh_quality']['mesh_files'] = [f.name for f in mesh_files]
    
    return scores


def process_single_frame(task: str, environment: str, frame_idx: int, output_dir: Path, 
                        device: str = 'cuda:0', use_gt_masks: bool = False) -> dict:
    """
    Process a single frame and return results.
    
    Args:
        task: Task name
        environment: Environment name
        frame_idx: Frame index to process
        output_dir: Output directory for results
        device: Device to run on
        use_gt_masks: Whether to use GT masks instead of GroundingSAM
    
    Returns:
        dict: Processing results
    """
    print(f"    Processing frame {frame_idx}...")
    
    try:
        # Load data
        loader = D3FieldsDataLoader()
        datapoint = loader.load_datapoint(task, environment, frame_idx)
        
        # Convert to fusion format
        obs = convert_to_fusion_format(datapoint)
        
        # Get task-specific queries
        queries = TASK_QUERIES.get(task, ['object', 'robotic arm', 'gripper'])
        
        # Real D3Fields processing
        num_cameras = obs['color'].shape[0]
        fusion = Fusion(num_cam=num_cameras, device=device, is_data_from_adamanip=True)
        
        # Update with observation
        fusion.update(obs)
        
        if use_gt_masks:
            # Load and apply GT masks
            gt_masks = load_gt_masks(task, environment, frame_idx, num_cameras)
            apply_gt_masks_to_fusion_simple(fusion, gt_masks)
        else:
            raise ValueError('use_gt_masks must be True')
        
        # Get results
        num_instances = fusion.get_inst_num()
        print(f"      Found {num_instances} instances")
        
        # Extract point clouds and features
        point_clouds = {}
        features_info = {}
        
        for inst_id in range(1, num_instances):
            print(f"      Extracting point cloud for instance {inst_id}")
            
            pcd = fusion.extract_masked_pcd([inst_id], None)
            print(f"      Extracted {pcd.shape[0]} points for instance {inst_id}")
            
            if pcd.shape[0] == 0:
                print(f"      Warning: No points extracted for instance {inst_id}")
                continue
            
            if pcd.shape[0] > 0:
                # Convert tensor to numpy if needed
                pcd_np = pcd.cpu().numpy() if hasattr(pcd, 'cpu') else pcd
                point_clouds[f'instance_{inst_id}'] = pcd_np
                
                # Extract features
                N = min(100, pcd.shape[0])
                if N > 0:
                    src_feats_list, src_pts_list, img_list = fusion.select_features_from_pcd(
                        pcd_np, N=N, per_instance=True, vis=False
                    )
                    
                    features_info[f'instance_{inst_id}'] = {
                        'num_features': len(src_feats_list),
                        'feature_dims': [f.shape for f in src_feats_list] if src_feats_list else [],
                        'num_points': N
                    }
        
        print(f"      Successfully extracted point clouds for {len(point_clouds)} instances")
        
        # Create frame-specific output directory
        frame_output_dir = output_dir / f"frame_{frame_idx}"
        frame_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save point clouds as PLY files - pass frame_output_dir directly 
        # and empty strings for task/environment since directory is already correct
        save_point_clouds_as_ply(point_clouds, frame_output_dir, "", "")
        
        # Create and save 3D meshes - pass frame_output_dir directly
        # and empty strings for task/environment since directory is already correct
        try:
            used_boundaries, used_step_size = create_meshes_and_save(
                fusion, frame_output_dir, "", "", 
                point_clouds=point_clouds, boundaries=None, step_size=None
            )
        except Exception as e:
            print(f"      Warning: Failed to create 3D meshes for frame {frame_idx}: {e}")
            used_boundaries, used_step_size = infer_boundaries_and_step(point_clouds or {})
        
        # Clean up fusion
        fusion.close()
        
        # Save point clouds as numpy arrays
        for inst_id, pcd in point_clouds.items():
            pcd_path = frame_output_dir / f"pointcloud_{inst_id}.npy"
            np.save(pcd_path, pcd)
        
        # Save original images for reference
        for cam_id, color_img in enumerate(obs['color']):
            img_path = frame_output_dir / f"camera_{cam_id}_color.png"
            color_bgr = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(img_path), color_bgr)
        
        # Save frame metadata
        frame_metadata = {
            'task': task,
            'environment': environment,
            'frame_idx': frame_idx,
            'queries': queries,
            'num_instances': num_instances,
            'num_cameras': obs['color'].shape[0],
            'boundaries': used_boundaries,
            'step_size': used_step_size,
            'thresholds': DEFAULT_THRESHOLDS,
            'image_shape': obs['color'].shape[1:3],
            'features_info': features_info,
            'use_gt_masks': use_gt_masks,
        }
        
        metadata_path = frame_output_dir / "frame_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(frame_metadata, f, indent=2)
        
        result = {
            'status': 'success',
            'frame_idx': frame_idx,
            'num_instances': num_instances,
            'output_dir': str(frame_output_dir),
            'point_cloud_count': len(point_clouds),
        }
        
        print(f"      ✓ Frame {frame_idx} processed: {num_instances} instances, saved to {frame_output_dir}")
        return result
        
    except Exception as e:
        print(f"      ❌ Frame {frame_idx} failed: {e}")
        print(f"      Error details: {traceback.format_exc()}")
        return {
            'status': 'failed',
            'frame_idx': frame_idx,
            'error': str(e),
            'output_dir': None,
        }


def process_all_frames(task: str, environment: str, output_dir: Path, 
                      device: str = 'cuda:0', use_gt_masks: bool = False, max_frames: int = None) -> dict:
    """
    Process all frames for a given environment and compute scores.
    
    Args:
        task: Task name
        environment: Environment name
        output_dir: Output directory for results
        device: Device to run on
        use_gt_masks: Whether to use GT masks instead of GroundingSAM
        max_frames: Maximum number of frames to process (None for all)
    
    Returns:
        dict: Processing results and scores
    """
    print(f"  Processing all frames for {task}/{environment}")
    
    # Get total frame count
    total_frames = get_frame_count(task, environment)
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)
    
    print(f"    Processing {total_frames} frames...")
    
    # Create all_frames directory structure
    all_frames_dir = output_dir / task / environment / "all_frames"
    all_frames_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each frame
    frame_results = {}
    frame_scores = {}
    
    for frame_idx in range(total_frames):
        try:
            # Process single frame
            frame_result = process_single_frame(
                task, environment, frame_idx, all_frames_dir, device, use_gt_masks
            )
            frame_results[f'frame_{frame_idx}'] = frame_result
            
            # Compute scores for this frame
            frame_score = compute_frame_scores(frame_result, all_frames_dir, task, environment)
            frame_scores[f'frame_{frame_idx}'] = frame_score
            
        except Exception as e:
            print(f"    Error processing frame {frame_idx}: {e}")
            frame_results[f'frame_{frame_idx}'] = {
                'status': 'failed',
                'frame_idx': frame_idx,
                'error': str(e)
            }
            frame_scores[f'frame_{frame_idx}'] = {
                'processing_success': False,
                'error': str(e)
            }
    
    # Compute overall statistics
    successful_frames = sum(1 for r in frame_results.values() if r.get('status') == 'success')
    total_instances = sum(r.get('num_instances', 0) for r in frame_results.values() if r.get('status') == 'success')
    avg_instances = total_instances / max(successful_frames, 1)
    
    # Compute temporal consistency scores
    temporal_scores = compute_temporal_consistency(frame_results, frame_scores)
    
    # Overall results
    all_frames_result = {
        'task': task,
        'environment': environment,
        'total_frames': total_frames,
        'successful_frames': successful_frames,
        'failed_frames': total_frames - successful_frames,
        'success_rate': successful_frames / total_frames if total_frames > 0 else 0,
        'avg_instances_per_frame': avg_instances,
        'total_instances': total_instances,
        'frame_results': frame_results,
        'frame_scores': frame_scores,
        'temporal_scores': temporal_scores,
        'output_dir': str(all_frames_dir),
    }
    
    # Save overall results
    results_file = all_frames_dir / "all_frames_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_frames_result, f, indent=2)
    
    print(f"    ✓ Processed {successful_frames}/{total_frames} frames successfully")
    print(f"    📊 Average {avg_instances:.1f} instances per frame, {total_instances} total instances")
    print(f"    📁 Results saved to {all_frames_dir}")
    
    return all_frames_result


def compute_temporal_consistency(frame_results: dict, frame_scores: dict) -> dict:
    """
    Compute temporal consistency scores across frames.
    
    Args:
        frame_results: Results from all frames
        frame_scores: Scores from all frames
    
    Returns:
        dict: Temporal consistency metrics
    """
    # Extract instance counts over time
    instance_counts = []
    point_cloud_counts = []
    
    for frame_key in sorted(frame_results.keys(), key=lambda x: int(x.split('_')[1])):
        result = frame_results[frame_key]
        if result.get('status') == 'success':
            instance_counts.append(result.get('num_instances', 0))
            point_cloud_counts.append(result.get('point_cloud_count', 0))
    
    if len(instance_counts) < 2:
        return {
            'instance_count_variance': 0,
            'instance_count_stability': 1.0,
            'point_cloud_count_variance': 0,
            'point_cloud_count_stability': 1.0,
            'temporal_consistency_score': 1.0,
        }
    
    # Compute variance and stability metrics
    instance_variance = np.var(instance_counts)
    instance_stability = 1.0 / (1.0 + instance_variance)  # Higher stability = lower variance
    
    point_cloud_variance = np.var(point_cloud_counts)
    point_cloud_stability = 1.0 / (1.0 + point_cloud_variance)
    
    # Overall temporal consistency score
    consistency_score = (instance_stability + point_cloud_stability) / 2
    
    return {
        'instance_count_variance': float(instance_variance),
        'instance_count_stability': float(instance_stability),
        'point_cloud_count_variance': float(point_cloud_variance),
        'point_cloud_count_stability': float(point_cloud_stability),
        'temporal_consistency_score': float(consistency_score),
        'instance_counts_over_time': instance_counts,
        'point_cloud_counts_over_time': point_cloud_counts,
    }


def process_environment(task: str, environment: str, output_dir: Path, 
                        device: str = 'cuda:0', frame_idx: int = 0, use_gt_masks: bool = False,
                        process_all_frames_flag: bool = False, max_frames: int = None):
    """
    Process a single environment - can process single frame or all frames.
    
    Args:
        task: Task name
        environment: Environment name
        output_dir: Output directory for results
        device: Device to run on
        frame_idx: Frame index to process (only used if process_all_frames_flag=False)
        use_gt_masks: Whether to use GT masks instead of GroundingSAM
        process_all_frames_flag: Whether to process all frames or just one
        max_frames: Maximum number of frames to process (None for all)
    
    Returns:
        dict: Processing results
    """
    if process_all_frames_flag:
        # Process all frames
        return process_all_frames(task, environment, output_dir, device, use_gt_masks, max_frames)
    else:
        # Process single frame (original behavior)
        print(f"  Processing {task}/{environment} (frame {frame_idx}, GT masks: {use_gt_masks})")
        
        # Load data
        loader = D3FieldsDataLoader()
        datapoint = loader.load_datapoint(task, environment, frame_idx)
        
        # Convert to fusion format
        obs = convert_to_fusion_format(datapoint)
        
        # Get task-specific queries
        queries = TASK_QUERIES.get(task, ['object', 'robotic arm', 'gripper'])
        
        # Real D3Fields processing
        num_cameras = obs['color'].shape[0]
        fusion = Fusion(num_cam=num_cameras, device=device, is_data_from_adamanip=True)
        
        # Update with observation
        fusion.update(obs)
        
        if use_gt_masks:
            # Load and apply GT masks
            gt_masks = load_gt_masks(task, environment, frame_idx, num_cameras)
            apply_gt_masks_to_fusion_simple(fusion, gt_masks)
        else:
            raise ValueError('use_gt_masks must be True')
        
        # Get results
        num_instances = fusion.get_inst_num()
        print(f"    Number of instances detected: {num_instances}")
        
        # Extract point clouds and features
        point_clouds = {}
        features_info = {}
        
        for inst_id in range(1, num_instances):
            print(f"    Attempting to extract point cloud for instance {inst_id}")
            
            pcd = fusion.extract_masked_pcd([inst_id], None)
            print(f"    Extracted {pcd.shape[0]} points for instance {inst_id}")
            
            if pcd.shape[0] == 0:
                raise ValueError(f"No points extracted for instance {inst_id}")
            
            if pcd.shape[0] > 0:
                # Convert tensor to numpy if needed
                pcd_np = pcd.cpu().numpy() if hasattr(pcd, 'cpu') else pcd
                point_clouds[f'instance_{inst_id}'] = pcd_np
                
                # Extract features
                N = min(100, pcd.shape[0])
                if N > 0:
                    src_feats_list, src_pts_list, img_list = fusion.select_features_from_pcd(
                        pcd_np, N=N, per_instance=True, vis=False
                    )
                    
                    features_info[f'instance_{inst_id}'] = {
                        'num_features': len(src_feats_list),
                        'feature_dims': [f.shape for f in src_feats_list] if src_feats_list else [],
                        'num_points': N
                    }
        
        print(f"    Successfully extracted point clouds for {len(point_clouds)} instances")
        
        # Save point clouds as PLY files
        save_point_clouds_as_ply(point_clouds, output_dir, task, environment)
        
        # Create and save 3D meshes (mask, feature, color)
        try:
            used_boundaries, used_step_size = create_meshes_and_save(fusion, output_dir, task, environment, point_clouds=point_clouds, boundaries=None, step_size=None)
        except Exception as e:
            print(f"    Warning: Failed to create 3D meshes: {e}")
            print(f"    Error details: {traceback.format_exc()}")
            used_boundaries, used_step_size = infer_boundaries_and_step(point_clouds or {})
        
        # Clean up fusion
        fusion.close()
        
        # Save results
        env_output_dir = output_dir / task / environment
        env_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save point clouds
        import numpy as np
        for inst_id, pcd in point_clouds.items():
            pcd_path = env_output_dir / f"pointcloud_{inst_id}.npy"
            np.save(pcd_path, pcd)
        
        # Save original images for reference
        import cv2
        for cam_id, color_img in enumerate(obs['color']):
            img_path = env_output_dir / f"camera_{cam_id}_color.png"
            color_bgr = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(img_path), color_bgr)
        
        # Save metadata
        metadata = {
            'task': task,
            'environment': environment,
            'frame_idx': frame_idx,
            'queries': queries,
            'num_instances': num_instances,
            'num_cameras': obs['color'].shape[0],
            'boundaries': used_boundaries,
            'step_size': used_step_size,
            'default_boundaries': DEFAULT_BOUNDARIES,
            'thresholds': DEFAULT_THRESHOLDS,
            'image_shape': obs['color'].shape[1:3],
            'features_info': features_info,
            'use_gt_masks': use_gt_masks,
        }
        
        metadata_path = env_output_dir / "d3fields_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        result = {
            'status': 'success',
            'num_instances': num_instances,
            'output_dir': str(env_output_dir),
        }
        
        print(f"    ✓ Found {num_instances} instances, saved to {env_output_dir}")
        return result


def process_task(task: str, output_dir: Path, max_envs: int = None, 
                env_types: list = ['grasp', 'manip'], device: str = 'cuda:0', use_gt_masks: bool = False,
                process_all_frames_flag: bool = False, max_frames: int = None):
    """
    Process all environments for a given task.
    
    Args:
        task: Task name
        output_dir: Output directory
        max_envs: Maximum environments per type (None for all)
        env_types: Environment types to process
        device: Device to run on
        use_gt_masks: Whether to use GT masks instead of GroundingSAM
        process_all_frames_flag: Whether to process all frames or just one
        max_frames: Maximum number of frames to process (None for all)
    
    Returns:
        dict: Task processing results
    """
    print(f"\nProcessing task: {task} (GT masks: {use_gt_masks}, All frames: {process_all_frames_flag})")
    print("-" * 40)
    
    loader = D3FieldsDataLoader()
    environments = loader.get_environments(task)
    
    results = {'task': task, 'environments': {}, 'summary': {'total': 0, 'success': 0, 'failed': 0}}
    
    for env_type in env_types:
        envs = environments.get(env_type, [])
        if max_envs != -1:
            envs = envs[:max_envs]
        
        print(f"  Processing {len(envs)} {env_type} environments...")
        
        for env in envs:
            env_result = process_environment(
                task, env, output_dir, device, 
                frame_idx=0,  # Only used for single frame mode
                use_gt_masks=use_gt_masks,
                process_all_frames_flag=process_all_frames_flag,
                max_frames=max_frames
            )
            results['environments'][env] = env_result
            results['summary']['total'] += 1
            
            if env_result.get('status') == 'success' or env_result.get('successful_frames', 0) > 0:
                results['summary']['success'] += 1
            else:
                results['summary']['failed'] += 1
    
    print(f"  Task summary: {results['summary']['success']}/{results['summary']['total']} successful")
    return results


def main(output_dir: str = "d3fields_results", 
         max_envs_per_type: int = 3,
         env_types: list = ['grasp', 'manip'],
         device: str = 'cuda:0',
         tasks: list = None,
         use_gt_masks: bool = False,
         process_all_frames_flag: bool = False,
         max_frames: int = None,
         ):
    """
    Main function to process all tasks in the dataset.
    
    Args:
        output_dir: Output directory for results
        max_envs_per_type: Maximum environments per type to process
        env_types: Environment types to process
        device: Device to run on
        tasks: Specific tasks to process (None for all)
        use_gt_masks: Whether to use GT masks instead of GroundingSAM
        process_all_frames_flag: Whether to process all frames or just one
        max_frames: Maximum number of frames to process (None for all)
    """
    print("🚀 D3Fields Processing for AdaManip Dataset")
    if process_all_frames_flag:
        print("📹 Multi-Frame Processing Mode")
    else:
        print("📷 Single-Frame Processing Mode")
    print("=" * 60)
    
    # Setup
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get available tasks
    loader = D3FieldsDataLoader()
    available_tasks = loader.get_tasks()
    
    if tasks is None:
        tasks = available_tasks
    else:
        tasks = [t for t in tasks if t in available_tasks]
    
    print(f"Will process {len(tasks)} tasks: {tasks}")
    print(f"Max environments per type: {max_envs_per_type}")
    print(f"Environment types: {env_types}")
    print(f"Device: {device}")
    print(f"Use GT masks: {use_gt_masks}")
    print(f"Process all frames: {process_all_frames_flag}")
    if process_all_frames_flag and max_frames:
        print(f"Max frames per environment: {max_frames}")
    print(f"Output directory: {output_dir}")
    
    
    # Process all tasks
    all_results = {'processing_info': {
        'output_dir': str(output_dir),
        'max_envs_per_type': max_envs_per_type,
        'env_types': env_types,
        'device': device,
        'use_gt_masks': use_gt_masks,
        'process_all_frames': process_all_frames_flag,
        'max_frames': max_frames,
    }, 'tasks': {}}
    
    total_success = 0
    total_failed = 0
    
    for task in tasks:
        task_results = process_task(
            task, output_dir, max_envs_per_type, env_types, device, 
            use_gt_masks, process_all_frames_flag, max_frames
        )
        all_results['tasks'][task] = task_results
        
        # Update totals
        total_success += task_results['summary']['success']
        total_failed += task_results['summary']['failed']
    
    # Save overall results
    all_results['overall_summary'] = {
        'total_environments': total_success + total_failed,
        'successful': total_success,
        'failed': total_failed,
        'success_rate': total_success / (total_success + total_failed) if (total_success + total_failed) > 0 else 0
    }
    
    results_file = output_dir / "processing_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print final summary
    print("\n" + "=" * 60)
    print("🎉 Processing Complete!")
    print("=" * 60)
    print(f"📊 Overall Results:")
    print(f"   Total environments: {total_success + total_failed}")
    print(f"   ✅ Successful: {total_success}")
    print(f"   ❌ Failed: {total_failed}")
    print(f"   📈 Success rate: {all_results['overall_summary']['success_rate']:.1%}")
    print(f"📁 Results saved to: {output_dir}")
    print(f"📄 Full results: {results_file}")
    

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute D3Fields for AdaManip dataset')
    parser.add_argument('--output_dir', type=str, default='output/d3fields_results',
                       help='Output directory for results')
    parser.add_argument('--max_envs', type=int, default=-1,
                       help='Maximum environments per type to process')
    parser.add_argument('--env_types', nargs='+', default=['grasp', 'manip'],
                       choices=['grasp', 'manip'],
                       help='Environment types to process')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to run on (cuda:0, cpu, etc.)')
    parser.add_argument('--tasks', nargs='+', default=None,
                       help='Specific tasks to process (default: all)')
    parser.add_argument('--use_gt_masks', action='store_true',
                       help='Use GT masks instead of GroundingSAM')
    parser.add_argument('--quick_test', action='store_true',
                       help='Quick test with 1 environment per task')
    
    # New arguments for multi-frame processing
    parser.add_argument('--all_frames', action='store_true',
                       help='Process all frames instead of just frame 0')
    parser.add_argument('--max_frames', type=int, default=None,
                       help='Maximum number of frames to process per environment (None for all)')
    
    args = parser.parse_args()
    
    # Quick test mode
    if args.quick_test:
        print("🧪 Running in quick test mode")
        args.max_envs = 1
        args.env_types = ['grasp']
        args.tasks = ['OpenBottle', 'OpenDoor']
        if not args.all_frames:
            print("   Single frame mode (use --all_frames for multi-frame test)")
    
    main(
        output_dir=args.output_dir,
        max_envs_per_type=args.max_envs,
        env_types=args.env_types,
        device=args.device,
        tasks=args.tasks,
        use_gt_masks=args.use_gt_masks,
        process_all_frames_flag=args.all_frames,
        max_frames=args.max_frames,
    ) 