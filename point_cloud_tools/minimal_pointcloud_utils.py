"""
Minimal Point Cloud Utilities for D3Fields
"""

import numpy as np
import cv2
from pathlib import Path

# Try to import Open3D, but make it optional
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("Warning: Open3D not available. PLY files will be created with basic writer.")


def load_camera_data(camera_dir):
    """
    Load camera parameters and extrinsics from a camera directory.
    
    Args:
        camera_dir: Path to camera directory containing camera_params.npy and camera_extrinsics.npy
    
    Returns:
        dict: Camera data with 'params' [fx, fy, cx, cy] and 'extrinsics' [4x4 matrix]
    """
    camera_dir = Path(camera_dir)
    
    # Load camera parameters (intrinsics)
    params_file = camera_dir / "camera_params.npy"
    params = np.load(params_file)  # [fx, fy, cx, cy]
    
    # Load camera extrinsics
    extrinsics_file = camera_dir / "camera_extrinsics.npy"
    extrinsics = np.load(extrinsics_file)  # 4x4 world-to-camera matrix
    
    return {
        'params': params,
        'extrinsics': extrinsics
    }


def depth_to_pointcloud(depth_image, mask, camera_params):
    """
    Convert depth image to point cloud in camera coordinates.
    
    Args:
        depth_image: (H, W) depth array in meters
        mask: (H, W) boolean mask for valid pixels
        camera_params: [fx, fy, cx, cy] camera intrinsics
    
    Returns:
        points: (N, 3) point cloud in camera coordinates
    """
    h, w = depth_image.shape
    
    # Filter valid depth points
    valid_mask = mask & (depth_image < 2.5)
    
    if not np.any(valid_mask):
        return np.empty((0, 3))
    
    fx, fy, cx, cy = camera_params
    
    # Get pixel coordinates
    pos_y, pos_x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    pos_x = pos_x[valid_mask]
    pos_y = pos_y[valid_mask]
    depth_values = depth_image[valid_mask]
    depth_values = - depth_values
    # Convert to 3D points in camera coordinates
    points = np.zeros((len(depth_values), 3))
    points[:, 0] = - (pos_x - cx) * depth_values / fx  # X
    points[:, 1] = (pos_y - cy) * depth_values / fy  # Y
    points[:, 2] = depth_values                       # Z
    
    return points


def transform_pointcloud(points, extrinsics):
    """
    Transform point cloud using extrinsics matrix.
    
    Args:
        points: (N, 3) point cloud
        extrinsics: (4, 4) transformation matrix (typically camera-to-world)
    
    Returns:
        transformed_points: (N, 3) transformed point cloud
    """
    if len(points) == 0:
        return points
    # Convert to homogeneous coordinates
    points_homo = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    
    # Apply transformation
    transformed_homo = (extrinsics @ points_homo.T).T
    
    # Convert back to 3D coordinates
    return transformed_homo[:, :3]


def save_pointcloud_npy(points, filepath):
    """
    Save point cloud as numpy array.
    
    Args:
        points: (N, 3) point cloud
        filepath: Path to save the .npy file
    """
    np.save(filepath, points)
    print(f"Saved {len(points)} points to {filepath}")


def load_pointcloud_npy(filepath):
    """
    Load point cloud from numpy array.
    
    Args:
        filepath: Path to .npy file
    
    Returns:
        points: (N, 3) point cloud
    """
    points = np.load(filepath)
    print(f"Loaded {len(points)} points from {filepath}")
    return points


def save_pointcloud_ply_simple(points, filepath, colors=None):
    """
    Simple PLY writer that doesn't require Open3D.
    
    Args:
        points: (N, 3) point cloud
        filepath: Path to save the .ply file
        colors: (N, 3) RGB colors [0-1] (optional)
    """
    n_points = len(points)
    
    # Ensure colors are in [0, 255] range for PLY format
    if colors is not None:
        if colors.max() <= 1.0:
            colors = (colors * 255).astype(np.uint8)
        else:
            colors = colors.astype(np.uint8)
    
    # Write PLY header
    with open(filepath, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        
        f.write("end_header\n")
        
        # Write vertex data
        for i in range(n_points):
            if colors is not None:
                f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} "
                       f"{colors[i, 0]} {colors[i, 1]} {colors[i, 2]}\n")
            else:
                f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f}\n")
    
    print(f"Saved {n_points} points to {filepath} (simple PLY format)")


def save_pointcloud_ply(points, filepath, colors=None):
    """
    Save point cloud as PLY file for visualization.
    Uses Open3D if available, otherwise falls back to simple writer.
    
    Args:
        points: (N, 3) point cloud
        filepath: Path to save the .ply file
        colors: (N, 3) RGB colors [0-1] (optional)
    """
    if HAS_OPEN3D:
        try:
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            if colors is not None:
                # Ensure colors are in [0, 1] range
                if colors.max() > 1.0:
                    colors = colors / 255.0
                pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # Save as PLY
            o3d.io.write_point_cloud(str(filepath), pcd)
            print(f"Saved {len(points)} points to {filepath} (Open3D)")
            return
        except Exception as e:
            print(f"Open3D failed ({e}), falling back to simple writer...")
    
    # Fallback to simple PLY writer
    save_pointcloud_ply_simple(points, filepath, colors)


def voxel_downsample_simple(points, voxel_size, colors=None):
    """
    Simple voxel downsampling without Open3D dependency.
    
    Args:
        points: (N, 3) point cloud
        voxel_size: float, voxel size for downsampling
        colors: (N, 3) optional colors
    
    Returns:
        downsampled_points: (M, 3) downsampled points
        downsampled_colors: (M, 3) downsampled colors (if colors provided)
    """
    if len(points) == 0:
        return points, colors
    
    # Compute voxel indices
    voxel_indices = np.floor(points / voxel_size).astype(int)
    
    # Find unique voxels
    unique_voxels, inverse_indices = np.unique(voxel_indices, axis=0, return_inverse=True)
    
    # Average points in each voxel
    downsampled_points = []
    downsampled_colors = [] if colors is not None else None
    
    for i in range(len(unique_voxels)):
        mask = inverse_indices == i
        voxel_points = points[mask]
        avg_point = np.mean(voxel_points, axis=0)
        downsampled_points.append(avg_point)
        
        if colors is not None:
            voxel_colors = colors[mask]
            avg_color = np.mean(voxel_colors, axis=0)
            downsampled_colors.append(avg_color)
    
    downsampled_points = np.array(downsampled_points)
    if downsampled_colors is not None:
        downsampled_colors = np.array(downsampled_colors)
    
    return downsampled_points, downsampled_colors


def merge_pointclouds(pointclouds, colors_list=None):
    """
    Merge multiple point clouds into one.
    
    Args:
        pointclouds: List of (N_i, 3) point clouds
        colors_list: List of (N_i, 3) color arrays (optional)
    
    Returns:
        merged_points: (N_total, 3) merged point cloud
        merged_colors: (N_total, 3) merged colors (if colors_list provided)
    """
    # Filter out empty point clouds
    valid_pcds = [pcd for pcd in pointclouds if len(pcd) > 0]
    
    if not valid_pcds:
        return np.empty((0, 3)), np.empty((0, 3))
    
    merged_points = np.vstack(valid_pcds)
    
    if colors_list is not None:
        valid_colors = [colors for i, colors in enumerate(colors_list) if len(pointclouds[i]) > 0]
        merged_colors = np.vstack(valid_colors) if valid_colors else None
        return merged_points, merged_colors
    
    return merged_points


def image_depth_to_pointcloud(image_path, depth_path, camera_params, mask=None):
    """
    Complete pipeline: Load image+depth and convert to point cloud.
    
    Args:
        image_path: Path to RGB image
        depth_path: Path to depth image  
        camera_params: [fx, fy, cx, cy] camera intrinsics
        mask: Optional (H, W) boolean mask
    
    Returns:
        points: (N, 3) point cloud in camera coordinates
        colors: (N, 3) RGB colors [0-1]
    """
    # Load RGB image
    rgb_image = cv2.imread(str(image_path))
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    
    # Load depth image
    depth_image = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
    depth_image = depth_image.astype(np.float32) / 1000.0  # Convert to meters
    
    # Create mask if not provided
    if mask is None:
        mask = depth_image <2.5  # Valid depth pixels
    
    # Convert to point cloud
    points = depth_to_pointcloud(depth_image, mask, camera_params)
    
    # Extract colors for valid points
    if len(points) > 0:
        h, w = depth_image.shape
        valid_mask = mask & (depth_image < 2.5)
        colors = rgb_image[valid_mask] / 255.0  # Normalize to [0, 1]
    else:
        colors = np.empty((0, 3))
    
    return points, colors 