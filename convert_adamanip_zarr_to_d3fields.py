#!/usr/bin/env python3
"""
Convert AdaManip zarr data to D3Fields dataset format.

This script extracts actual multi-camera RGB and depth data from AdaManip zarr files
and converts them to the D3Fields dataset structure.
"""

import os
import sys
import argparse
import numpy as np
import cv2
import zarr
from pathlib import Path
import shutil
from tqdm import tqdm


def create_camera_params(fx, fy, cx, cy):
    """Create camera parameters array in D3Fields format."""
    return np.array([fx, fy, cx, cy], dtype=np.float32)


def create_camera_extrinsics(rotation, translation):
    """Create 4x4 extrinsics matrix from rotation and translation."""
    extrinsics = np.eye(4, dtype=np.float32)
    extrinsics[:3, :3] = rotation
    extrinsics[:3, 3] = translation
    return extrinsics


def extract_from_zarr_multicam(zarr_path, output_dir, sample_every=1, max_frames=None):
    """
    Extract multi-camera RGB and depth data from zarr format.
    
    Args:
        zarr_path: Path to the zarr file
        output_dir: Output directory for D3Fields format
        sample_every: Sample every N frames (default: 1 for all frames)
        max_frames: Maximum number of frames to extract (None for all)
    """
    print(f"Loading zarr data from {zarr_path}")
    store = zarr.open(zarr_path, mode='r')
    
    # Get data info
    rgb_data = store['data/rgb_images']
    depth_data = store['data/depth_images']
    
    print(f"RGB shape: {rgb_data.shape}")
    print(f"Depth shape: {depth_data.shape}")
    
    # Parse dimensions
    total_steps, num_cameras, height, width = depth_data.shape
    _, _, _, _, channels = rgb_data.shape
    
    print(f"Total steps: {total_steps}")
    print(f"Number of cameras: {num_cameras}")
    print(f"Image resolution: {height}x{width}")
    print(f"RGB channels: {channels}")
    
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample frames
    if max_frames is not None:
        total_steps = min(total_steps, max_frames)
    
    frame_indices = list(range(0, total_steps, sample_every))
    print(f"Will extract {len(frame_indices)} frames")
    
    # Create camera directories
    for cam_id in range(num_cameras):
        cam_dir = os.path.join(output_dir, f'camera_{cam_id}')
        color_dir = os.path.join(cam_dir, 'color')
        depth_dir = os.path.join(cam_dir, 'depth')
        
        os.makedirs(color_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
    
    # Extract images
    print("Extracting images...")
    for i, frame_idx in enumerate(tqdm(frame_indices)):
        for cam_id in range(num_cameras):
            # Extract RGB image
            rgb_img = rgb_data[frame_idx, cam_id]  # Shape: (H, W, 3)
            
            # Convert RGB to BGR for OpenCV
            bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            
            # Save color image
            color_path = os.path.join(output_dir, f'camera_{cam_id}', 'color', f'{i}.png')
            cv2.imwrite(color_path, bgr_img)
            
            # Extract depth image
            depth_img = depth_data[frame_idx, cam_id]  # Shape: (H, W)
            
            # Convert depth to uint16 (millimeters) for PNG storage
            # Assuming depth is in meters, convert to millimeters
            depth_mm = (depth_img * 1000).astype(np.uint16)
            
            # Save depth image
            depth_path = os.path.join(output_dir, f'camera_{cam_id}', 'depth', f'{i}.png')
            cv2.imwrite(depth_path, depth_mm)
    
    # Create camera parameters and extrinsics for each camera
    print("Creating camera parameters and extrinsics...")
    for cam_id in range(num_cameras):
        cam_dir = os.path.join(output_dir, f'camera_{cam_id}')
        
        # Camera parameters (you may need to adjust these based on your setup)
        # These are reasonable defaults for a 512x512 image
        fx, fy = 400.0, 400.0  # focal lengths in pixels
        cx, cy = 256.0, 256.0  # principal point (image center for 512x512)
        
        camera_params = create_camera_params(fx, fy, cx, cy)
        np.save(os.path.join(cam_dir, 'camera_params.npy'), camera_params)
        
        # Create different poses for each camera
        # These are example poses - you should replace with actual camera poses if available
        if cam_id == 0:
            # Camera 0: front view
            rotation = np.eye(3, dtype=np.float32)
            translation = np.array([0.0, 0.0, 0.8], dtype=np.float32)
        elif cam_id == 1:
            # Camera 1: right view (90 degrees around Y axis)
            rotation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float32)
            translation = np.array([0.8, 0.0, 0.0], dtype=np.float32)
        elif cam_id == 2:
            # Camera 2: left view (-90 degrees around Y axis)
            rotation = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float32)
            translation = np.array([-0.8, 0.0, 0.0], dtype=np.float32)
        else:
            # Additional cameras: place around the scene
            angle = 2 * np.pi * cam_id / num_cameras
            rotation = np.array([
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)]
            ], dtype=np.float32)
            translation = np.array([
                0.8 * np.cos(angle), 
                0.0, 
                0.8 * np.sin(angle)
            ], dtype=np.float32)
        
        extrinsics = create_camera_extrinsics(rotation, translation)
        np.save(os.path.join(cam_dir, 'camera_extrinsics.npy'), extrinsics)
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Convert AdaManip zarr data to D3Fields format')
    parser.add_argument('--source', required=True, help='Source zarr file path')
    parser.add_argument('--output', required=True, help='Output directory for D3Fields format')
    parser.add_argument('--sample_every', type=int, default=1, help='Sample every N frames (default: 1)')
    parser.add_argument('--max_frames', type=int, default=None, help='Maximum number of frames to extract')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.source):
        print(f"Error: Zarr file {args.source} not found")
        return
    
    success = extract_from_zarr_multicam(args.source, args.output, args.sample_every, args.max_frames)
    
    if success:
        print(f"Conversion complete! D3Fields dataset created at: {args.output}")
        print("\nIMPORTANT NOTES:")
        print("1. Camera parameters are estimated based on image size - you may need to adjust them")
        print("2. Camera extrinsics are example poses - replace with actual camera positions if available")
        print("3. Depth values are converted from meters to millimeters for PNG storage")
        print("\nTo use with D3Fields:")
        print("1. Create a PCA model for your object queries")
        print("2. Define workspace boundaries")
        print("3. Run: python vis_repr_custom.py --data_path <output_dir> --pca_path <pca_file> ...")
    else:
        print("Conversion failed!")


if __name__ == '__main__':
    main() 