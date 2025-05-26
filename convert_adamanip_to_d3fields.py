#!/usr/bin/env python3
"""
Convert AdaManip RGBD data to D3Fields dataset format.

This script converts RGB and depth data from AdaManip format to the D3Fields dataset structure.
The D3Fields format requires:
- dataset_name/
  ├── camera_0/
  │   ├── color/
  │   │   ├── 0.png
  │   │   ├── 1.png
  │   │   └── ...
  │   ├── depth/
  │   │   ├── 0.png
  │   │   ├── 1.png
  │   │   └── ...
  │   ├── camera_extrinsics.npy
  │   └── camera_params.npy
  ├── camera_1/
  └── ...
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


def extract_from_zarr(zarr_path, output_dir, num_cameras=4, sample_every=1):
    """
    Extract RGB and depth data from zarr format.
    
    Args:
        zarr_path: Path to the zarr file
        output_dir: Output directory for D3Fields format
        num_cameras: Number of cameras (default: 4)
        sample_every: Sample every N frames (default: 1 for all frames)
    """
    print(f"Loading zarr data from {zarr_path}")
    store = zarr.open(zarr_path, mode='r')
    
    # Get data info
    rgb_data = store['data/rgb_images']
    depth_data = store['data/depth_images']
    total_steps = int(store['rgbd_meta/total_steps'][()])
    
    print(f"Total steps: {total_steps}")
    print(f"RGB shape: {rgb_data.shape}")
    print(f"Depth shape: {depth_data.shape}")
    
    # Check if data is actually stored in zarr or if we need to look elsewhere
    if rgb_data.shape[1:3] == (1, 1):
        print("WARNING: RGB data appears to be compressed/placeholder (1x1 pixels)")
        print("You may need to provide the actual RGB/depth image files separately")
        return False
    
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample frames
    frame_indices = list(range(0, total_steps, sample_every))
    
    for cam_id in range(num_cameras):
        cam_dir = os.path.join(output_dir, f'camera_{cam_id}')
        color_dir = os.path.join(cam_dir, 'color')
        depth_dir = os.path.join(cam_dir, 'depth')
        
        os.makedirs(color_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        
        print(f"Processing camera {cam_id}")
        
        for i, frame_idx in enumerate(tqdm(frame_indices)):
            # Extract RGB image
            rgb_img = rgb_data[frame_idx]  # This might need adjustment based on actual data format
            if rgb_img.shape[0] > 1 and rgb_img.shape[1] > 1:  # Check if actual image data
                cv2.imwrite(os.path.join(color_dir, f'{i}.png'), rgb_img)
            
            # Extract depth image
            depth_img = depth_data[frame_idx]  # This might need adjustment based on actual data format
            if depth_img.shape[0] > 1 and depth_img.shape[1] > 1:  # Check if actual image data
                # Convert depth to uint16 (millimeters) for PNG storage
                depth_mm = (depth_img * 1000).astype(np.uint16)
                cv2.imwrite(os.path.join(depth_dir, f'{i}.png'), depth_mm)
        
        # Create placeholder camera parameters (you'll need to replace these with actual values)
        # These are example values - you need to get the actual camera intrinsics
        fx, fy = 525.0, 525.0  # focal lengths
        cx, cy = 320.0, 240.0  # principal point
        camera_params = create_camera_params(fx, fy, cx, cy)
        np.save(os.path.join(cam_dir, 'camera_params.npy'), camera_params)
        
        # Create placeholder extrinsics (identity matrix - you'll need actual camera poses)
        # This assumes all cameras are at the same position - you need actual camera poses
        rotation = np.eye(3, dtype=np.float32)
        translation = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        extrinsics = create_camera_extrinsics(rotation, translation)
        np.save(os.path.join(cam_dir, 'camera_extrinsics.npy'), extrinsics)
    
    return True


def extract_from_images(source_dir, output_dir, num_cameras=4):
    """
    Extract RGB and depth data from separate image files.
    
    This function assumes you have RGB and depth images stored in separate directories.
    Modify the paths according to your actual data structure.
    
    Args:
        source_dir: Source directory containing RGB and depth images
        output_dir: Output directory for D3Fields format
        num_cameras: Number of cameras
    """
    print(f"Converting image data from {source_dir} to D3Fields format")
    
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    
    for cam_id in range(num_cameras):
        cam_dir = os.path.join(output_dir, f'camera_{cam_id}')
        color_dir = os.path.join(cam_dir, 'color')
        depth_dir = os.path.join(cam_dir, 'depth')
        
        os.makedirs(color_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        
        # Look for RGB images (modify path according to your structure)
        rgb_source = os.path.join(source_dir, f'camera_{cam_id}', 'rgb')
        depth_source = os.path.join(source_dir, f'camera_{cam_id}', 'depth')
        
        if os.path.exists(rgb_source):
            rgb_files = sorted([f for f in os.listdir(rgb_source) if f.endswith(('.png', '.jpg', '.jpeg'))])
            print(f"Found {len(rgb_files)} RGB images for camera {cam_id}")
            
            for i, rgb_file in enumerate(tqdm(rgb_files)):
                src_path = os.path.join(rgb_source, rgb_file)
                dst_path = os.path.join(color_dir, f'{i}.png')
                
                # Copy and convert to PNG if needed
                img = cv2.imread(src_path)
                cv2.imwrite(dst_path, img)
        
        if os.path.exists(depth_source):
            depth_files = sorted([f for f in os.listdir(depth_source) if f.endswith(('.png', '.tiff', '.tif'))])
            print(f"Found {len(depth_files)} depth images for camera {cam_id}")
            
            for i, depth_file in enumerate(tqdm(depth_files)):
                src_path = os.path.join(depth_source, depth_file)
                dst_path = os.path.join(depth_dir, f'{i}.png')
                
                # Read depth image (might be 16-bit)
                depth_img = cv2.imread(src_path, cv2.IMREAD_ANYDEPTH)
                cv2.imwrite(dst_path, depth_img)
        
        # Create placeholder camera parameters (replace with actual values)
        fx, fy = 525.0, 525.0  # focal lengths
        cx, cy = 320.0, 240.0  # principal point
        camera_params = create_camera_params(fx, fy, cx, cy)
        np.save(os.path.join(cam_dir, 'camera_params.npy'), camera_params)
        
        # Create placeholder extrinsics (replace with actual camera poses)
        rotation = np.eye(3, dtype=np.float32)
        translation = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        extrinsics = create_camera_extrinsics(rotation, translation)
        np.save(os.path.join(cam_dir, 'camera_extrinsics.npy'), extrinsics)


def main():
    parser = argparse.ArgumentParser(description='Convert AdaManip data to D3Fields format')
    parser.add_argument('--source', required=True, help='Source data path (zarr file or image directory)')
    parser.add_argument('--output', required=True, help='Output directory for D3Fields format')
    parser.add_argument('--num_cameras', type=int, default=4, help='Number of cameras (default: 4)')
    parser.add_argument('--sample_every', type=int, default=1, help='Sample every N frames (default: 1)')
    parser.add_argument('--mode', choices=['zarr', 'images'], default='zarr', 
                       help='Data source mode: zarr file or image directories')
    
    args = parser.parse_args()
    
    if args.mode == 'zarr':
        if not os.path.exists(args.source):
            print(f"Error: Zarr file {args.source} not found")
            return
        
        success = extract_from_zarr(args.source, args.output, args.num_cameras, args.sample_every)
        if not success:
            print("Zarr extraction failed. You may need to use --mode images with actual image files.")
    
    elif args.mode == 'images':
        if not os.path.exists(args.source):
            print(f"Error: Source directory {args.source} not found")
            return
        
        extract_from_images(args.source, args.output, args.num_cameras)
    
    print(f"Conversion complete! D3Fields dataset created at: {args.output}")
    print("\nIMPORTANT: Please update the camera parameters and extrinsics with actual values!")
    print("The current values are placeholders and need to be replaced with your actual camera calibration data.")


if __name__ == '__main__':
    main() 