#!/usr/bin/env python3
"""
Convert AdaManip PNG samples to D3Fields dataset format.

This script works with the PNG sample files from AdaManip and creates a basic
D3Fields dataset structure. Since the PNG samples appear to be composite images,
this script will help you get started with the D3Fields format.
"""

import os
import sys
import argparse
import numpy as np
import cv2
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


def split_composite_image(img, num_cameras=4):
    """
    Split a composite image into individual camera views.
    This assumes the composite image contains multiple camera views arranged in a grid.
    You may need to adjust this based on your actual image layout.
    """
    h, w = img.shape[:2]
    
    if num_cameras == 4:
        # Assume 2x2 grid
        h_half, w_half = h // 2, w // 2
        cameras = [
            img[:h_half, :w_half],      # Top-left
            img[:h_half, w_half:],      # Top-right
            img[h_half:, :w_half],      # Bottom-left
            img[h_half:, w_half:]       # Bottom-right
        ]
    elif num_cameras == 2:
        # Assume side-by-side
        w_half = w // 2
        cameras = [
            img[:, :w_half],            # Left
            img[:, w_half:]             # Right
        ]
    else:
        # For other configurations, just duplicate the image
        cameras = [img] * num_cameras
    
    return cameras


def convert_png_samples(source_dir, output_dir, num_cameras=4):
    """
    Convert PNG sample files to D3Fields format.
    
    Args:
        source_dir: Directory containing PNG sample files
        output_dir: Output directory for D3Fields format
        num_cameras: Number of cameras to simulate
    """
    print(f"Converting PNG samples from {source_dir} to D3Fields format")
    
    # Find all PNG sample files
    png_files = sorted([f for f in os.listdir(source_dir) 
                       if f.startswith('sample_') and f.endswith('.png')])
    
    if not png_files:
        print("No sample PNG files found!")
        return
    
    print(f"Found {len(png_files)} sample files")
    
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    
    for cam_id in range(num_cameras):
        cam_dir = os.path.join(output_dir, f'camera_{cam_id}')
        color_dir = os.path.join(cam_dir, 'color')
        depth_dir = os.path.join(cam_dir, 'depth')
        
        os.makedirs(color_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
    
    # Process each PNG file
    for i, png_file in enumerate(tqdm(png_files)):
        img_path = os.path.join(source_dir, png_file)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Warning: Could not read {png_file}")
            continue
        
        # Split the composite image into camera views
        camera_views = split_composite_image(img, num_cameras)
        
        for cam_id, cam_view in enumerate(camera_views):
            # Save as color image
            color_path = os.path.join(output_dir, f'camera_{cam_id}', 'color', f'{i}.png')
            cv2.imwrite(color_path, cam_view)
            
            # Create a dummy depth image (you'll need to replace this with actual depth data)
            # For now, create a depth image based on grayscale intensity
            gray = cv2.cvtColor(cam_view, cv2.COLOR_BGR2GRAY)
            # Convert to depth-like values (this is just a placeholder)
            depth_dummy = (gray.astype(np.float32) / 255.0 * 2.0 * 1000).astype(np.uint16)  # 0-2m in mm
            depth_path = os.path.join(output_dir, f'camera_{cam_id}', 'depth', f'{i}.png')
            cv2.imwrite(depth_path, depth_dummy)
    
    # Create camera parameters and extrinsics for each camera
    for cam_id in range(num_cameras):
        cam_dir = os.path.join(output_dir, f'camera_{cam_id}')
        
        # Example camera parameters (you need to replace with actual values)
        # These are typical values for a RGB-D camera like Kinect
        fx, fy = 525.0, 525.0  # focal lengths in pixels
        cx, cy = 320.0, 240.0  # principal point (image center)
        
        camera_params = create_camera_params(fx, fy, cx, cy)
        np.save(os.path.join(cam_dir, 'camera_params.npy'), camera_params)
        
        # Create different poses for each camera (example positions)
        if cam_id == 0:
            # Camera 0: front view
            rotation = np.eye(3, dtype=np.float32)
            translation = np.array([0.0, 0.0, 0.5], dtype=np.float32)
        elif cam_id == 1:
            # Camera 1: right view
            rotation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float32)
            translation = np.array([0.5, 0.0, 0.0], dtype=np.float32)
        elif cam_id == 2:
            # Camera 2: back view
            rotation = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32)
            translation = np.array([0.0, 0.0, -0.5], dtype=np.float32)
        elif cam_id == 3:
            # Camera 3: left view
            rotation = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float32)
            translation = np.array([-0.5, 0.0, 0.0], dtype=np.float32)
        else:
            # Default: identity
            rotation = np.eye(3, dtype=np.float32)
            translation = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        extrinsics = create_camera_extrinsics(rotation, translation)
        np.save(os.path.join(cam_dir, 'camera_extrinsics.npy'), extrinsics)


def main():
    parser = argparse.ArgumentParser(description='Convert AdaManip PNG samples to D3Fields format')
    parser.add_argument('--source', required=True, help='Source directory containing PNG sample files')
    parser.add_argument('--output', required=True, help='Output directory for D3Fields format')
    parser.add_argument('--num_cameras', type=int, default=4, help='Number of cameras (default: 4)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.source):
        print(f"Error: Source directory {args.source} not found")
        return
    
    convert_png_samples(args.source, args.output, args.num_cameras)
    
    print(f"Conversion complete! D3Fields dataset created at: {args.output}")
    print("\nIMPORTANT NOTES:")
    print("1. The depth images are currently dummy data based on grayscale intensity")
    print("2. Camera parameters are example values - replace with actual calibration data")
    print("3. Camera extrinsics are example poses - replace with actual camera positions")
    print("4. You may need to adjust the image splitting logic based on your composite image layout")
    print("\nTo use with D3Fields, you'll also need to:")
    print("1. Create a PCA model for your object queries")
    print("2. Define workspace boundaries")
    print("3. Run: python vis_repr_custom.py --data_path <output_dir> --pca_path <pca_file> ...")


if __name__ == '__main__':
    main() 