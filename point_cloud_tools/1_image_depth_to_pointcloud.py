#!/usr/bin/env python3
"""
Script 1: Convert image + depth to point cloud and save as .npy file

Usage:
    python 1_image_depth_to_pointcloud.py <image_path> <depth_path> <camera_dir> <output_npy>

Example:
    python 1_image_depth_to_pointcloud.py \
        /ssdstore/azadaia/project_snellius_sync/d3fields/data/2023-09-11-14-15-50-607452/camera_0/color/0.png \
        /ssdstore/azadaia/project_snellius_sync/d3fields/data/2023-09-11-14-15-50-607452/camera_0/depth/0.png \
        /ssdstore/azadaia/project_snellius_sync/d3fields/data/2023-09-11-14-15-50-607452/camera_0 \
        point_cloud_tools/output/camera0_frame0_points.npy
"""

import sys
import argparse
from pathlib import Path
from minimal_pointcloud_utils import load_camera_data, image_depth_to_pointcloud, save_pointcloud_npy


def main():
    parser = argparse.ArgumentParser(description="Convert image + depth to point cloud")
    parser.add_argument("image_path", help="Path to RGB image")
    parser.add_argument("depth_path", help="Path to depth image")
    parser.add_argument("camera_dir", help="Path to camera directory with params/extrinsics")
    parser.add_argument("output_npy", help="Output path for .npy point cloud file")
    parser.add_argument("--max_depth", type=float, default=2.5, help="Maximum depth threshold")
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_npy)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting image + depth to point cloud...")
    print(f"Image: {args.image_path}")
    print(f"Depth: {args.depth_path}")
    print(f"Camera: {args.camera_dir}")
    
    # Load camera data
    camera_data = load_camera_data(args.camera_dir)
    camera_params = camera_data['params']  # [fx, fy, cx, cy]
    
    print(f"Camera params: fx={camera_params[0]:.1f}, fy={camera_params[1]:.1f}, "
          f"cx={camera_params[2]:.1f}, cy={camera_params[3]:.1f}")
    
    # Convert to point cloud
    points, colors = image_depth_to_pointcloud(
        args.image_path, 
        args.depth_path, 
        camera_params
    )
    
    print(f"Generated {len(points)} points")
    
    if len(points) > 0:
        # Apply depth filtering
        depth_mask = points[:, 2] < args.max_depth
        points_filtered = points[depth_mask]
        colors_filtered = colors[depth_mask]
        
        print(f"After depth filtering (<{args.max_depth}m): {len(points_filtered)} points")
        
        # Save point cloud
        save_pointcloud_npy(points_filtered, args.output_npy)
        
        # Also save colors if needed
        colors_path = str(output_path).replace('.npy', '_colors.npy')
        save_pointcloud_npy(colors_filtered, colors_path)
        
        print(f"Point cloud stats:")
        print(f"  X range: [{points_filtered[:, 0].min():.3f}, {points_filtered[:, 0].max():.3f}]")
        print(f"  Y range: [{points_filtered[:, 1].min():.3f}, {points_filtered[:, 1].max():.3f}]")
        print(f"  Z range: [{points_filtered[:, 2].min():.3f}, {points_filtered[:, 2].max():.3f}]")
        
    else:
        print("No valid points found!")


if __name__ == "__main__":
    main() 