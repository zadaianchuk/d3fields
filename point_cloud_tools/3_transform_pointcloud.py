#!/usr/bin/env python3
"""
Script 3: Transform point cloud using extrinsics and save results + visualization

Usage:
    python 3_transform_pointcloud.py <input_npy> <camera_dir> <output_npy> [colors_npy]

Example:
    python 3_transform_pointcloud.py \
        point_cloud_tools/output/camera0_frame0_points.npy \
        /ssdstore/azadaia/project_snellius_sync/d3fields/data/2023-09-11-14-15-50-607452/camera_0 \
        point_cloud_tools/output/camera0_frame0_transformed.npy \
        point_cloud_tools/output/camera0_frame0_points_colors.npy
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from minimal_pointcloud_utils import (
    load_camera_data, load_pointcloud_npy, transform_pointcloud, 
    save_pointcloud_npy, save_pointcloud_ply
)


def adjust_extrinsics_for_data_format(extrinsics, is_adamanip=False):
    """
    Adjust extrinsics based on data format.
    Based on the existing codebase logic.
    """
    if is_adamanip:
        # For AdaManip data, apply Z-flip and transpose
        T_new = extrinsics.T
        return T_new
    else:
        # For regular data, camera-to-world transformation
        return np.linalg.inv(extrinsics)


def main():
    parser = argparse.ArgumentParser(description="Transform point cloud using camera extrinsics")
    parser.add_argument("input_npy", help="Input .npy point cloud file (N, 3) in camera coordinates")
    parser.add_argument("camera_dir", help="Camera directory containing extrinsics")
    parser.add_argument("output_npy", help="Output .npy file for transformed points")
    parser.add_argument("colors_npy", nargs='?', help="Optional colors .npy file (N, 3)")
    parser.add_argument("--adamanip", action='store_true', help="Use AdaManip data format adjustments")
    parser.add_argument("--no_visualization", action='store_true', help="Skip PLY visualization output")
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_npy)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Transforming point cloud...")
    print(f"Input points: {args.input_npy}")
    print(f"Camera dir: {args.camera_dir}")
    print(f"Data format: {'AdaManip' if args.adamanip else 'Standard'}")
    
    # Load point cloud
    points = load_pointcloud_npy(args.input_npy)
    
    if len(points) == 0:
        print("Empty point cloud!")
        return
    
    # Load colors if available
    colors = None
    if args.colors_npy:
        colors = load_pointcloud_npy(args.colors_npy)
        if len(colors) != len(points):
            print(f"Warning: Color count doesn't match point count")
            colors = None
    
    # Load camera data
    camera_data = load_camera_data(args.camera_dir)
    extrinsics = camera_data['extrinsics']
    
    print(f"Original extrinsics shape: {extrinsics.shape}")
    print(f"Extrinsics matrix:")
    print(extrinsics)
    
    # Adjust extrinsics for transformation
    transform_matrix = adjust_extrinsics_for_data_format(extrinsics, args.adamanip)
    
    print(f"\nTransformation matrix:")
    print(transform_matrix)
    
    # Transform points from camera coordinates to world coordinates
    print(f"\nTransforming {len(points)} points...")
    print(f"Original coordinates range:")
    print(f"  X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
    print(f"  Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
    print(f"  Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
    
    transformed_points = transform_pointcloud(points, transform_matrix)
    
    # Display transformation statistics
    print(f"\nTransformation statistics:")
    print(f"Original point cloud:")
    print(f"  X range: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
    print(f"  Y range: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
    print(f"  Z range: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
    
    print(f"Transformed point cloud:")
    print(f"  X range: [{transformed_points[:, 0].min():.3f}, {transformed_points[:, 0].max():.3f}]")
    print(f"  Y range: [{transformed_points[:, 1].min():.3f}, {transformed_points[:, 1].max():.3f}]")
    print(f"  Z range: [{transformed_points[:, 2].min():.3f}, {transformed_points[:, 2].max():.3f}]")
    
    # Save transformed points
    save_pointcloud_npy(transformed_points, args.output_npy)
    
    # Save visualization if requested
    if not args.no_visualization:
        ply_path = str(output_path).replace('.npy', '.ply')
        print(f"Saving visualization to: {ply_path}")
        save_pointcloud_ply(transformed_points, ply_path, colors)
        
        # Also save comparison visualization with both original and transformed
        if colors is not None:
            comparison_ply = str(output_path).replace('.npy', '_comparison.ply')
            print(f"Saving comparison visualization to: {comparison_ply}")
            
            # Create comparison colors: red for original, blue for transformed
            n_points = len(points)
            comparison_points = np.vstack([points, transformed_points])
            comparison_colors = np.vstack([
                np.column_stack([np.ones(n_points), np.zeros(n_points), np.zeros(n_points)]),  # Red for original
                np.column_stack([np.zeros(n_points), np.zeros(n_points), np.ones(n_points)])   # Blue for transformed
            ])
            save_pointcloud_ply(comparison_points, comparison_ply, comparison_colors)
    
    print(f"\nDone! Transformed point cloud saved to: {args.output_npy}")


if __name__ == "__main__":
    main() 