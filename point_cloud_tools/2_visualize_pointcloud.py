#!/usr/bin/env python3
"""
Script 2: Load point cloud from .npy file and save as .ply for visualization

Usage:
    python 2_visualize_pointcloud.py <input_npy> <output_ply> [colors_npy]

Example:
    python 2_visualize_pointcloud.py \
        point_cloud_tools/output/camera0_frame0_points.npy \
        point_cloud_tools/output/camera0_frame0_points.ply \
        point_cloud_tools/output/camera0_frame0_points_colors.npy
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from minimal_pointcloud_utils import load_pointcloud_npy, save_pointcloud_ply


def main():
    parser = argparse.ArgumentParser(description="Visualize point cloud from numpy array")
    parser.add_argument("input_npy", help="Input .npy point cloud file (N, 3)")
    parser.add_argument("output_ply", help="Output .ply file for visualization")
    parser.add_argument("colors_npy", nargs='?', help="Optional colors .npy file (N, 3)")
    parser.add_argument("--random_colors", action='store_true', 
                       help="Generate random colors if no color file provided")
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_ply)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading point cloud from: {args.input_npy}")
    
    # Load point cloud
    points = load_pointcloud_npy(args.input_npy)
    
    if len(points) == 0:
        print("Empty point cloud!")
        return
    
    # Load or generate colors
    colors = None
    if args.colors_npy:
        print(f"Loading colors from: {args.colors_npy}")
        colors = load_pointcloud_npy(args.colors_npy)
        if len(colors) != len(points):
            print(f"Warning: Color count ({len(colors)}) doesn't match point count ({len(points)})")
            colors = None
    
    if colors is None and args.random_colors:
        print("Generating random colors...")
        colors = np.random.rand(len(points), 3)
    
    # Display point cloud statistics
    print(f"\nPoint cloud statistics:")
    print(f"  Number of points: {len(points)}")
    print(f"  X range: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
    print(f"  Y range: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
    print(f"  Z range: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
    
    if colors is not None:
        print(f"  Has colors: Yes ({colors.shape})")
        if colors.max() > 1.0:
            print(f"  Color range: [0, {colors.max():.1f}] (will be normalized)")
        else:
            print(f"  Color range: [0, 1]")
    else:
        print(f"  Has colors: No")
    
    # Save as PLY
    print(f"\nSaving to: {args.output_ply}")
    save_pointcloud_ply(points, args.output_ply, colors)
    
    print(f"\nDone! You can view the point cloud with:")
    print(f"  - Open3D: python -c \"import open3d as o3d; o3d.visualization.draw_geometries([o3d.io.read_point_cloud('{args.output_ply}')])\"")
    print(f"  - MeshLab, CloudCompare, or any PLY viewer")


if __name__ == "__main__":
    main() 