#!/usr/bin/env python3
"""
Script 4: Merge multiple camera point clouds into one common point cloud

Usage:
    python 4_merge_multi_camera_pointclouds.py <base_data_dir> <frame_index> <output_npy> [--num_cameras N]

Example:
    python 4_merge_multi_camera_pointclouds.py \
        /ssdstore/azadaia/project_snellius_sync/d3fields/data/2023-09-11-14-15-50-607452 \
        0 \
        point_cloud_tools/output/merged_frame0.npy \
        --num_cameras 4
        
    # For AdaManip data:
    python 4_merge_multi_camera_pointclouds.py \
        /ssdstore/azadaia/project_snellius_sync/d3fields/output/adamanip_d3fields/OpenCoffeeMachine/grasp_env_0 \
        0 \
        point_cloud_tools/output/merged_coffee_machine.npy \
        --num_cameras 2 \
        --adamanip
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from minimal_pointcloud_utils import (
    load_camera_data, image_depth_to_pointcloud, transform_pointcloud,
    merge_pointclouds, save_pointcloud_npy, save_pointcloud_ply
)


def adjust_extrinsics_for_data_format(extrinsics, is_adamanip=False):
    """Adjust extrinsics based on data format."""
    if is_adamanip:
        # For AdaManip data, use inverse like in working version: pose = np.linalg.inv(pose)
        return np.linalg.inv(extrinsics)
    else:
        # For regular data, camera-to-world transformation
        return np.linalg.inv(extrinsics)


def main():
    parser = argparse.ArgumentParser(description="Merge multiple camera point clouds")
    parser.add_argument("base_data_dir", help="Base directory containing camera_0, camera_1, etc.")
    parser.add_argument("frame_index", type=int, help="Frame index to process")
    parser.add_argument("output_npy", help="Output .npy file for merged point cloud")
    parser.add_argument("--num_cameras", type=int, default=4, help="Number of cameras")
    parser.add_argument("--adamanip", action='store_true', help="Use AdaManip data format")
    parser.add_argument("--max_depth", type=float, default=2.5, help="Maximum depth threshold")
    parser.add_argument("--downsample", type=float, help="Voxel size for downsampling (optional)")
    parser.add_argument("--no_visualization", action='store_true', help="Skip PLY visualization")
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_npy)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    base_dir = Path(args.base_data_dir)
    
    print(f"Merging point clouds from {args.num_cameras} cameras...")
    print(f"Base directory: {base_dir}")
    print(f"Frame index: {args.frame_index}")
    print(f"Data format: {'AdaManip' if args.adamanip else 'Standard'}")
    
    all_pointclouds = []
    all_colors = []
    camera_info = []
    
    # Process each camera
    for cam_id in range(args.num_cameras):
        camera_dir = base_dir / f"camera_{cam_id}"
        
        if not camera_dir.exists():
            print(f"Warning: Camera directory {camera_dir} does not exist, skipping...")
            continue
        
        print(f"\nProcessing camera_{cam_id}...")
        
        # Check for image and depth files
        color_path = camera_dir / "color" / f"{args.frame_index}.png"
        depth_path = camera_dir / "depth" / f"{args.frame_index}.png"
        
        if not color_path.exists() or not depth_path.exists():
            print(f"  Warning: Missing files for camera_{cam_id}, frame {args.frame_index}")
            continue
        
        try:
            # Load camera data
            camera_data = load_camera_data(camera_dir)
            camera_params = camera_data['params']
            extrinsics = camera_data['extrinsics']
            
            print(f"  Camera params: fx={camera_params[0]:.1f}, fy={camera_params[1]:.1f}")
            
            # Convert image + depth to point cloud
            points, colors = image_depth_to_pointcloud(color_path, depth_path, camera_params)
            
            if len(points) == 0:
                print(f"  No valid points for camera_{cam_id}")
                continue
            
            # Apply depth filtering
            depth_mask = points[:, 2] < args.max_depth
            points = points[depth_mask]
            colors = colors[depth_mask]
            
            print(f"  Generated {len(points)} points (after depth filtering)")
            
            # Transform to world coordinates
            transform_matrix = adjust_extrinsics_for_data_format(extrinsics, args.adamanip)
            world_points = transform_pointcloud(points, transform_matrix)
            
            print(f"  World coordinates range:")
            print(f"    X: [{world_points[:, 0].min():.3f}, {world_points[:, 0].max():.3f}]")
            print(f"    Y: [{world_points[:, 1].min():.3f}, {world_points[:, 1].max():.3f}]")
            print(f"    Z: [{world_points[:, 2].min():.3f}, {world_points[:, 2].max():.3f}]")
            
            all_pointclouds.append(world_points)
            all_colors.append(colors)
            camera_info.append({
                'camera_id': cam_id,
                'num_points': len(world_points),
                'extrinsics': extrinsics,
                'params': camera_params
            })
            
        except Exception as e:
            print(f"  Error processing camera_{cam_id}: {e}")
            continue
    
    if not all_pointclouds:
        print("No valid point clouds generated!")
        return
    
    print(f"\nMerging point clouds from {len(all_pointclouds)} cameras...")
    
    # Merge all point clouds
    merged_points, merged_colors = merge_pointclouds(all_pointclouds, all_colors)
    
    print(f"Merged point cloud statistics:")
    print(f"  Total points: {len(merged_points)}")
    print(f"  X range: [{merged_points[:, 0].min():.3f}, {merged_points[:, 0].max():.3f}]")
    print(f"  Y range: [{merged_points[:, 1].min():.3f}, {merged_points[:, 1].max():.3f}]")
    print(f"  Z range: [{merged_points[:, 2].min():.3f}, {merged_points[:, 2].max():.3f}]")
    
    # Downsample if requested
    if args.downsample:
        print(f"\nDownsampling with voxel size {args.downsample}...")
        try:
            import open3d as o3d
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(merged_points)
            if merged_colors is not None:
                pcd.colors = o3d.utility.Vector3dVector(merged_colors)
            
            pcd_down = pcd.voxel_down_sample(args.downsample)
            merged_points = np.asarray(pcd_down.points)
            merged_colors = np.asarray(pcd_down.colors) if pcd_down.has_colors() else None
            
            print(f"After downsampling: {len(merged_points)} points")
        except ImportError:
            print("Open3D not available, using simple downsampling...")
            from minimal_pointcloud_utils import voxel_downsample_simple
            merged_points, merged_colors = voxel_downsample_simple(merged_points, args.downsample, merged_colors)
            print(f"After downsampling: {len(merged_points)} points")
        except Exception as e:
            print(f"Downsampling failed: {e}")
            print("Proceeding without downsampling...")
    
    # Save merged point cloud
    save_pointcloud_npy(merged_points, args.output_npy)
    
    # Save colors
    if merged_colors is not None:
        colors_path = str(output_path).replace('.npy', '_colors.npy')
        save_pointcloud_npy(merged_colors, colors_path)
    
    # Save visualization
    if not args.no_visualization:
        ply_path = str(output_path).replace('.npy', '.ply')
        print(f"Saving visualization to: {ply_path}")
        save_pointcloud_ply(merged_points, ply_path, merged_colors)
        
        # Save per-camera visualization with different colors
        if len(all_pointclouds) > 1:
            per_camera_ply = str(output_path).replace('.npy', '_per_camera.ply')
            print(f"Saving per-camera visualization to: {per_camera_ply}")
            
            # Create colors for each camera
            camera_colors = [
                [1, 0, 0],  # Red
                [0, 1, 0],  # Green  
                [0, 0, 1],  # Blue
                [1, 1, 0],  # Yellow
                [1, 0, 1],  # Magenta
                [0, 1, 1],  # Cyan
            ]
            
            per_camera_points = []
            per_camera_point_colors = []
            
            for i, points in enumerate(all_pointclouds):
                per_camera_points.append(points)
                color = camera_colors[i % len(camera_colors)]
                point_colors = np.tile(color, (len(points), 1))
                per_camera_point_colors.append(point_colors)
            
            final_points, final_colors = merge_pointclouds(per_camera_points, per_camera_point_colors)
            save_pointcloud_ply(final_points, per_camera_ply, final_colors)
    
    # Print summary
    print(f"\nSummary:")
    for info in camera_info:
        print(f"  Camera {info['camera_id']}: {info['num_points']} points")
    print(f"  Total merged: {len(merged_points)} points")
    print(f"  Saved to: {args.output_npy}")


if __name__ == "__main__":
    main() 