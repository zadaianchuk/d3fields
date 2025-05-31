#!/usr/bin/env python3
"""
Demo script showing the complete point cloud pipeline

This script demonstrates all 4 steps:
1. Convert image + depth to point cloud 
2. Visualize point cloud
3. Transform point cloud with extrinsics
4. Merge multiple camera point clouds

Usage:
    python demo_pointcloud_pipeline.py
"""

import sys
import numpy as np
from pathlib import Path

# Add the point_cloud_tools directory to the path
sys.path.append(str(Path(__file__).parent))

from minimal_pointcloud_utils import *


def demo_single_camera_pipeline():
    """Demo the single camera pipeline (steps 1-3)"""
    print("=" * 60)
    print("DEMO 1: Single Camera Pipeline")
    print("=" * 60)
    
    # Test data paths
    base_dir = Path("/ssdstore/azadaia/project_snellius_sync/d3fields/data/2023-09-11-14-15-50-607452")
    camera_dir = base_dir / "camera_0"
    frame_idx = 0
    
    # Output directory
    output_dir = Path("point_cloud_tools/output/demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not camera_dir.exists():
        print(f"Test data not found at {camera_dir}")
        return False
    
    print(f"Using test data from: {camera_dir}")
    
    # Step 1: Load image + depth and convert to point cloud
    print("\nStep 1: Converting image + depth to point cloud...")
    
    image_path = camera_dir / "color" / f"{frame_idx}.png"
    depth_path = camera_dir / "depth" / f"{frame_idx}.png"
    
    if not image_path.exists() or not depth_path.exists():
        print(f"Required files not found:")
        print(f"  Color: {image_path}")
        print(f"  Depth: {depth_path}")
        return False
    
    # Load camera data
    camera_data = load_camera_data(camera_dir)
    camera_params = camera_data['params']
    
    print(f"Camera parameters: fx={camera_params[0]:.1f}, fy={camera_params[1]:.1f}")
    
    # Convert to point cloud
    points, colors = image_depth_to_pointcloud(image_path, depth_path, camera_params)
    
    # Filter by depth
    depth_mask = points[:, 2] > -2.5    
    points = points[depth_mask]
    colors = colors[depth_mask]
    
    print(f"Generated {len(points)} points (camera coordinates)")
    
    # Save point cloud
    points_file = output_dir / "camera0_points.npy"
    colors_file = output_dir / "camera0_colors.npy"
    save_pointcloud_npy(points, points_file)
    save_pointcloud_npy(colors, colors_file)
    
    # Step 2: Visualize point cloud
    print("\nStep 2: Creating visualization...")
    ply_file = output_dir / "camera0_points.ply"
    save_pointcloud_ply(points, ply_file, colors)
    
    # Step 3: Transform to world coordinates
    print("\nStep 3: Transforming to world coordinates...")
    extrinsics = camera_data['extrinsics']
    
    # Transform (camera-to-world)
    transform_matrix = np.linalg.inv(extrinsics)
    world_points = transform_pointcloud(points, transform_matrix)
    
    print(f"World coordinates range:")
    print(f"  X: [{world_points[:, 0].min():.3f}, {world_points[:, 0].max():.3f}]")
    print(f"  Y: [{world_points[:, 1].min():.3f}, {world_points[:, 1].max():.3f}]")
    print(f"  Z: [{world_points[:, 2].min():.3f}, {world_points[:, 2].max():.3f}]")
    
    # Save transformed points
    world_points_file = output_dir / "camera0_world_points.npy"
    world_ply_file = output_dir / "camera0_world_points.ply"
    save_pointcloud_npy(world_points, world_points_file)
    save_pointcloud_ply(world_points, world_ply_file, colors)
    
    print(f"\nSingle camera demo completed!")
    print(f"Files saved to: {output_dir}")
    
    return True


def demo_multi_camera_pipeline():
    """Demo the multi-camera pipeline (step 4)"""
    print("\n" + "=" * 60)
    print("DEMO 2: Multi-Camera Pipeline")
    print("=" * 60)
    
    # Test data paths
    base_dir = Path("/ssdstore/azadaia/project_snellius_sync/d3fields/data/2023-09-11-14-15-50-607452")
    frame_idx = 0
    num_cameras = 4
    
    # Output directory
    output_dir = Path("point_cloud_tools/output/demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {num_cameras} cameras from: {base_dir}")
    
    all_pointclouds = []
    all_colors = []
    camera_info = []
    
    # Process each camera
    for cam_id in range(num_cameras):
        camera_dir = base_dir / f"camera_{cam_id}"
        
        if not camera_dir.exists():
            print(f"Camera {cam_id} directory not found, skipping...")
            continue
        
        print(f"\nProcessing camera_{cam_id}...")
        
        # Check for required files
        color_path = camera_dir / "color" / f"{frame_idx}.png"
        depth_path = camera_dir / "depth" / f"{frame_idx}.png"
        
        if not color_path.exists() or not depth_path.exists():
            print(f"  Missing files for camera_{cam_id}, skipping...")
            continue
        
        try:
            # Load camera data
            camera_data = load_camera_data(camera_dir)
            camera_params = camera_data['params']
            extrinsics = camera_data['extrinsics']
            
            # Convert to point cloud
            points, colors = image_depth_to_pointcloud(color_path, depth_path, camera_params)
            
            if len(points) == 0:
                print(f"  No valid points for camera_{cam_id}")
                continue
            
            # Filter by depth
            depth_mask = points[:, 2] < 2.0
            points = points[depth_mask]
            colors = colors[depth_mask]
            
            print(f"  Generated {len(points)} points")
            
            # Transform to world coordinates
            transform_matrix = np.linalg.inv(extrinsics)
            world_points = transform_pointcloud(points, transform_matrix)
            
            all_pointclouds.append(world_points)
            all_colors.append(colors)
            camera_info.append({
                'camera_id': cam_id,
                'num_points': len(world_points)
            })
            
        except Exception as e:
            print(f"  Error processing camera_{cam_id}: {e}")
            continue
    
    if not all_pointclouds:
        print("No valid point clouds generated!")
        return False
    
    print(f"\nMerging point clouds from {len(all_pointclouds)} cameras...")
    
    # Merge all point clouds
    merged_points, merged_colors = merge_pointclouds(all_pointclouds, all_colors)
    
    print(f"Merged point cloud statistics:")
    print(f"  Total points: {len(merged_points)}")
    print(f"  X range: [{merged_points[:, 0].min():.3f}, {merged_points[:, 0].max():.3f}]")
    print(f"  Y range: [{merged_points[:, 1].min():.3f}, {merged_points[:, 1].max():.3f}]")
    print(f"  Z range: [{merged_points[:, 2].min():.3f}, {merged_points[:, 2].max():.3f}]")
    
    # Save merged point cloud
    merged_file = output_dir / "merged_all_cameras.npy"
    merged_colors_file = output_dir / "merged_all_cameras_colors.npy"
    merged_ply_file = output_dir / "merged_all_cameras.ply"
    
    save_pointcloud_npy(merged_points, merged_file)
    save_pointcloud_npy(merged_colors, merged_colors_file)
    save_pointcloud_ply(merged_points, merged_ply_file, merged_colors)
    
    # Create per-camera visualization
    per_camera_ply = output_dir / "merged_per_camera_colors.ply"
    camera_colors = [
        [1, 0, 0],  # Red
        [0, 1, 0],  # Green  
        [0, 0, 1],  # Blue
        [1, 1, 0],  # Yellow
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
    print(f"\nMulti-camera pipeline completed!")
    for info in camera_info:
        print(f"  Camera {info['camera_id']}: {info['num_points']} points")
    print(f"  Total merged: {len(merged_points)} points")
    print(f"Files saved to: {output_dir}")
    
    return True


def demo_adamanip_data():
    """Demo with AdaManip data format"""
    print("\n" + "=" * 60)
    print("DEMO 3: AdaManip Data Format")
    print("=" * 60)
    
    # AdaManip test data path
    base_dir = Path("/ssdstore/azadaia/project_snellius_sync/d3fields/output/adamanip_d3fields/OpenCoffeeMachine/grasp_env_0")
    frame_idx = 0
    num_cameras = 2
    
    if not base_dir.exists():
        print(f"AdaManip test data not found at {base_dir}")
        return False
    
    # Output directory
    output_dir = Path("point_cloud_tools/output/demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing AdaManip data from: {base_dir}")
    
    all_pointclouds = []
    all_colors = []
    
    for cam_id in range(num_cameras):
        camera_dir = base_dir / f"camera_{cam_id}"
        
        if not camera_dir.exists():
            continue
        
        color_path = camera_dir / "color" / f"{frame_idx}.png"
        depth_path = camera_dir / "depth" / f"{frame_idx}.png"
        
        if not color_path.exists() or not depth_path.exists():
            continue
        
        try:
            print(f"\nProcessing AdaManip camera_{cam_id}...")
            
            # Load camera data
            camera_data = load_camera_data(camera_dir)
            camera_params = camera_data['params']
            extrinsics = camera_data['extrinsics']
            
            # Convert to point cloud
            points, colors = image_depth_to_pointcloud(color_path, depth_path, camera_params)
            
            if len(points) == 0:
                continue
            
            # Filter by depth (AdaManip uses larger depth range)
            depth_mask = points[:, 2] > -2.5
            points = points[depth_mask]
            colors = colors[depth_mask]
            
            print(f"  Generated {len(points)} points")
            
            # Transform using AdaManip format

            extrinsics = np.linalg.inv(extrinsics)
            extrinsics = extrinsics.T
            world_points = transform_pointcloud(points, extrinsics)
            
            all_pointclouds.append(world_points)
            all_colors.append(colors)
            
        except Exception as e:
            print(f"  Error processing camera_{cam_id}: {e}")
            continue
    
    if all_pointclouds:
        merged_points, merged_colors = merge_pointclouds(all_pointclouds, all_colors)
        
        # Save AdaManip result
        adamanip_file = output_dir / "adamanip_coffee_machine.npy"
        adamanip_ply_file = output_dir / "adamanip_coffee_machine.ply"
        
        save_pointcloud_npy(merged_points, adamanip_file)
        save_pointcloud_ply(merged_points, adamanip_ply_file, merged_colors)
        
        print(f"AdaManip demo completed! Saved {len(merged_points)} points")
        return True
    
    return False


def main():
    """Run all demos"""
    print("Point Cloud Processing Pipeline Demo")
    print("====================================")
    
    # Run demos
    success1 = demo_single_camera_pipeline()
    success2 = demo_multi_camera_pipeline()
    success3 = demo_adamanip_data()
    
    # Summary
    print("\n" + "=" * 60)
    print("DEMO SUMMARY")
    print("=" * 60)
    print(f"Single camera pipeline: {'✓' if success1 else '✗'}")
    print(f"Multi-camera pipeline: {'✓' if success2 else '✗'}")
    print(f"AdaManip data format: {'✓' if success3 else '✗'}")
    
    if any([success1, success2, success3]):
        print(f"\nOutput files saved to: point_cloud_tools/output/demo/")
        print(f"\nTo view PLY files, you can use:")
        print(f"  python -c \"import open3d as o3d; o3d.visualization.draw_geometries([o3d.io.read_point_cloud('point_cloud_tools/output/demo/merged_all_cameras.ply')])\"")
    else:
        print(f"\nNo demos completed successfully. Please check data paths.")


if __name__ == "__main__":
    main() 