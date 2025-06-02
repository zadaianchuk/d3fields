#!/usr/bin/env python3
"""
Script to visualize .ply frame files and create a video.
This script loads .ply files from frame directories and renders them into a video.
"""

import os
import sys
import argparse
import numpy as np
import cv2
from pathlib import Path
import glob
import json
from typing import List, Optional, Tuple
import re

try:
    import open3d as o3d
except ImportError:
    print("Error: open3d is required. Install with: pip install open3d")
    sys.exit(1)

def natural_sort_key(text):
    """
    A natural sort key function to sort frame directories properly 
    (e.g., frame_0, frame_1, frame_10 instead of frame_0, frame_1, frame_10)
    """
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', text)]

def find_ply_files(frame_dir: Path) -> List[Path]:
    """Find all .ply files in a frame directory."""
    ply_files = list(frame_dir.glob("*.ply"))
    return sorted(ply_files)

def load_and_merge_ply_files(ply_files: List[Path]) -> Optional[o3d.geometry.PointCloud]:
    """Load multiple .ply files and merge them into a single point cloud."""
    if not ply_files:
        return None
    
    merged_pcd = o3d.geometry.PointCloud()
    
    for ply_file in ply_files:
        try:
            # Try loading as mesh first, then as point cloud
            try:
                mesh = o3d.io.read_triangle_mesh(str(ply_file))
                if len(mesh.vertices) > 0:
                    # Convert mesh to point cloud
                    pcd = mesh.sample_points_uniformly(number_of_points=10000)
                else:
                    # Load as point cloud directly
                    pcd = o3d.io.read_point_cloud(str(ply_file))
            except:
                pcd = o3d.io.read_point_cloud(str(ply_file))
            
            if len(pcd.points) > 0:
                merged_pcd += pcd
                print(f"Loaded {ply_file.name}: {len(pcd.points)} points")
            else:
                print(f"Warning: No points in {ply_file.name}")
                
        except Exception as e:
            print(f"Error loading {ply_file}: {e}")
    
    if len(merged_pcd.points) == 0:
        return None
    
    return merged_pcd

def setup_visualizer() -> Tuple[o3d.visualization.Visualizer, any]:
    """Setup Open3D visualizer for rendering."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1080, visible=False)
    
    # Set render options
    render_option = vis.get_render_option()
    render_option.show_coordinate_frame = True
    render_option.background_color = np.array([0.1, 0.1, 0.1])
    render_option.point_size = 2.0
    
    return vis, render_option

def render_frame(vis: o3d.visualization.Visualizer, pcd: o3d.geometry.PointCloud, 
                frame_idx: int, rotation_speed: float = 0.5) -> np.ndarray:
    """Render a single frame and return the image."""
    vis.clear_geometries()
    vis.add_geometry(pcd)
    
    # Set camera view
    ctr = vis.get_view_control()
    
    # Rotate the camera around the scene
    angle = frame_idx * rotation_speed * np.pi / 180
    
    # Set camera parameters for a nice view
    parameters = o3d.camera.PinholeCameraParameters()
    
    # Calculate camera position in a circle around the point cloud
    center = pcd.get_center()
    bound = pcd.get_max_bound() - pcd.get_min_bound()
    radius = np.linalg.norm(bound) * 1.5
    
    # Camera position
    cam_x = center[0] + radius * np.cos(angle)
    cam_y = center[1] + radius * np.sin(angle)
    cam_z = center[2] + radius * 0.3
    
    # Look at the center
    eye = [cam_x, cam_y, cam_z]
    up = [0, 0, 1]
    
    ctr.set_lookat(center)
    ctr.set_up(up)
    ctr.set_front([center[0] - eye[0], center[1] - eye[1], center[2] - eye[2]])
    ctr.set_zoom(0.8)
    
    # Update and capture
    vis.poll_events()
    vis.update_renderer()
    
    # Capture image
    image = vis.capture_screen_float_buffer(do_render=True)
    image_np = np.asarray(image)
    
    # Convert from float [0,1] to uint8 [0,255]
    image_np = (image_np * 255).astype(np.uint8)
    
    return image_np

def create_video_from_frames(frames_dir: Path, output_path: Path, 
                           fps: int = 10, rotation_speed: float = 1.0,
                           ply_pattern: str = "*.ply") -> bool:
    """Create a video from .ply frame files."""
    
    # Find all frame directories
    frame_dirs = []
    for item in frames_dir.iterdir():
        if item.is_dir() and item.name.startswith('frame_'):
            frame_dirs.append(item)
    
    if not frame_dirs:
        print(f"No frame directories found in {frames_dir}")
        return False
    
    # Sort frame directories naturally
    frame_dirs.sort(key=lambda x: natural_sort_key(x.name))
    print(f"Found {len(frame_dirs)} frame directories")
    
    # Setup visualizer
    vis, render_option = setup_visualizer()
    
    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = None
    
    try:
        for frame_idx, frame_dir in enumerate(frame_dirs):
            print(f"Processing {frame_dir.name} ({frame_idx + 1}/{len(frame_dirs)})")
            
            # Find .ply files in this frame
            ply_files = find_ply_files(frame_dir)
            
            if not ply_files:
                print(f"  No .ply files found in {frame_dir.name}")
                continue
            
            # Load and merge point clouds
            merged_pcd = load_and_merge_ply_files(ply_files)
            
            if merged_pcd is None:
                print(f"  Failed to load point clouds from {frame_dir.name}")
                continue
            
            print(f"  Total points: {len(merged_pcd.points)}")
            
            # Render the frame
            image = render_frame(vis, merged_pcd, frame_idx, rotation_speed)
            
            # Initialize video writer with first frame dimensions
            if video_writer is None:
                height, width = image.shape[:2]
                video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
                print(f"Video dimensions: {width}x{height}")
            
            # Convert RGB to BGR for OpenCV
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Write frame to video
            video_writer.write(image_bgr)
            
            print(f"  Rendered frame {frame_idx}")
    
    except Exception as e:
        print(f"Error during video creation: {e}")
        return False
    
    finally:
        # Cleanup
        vis.destroy_window()
        if video_writer is not None:
            video_writer.release()
    
    print(f"Video saved to: {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Create video from .ply frame files')
    parser.add_argument('frames_dir', type=str, 
                       help='Directory containing frame subdirectories with .ply files')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output video path (default: frames_dir/pointcloud_video.mp4)')
    parser.add_argument('--fps', type=int, default=10,
                       help='Video frame rate (default: 10)')
    parser.add_argument('--rotation-speed', type=float, default=1.0,
                       help='Camera rotation speed in degrees per frame (default: 1.0)')
    parser.add_argument('--ply-pattern', type=str, default='*.ply',
                       help='Pattern to match .ply files (default: *.ply)')
    
    args = parser.parse_args()
    
    # Validate input directory
    frames_dir = Path(args.frames_dir)
    if not frames_dir.exists():
        print(f"Error: Directory {frames_dir} does not exist")
        sys.exit(1)
    
    if not frames_dir.is_dir():
        print(f"Error: {frames_dir} is not a directory")
        sys.exit(1)
    
    # Set output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = frames_dir / "pointcloud_video.mp4"
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Input directory: {frames_dir}")
    print(f"Output video: {output_path}")
    print(f"FPS: {args.fps}")
    print(f"Rotation speed: {args.rotation_speed} degrees/frame")
    print(f"PLY pattern: {args.ply_pattern}")
    
    # Create the video
    success = create_video_from_frames(
        frames_dir=frames_dir,
        output_path=output_path,
        fps=args.fps,
        rotation_speed=args.rotation_speed,
        ply_pattern=args.ply_pattern
    )
    
    if success:
        print("Video creation completed successfully!")
    else:
        print("Video creation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 