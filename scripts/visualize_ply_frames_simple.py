#!/usr/bin/env python3
"""
Simple script to visualize .ply frame files and create a video.
Uses trimesh and matplotlib for compatibility.
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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

try:
    import trimesh
except ImportError:
    print("Error: trimesh is required. Install with: pip install trimesh")
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

def load_and_merge_ply_files(ply_files: List[Path]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load multiple .ply files and merge them into point cloud data."""
    if not ply_files:
        return None
    
    all_vertices = []
    all_colors = []
    
    for ply_file in ply_files:
        try:
            # Load mesh using trimesh
            mesh = trimesh.load(str(ply_file))
            
            if hasattr(mesh, 'vertices') and len(mesh.vertices) > 0:
                vertices = np.array(mesh.vertices)
                
                # Get colors if available
                if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
                    colors = np.array(mesh.visual.vertex_colors)[:, :3] / 255.0  # RGB only, normalize
                elif hasattr(mesh.visual, 'face_colors') and mesh.visual.face_colors is not None:
                    # If we have face colors, sample points from the mesh and use face colors
                    if hasattr(mesh, 'sample'):
                        sampled_points, face_indices = mesh.sample(min(len(vertices), 10000), return_index=True)
                        vertices = sampled_points
                        colors = np.array(mesh.visual.face_colors)[face_indices][:, :3] / 255.0
                    else:
                        colors = np.tile([0.7, 0.7, 0.7], (len(vertices), 1))  # Default gray
                else:
                    colors = np.tile([0.7, 0.7, 0.7], (len(vertices), 1))  # Default gray
                
                all_vertices.append(vertices)
                all_colors.append(colors)
                print(f"Loaded {ply_file.name}: {len(vertices)} points")
            else:
                print(f"Warning: No vertices in {ply_file.name}")
                
        except Exception as e:
            print(f"Error loading {ply_file}: {e}")
    
    if not all_vertices:
        return None
    
    # Combine all vertices and colors
    combined_vertices = np.vstack(all_vertices)
    combined_colors = np.vstack(all_colors)
    
    return combined_vertices, combined_colors

def render_frame_matplotlib(vertices: np.ndarray, colors: np.ndarray, 
                          frame_idx: int, rotation_speed: float = 1.0,
                          width: int = 1920, height: int = 1080, 
                          total_frames: int = 6) -> np.ndarray:
    """Render a single frame using matplotlib with dynamic camera movement."""
    
    # Create figure
    fig = plt.figure(figsize=(width/100, height/100), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    # Subsample points if too many (for performance)
    if len(vertices) > 50000:
        indices = np.random.choice(len(vertices), 50000, replace=False)
        vertices = vertices[indices]
        colors = colors[indices]
    
    # Plot points
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
              c=colors, s=0.1, alpha=0.8)
    
    # Dynamic camera movement
    progress = frame_idx / max(total_frames - 1, 1)  # Normalize to 0-1
    
    # Azimuth: Full rotation around the scene
    azim = frame_idx * rotation_speed
    
    # Elevation: Oscillating elevation for more dynamic movement
    elev_base = 20
    elev_amplitude = 15
    elev_period = total_frames / 2  # Complete one oscillation cycle
    elev = elev_base + elev_amplitude * np.sin(2 * np.pi * frame_idx / elev_period)
    
    # Distance/Zoom: Vary the viewing distance for zoom effect
    distance_base = 1.5
    distance_amplitude = 0.3
    distance_period = total_frames / 1.5  # Slightly different period for complexity
    distance_factor = distance_base + distance_amplitude * np.cos(2 * np.pi * frame_idx / distance_period)
    
    # Calculate point cloud bounds for camera positioning
    center = vertices.mean(axis=0)
    bounds = vertices.max(axis=0) - vertices.min(axis=0)
    max_bound = np.max(bounds)
    
    # Set the view
    ax.view_init(elev=elev, azim=azim)
    
    # Set the viewing box with dynamic zoom
    zoom_factor = 0.8 / distance_factor
    box_size = max_bound * distance_factor
    
    ax.set_xlim([center[0] - box_size/2, center[0] + box_size/2])
    ax.set_ylim([center[1] - box_size/2, center[1] + box_size/2]) 
    ax.set_zlim([center[2] - box_size/2, center[2] + box_size/2])
    
    # Set equal aspect ratio and remove axes
    ax.set_box_aspect([1,1,1])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    # Set background
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)
    
    # Add a subtle background color gradient effect
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    
    # Convert to image
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    
    return buf

def create_video_from_frames(frames_dir: Path, output_path: Path, 
                           fps: int = 8, rotation_speed: float = 3.0) -> bool:
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
            result = load_and_merge_ply_files(ply_files)
            
            if result is None:
                print(f"  Failed to load point clouds from {frame_dir.name}")
                continue
            
            vertices, colors = result
            print(f"  Total points: {len(vertices)}")
            
            # Render the frame with dynamic camera movement
            image = render_frame_matplotlib(vertices, colors, frame_idx, rotation_speed, 
                                          total_frames=len(frame_dirs))
            
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
        if video_writer is not None:
            video_writer.release()
    
    print(f"Video saved to: {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Create video from .ply frame files (simple version)')
    parser.add_argument('frames_dir', type=str, 
                       help='Directory containing frame subdirectories with .ply files')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output video path (default: frames_dir/pointcloud_video.mp4)')
    parser.add_argument('--fps', type=int, default=8,
                       help='Video frame rate (default: 8 for slower video)')
    parser.add_argument('--rotation-speed', type=float, default=3.0,
                       help='Camera rotation speed in degrees per frame (default: 3.0)')
    
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
    print("Camera movements: Orbital rotation + Elevation oscillation + Dynamic zoom")
    
    # Create the video
    success = create_video_from_frames(
        frames_dir=frames_dir,
        output_path=output_path,
        fps=args.fps,
        rotation_speed=args.rotation_speed
    )
    
    if success:
        print("Video creation completed successfully!")
    else:
        print("Video creation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 