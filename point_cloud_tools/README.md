# Point Cloud Tools for D3Fields

This directory contains minimal utilities and scripts for point cloud transformations in the D3Fields project.

## Overview

The point cloud processing pipeline consists of 4 main steps:

1. **Image + Depth → Point Cloud**: Convert RGB-D images to point clouds (saved as .npy)
2. **Visualization**: Convert point clouds to PLY files for visualization 
3. **Transformation**: Apply camera extrinsics to transform point clouds to world coordinates
4. **Multi-camera Fusion**: Merge point clouds from multiple cameras into a common coordinate frame

## Files

### Core Utilities
- `minimal_pointcloud_utils.py` - Core functions for point cloud operations

### Individual Scripts 
- `1_image_depth_to_pointcloud.py` - Convert image + depth to point cloud
- `2_visualize_pointcloud.py` - Create PLY visualization from numpy point cloud
- `3_transform_pointcloud.py` - Apply extrinsic transformations
- `4_merge_multi_camera_pointclouds.py` - Merge multiple camera point clouds

### Demo
- `demo_pointcloud_pipeline.py` - Complete demo showing all functionality

## Usage Examples

### Script 1: Image + Depth to Point Cloud

```bash
python 1_image_depth_to_pointcloud.py \
    /ssdstore/azadaia/project_snellius_sync/d3fields/data/2023-09-11-14-15-50-607452/camera_0/color/0.png \
    /ssdstore/azadaia/project_snellius_sync/d3fields/data/2023-09-11-14-15-50-607452/camera_0/depth/0.png \
    /ssdstore/azadaia/project_snellius_sync/d3fields/data/2023-09-11-14-15-50-607452/camera_0 \
    point_cloud_tools/output/camera0_frame0_points.npy
```

**Output:**
- `camera0_frame0_points.npy` - Point cloud in camera coordinates (N, 3)
- `camera0_frame0_points_colors.npy` - RGB colors for each point (N, 3)

### Script 2: Visualize Point Cloud

```bash
python 2_visualize_pointcloud.py \
    point_cloud_tools/output/camera0_frame0_points.npy \
    point_cloud_tools/output/camera0_frame0_points.ply \
    point_cloud_tools/output/camera0_frame0_points_colors.npy
```

**Output:**
- `camera0_frame0_points.ply` - PLY file for visualization

### Script 3: Transform Point Cloud

```bash
python 3_transform_pointcloud.py \
    point_cloud_tools/output/camera0_frame0_points.npy \
    /ssdstore/azadaia/project_snellius_sync/d3fields/data/2023-09-11-14-15-50-607452/camera_0 \
    point_cloud_tools/output/camera0_frame0_transformed.npy \
    point_cloud_tools/output/camera0_frame0_points_colors.npy
```

**Options:**
- `--adamanip` - Use AdaManip data format adjustments
- `--no_visualization` - Skip PLY output

**Output:**
- `camera0_frame0_transformed.npy` - Transformed points in world coordinates
- `camera0_frame0_transformed.ply` - Visualization
- `camera0_frame0_transformed_comparison.ply` - Side-by-side comparison

### Script 4: Merge Multi-Camera Point Clouds

```bash
# Standard data format
python 4_merge_multi_camera_pointclouds.py \
    /ssdstore/azadaia/project_snellius_sync/d3fields/data/2023-09-11-14-15-50-607452 \
    0 \
    point_cloud_tools/output/merged_frame0.npy \
    --num_cameras 4

# AdaManip data format  
python 4_merge_multi_camera_pointclouds.py \
    /ssdstore/azadaia/project_snellius_sync/d3fields/output/adamanip_d3fields/OpenCoffeeMachine/grasp_env_0 \
    0 \
    point_cloud_tools/output/merged_coffee_machine.npy \
    --num_cameras 2 \
    --adamanip
```

**Options:**
- `--num_cameras N` - Number of cameras (default: 4)
- `--adamanip` - Use AdaManip data format
- `--max_depth X` - Maximum depth threshold (default: 2.5)
- `--downsample X` - Voxel size for downsampling
- `--no_visualization` - Skip PLY output

**Output:**
- `merged_frame0.npy` - Merged point cloud in world coordinates
- `merged_frame0_colors.npy` - Merged colors
- `merged_frame0.ply` - Visualization with original colors
- `merged_frame0_per_camera.ply` - Visualization with per-camera colors

## Demo

Run the complete demo to test all functionality:

```bash
python demo_pointcloud_pipeline.py
```

This will:
1. Process single camera data (camera_0)
2. Process multi-camera data (all 4 cameras)
3. Process AdaManip data format (OpenCoffeeMachine)

Output will be saved to `point_cloud_tools/output/demo/`

## Core Functions

### `minimal_pointcloud_utils.py`

Key functions:

- `load_camera_data(camera_dir)` - Load camera parameters and extrinsics
- `depth_to_pointcloud(depth, mask, camera_params)` - Convert depth image to 3D points
- `transform_pointcloud(points, extrinsics)` - Apply 4x4 transformation matrix
- `image_depth_to_pointcloud(image_path, depth_path, camera_params)` - Complete RGB-D to point cloud pipeline
- `save_pointcloud_npy(points, filepath)` - Save as numpy array
- `save_pointcloud_ply(points, filepath, colors)` - Save as PLY for visualization
- `merge_pointclouds(pointclouds, colors_list)` - Merge multiple point clouds

## Data Formats

### Standard D3Fields Data
```
data/2023-09-11-14-15-50-607452/
├── camera_0/
│   ├── color/0.png, 1.png, ...
│   ├── depth/0.png, 1.png, ...
│   ├── camera_params.npy      # [fx, fy, cx, cy]
│   └── camera_extrinsics.npy  # 4x4 world-to-camera matrix
├── camera_1/
└── ...
```

### AdaManip Data Format
```
output/adamanip_d3fields/OpenCoffeeMachine/grasp_env_0/
├── camera_0/
│   ├── color/0.png, 1.png, ...
│   ├── depth/0.png, 1.png, ...
│   ├── camera_params.npy      # [fx, fy, cx, cy] 
│   └── camera_extrinsics.npy  # 4x4 matrix (requires special handling)
├── camera_1/
└── ...
```

## Camera Coordinate Systems

### Standard Format
- Extrinsics: World-to-camera transformation
- Transform to world: `world_points = inv(extrinsics) @ camera_points`

### AdaManip Format  
- Extrinsics: Requires Z-flip and transpose
- Transform to world: `world_points = (extrinsics @ F).T @ camera_points`
- Where `F = diag([1, 1, -1, 1])`

## Visualization

Generated PLY files can be viewed with:

1. **Open3D (Python)**:
   ```python
   import open3d as o3d
   pcd = o3d.io.read_point_cloud('file.ply')
   o3d.visualization.draw_geometries([pcd])
   ```

2. **Command line**:
   ```bash
   python -c "import open3d as o3d; o3d.visualization.draw_geometries([o3d.io.read_point_cloud('file.ply')])"
   ```

3. **External viewers**: MeshLab, CloudCompare, PCL viewer

## Dependencies

Required Python packages:
- numpy
- opencv-python (cv2)
- open3d-python
- pathlib (built-in Python 3.4+)

## Output Structure

```
point_cloud_tools/output/
├── demo/                              # Demo outputs
│   ├── camera0_points.npy            # Single camera point cloud
│   ├── camera0_points.ply            # Single camera visualization
│   ├── camera0_world_points.npy      # Transformed to world coordinates
│   ├── merged_all_cameras.npy        # Multi-camera merged
│   ├── merged_all_cameras.ply        # Multi-camera visualization
│   ├── merged_per_camera_colors.ply  # Per-camera color coding
│   └── adamanip_coffee_machine.ply   # AdaManip data result
└── [custom outputs from scripts]
```

## Tips

1. **Depth filtering**: Use `--max_depth` to filter out background/invalid depths
2. **Memory management**: For large point clouds, use `--downsample` for voxel downsampling
3. **Debugging**: Check point cloud statistics in script outputs to verify transformations
4. **AdaManip data**: Always use `--adamanip` flag for AdaManip datasets
5. **Visualization**: Use per-camera color coding to debug multi-camera alignment 