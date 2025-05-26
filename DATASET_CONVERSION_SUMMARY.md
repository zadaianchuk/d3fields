# AdaManip to D³Fields Dataset Conversion Summary

## Successfully Converted Datasets

### 1. Microwave Dataset (PNG Samples)
- **Source**: `/home/azadaianchuk/projects/AdaManip/demo_data/rgbd_manip_OpenMicroWave_adaptive_10_eps25_clock0.5/png_samples`
- **Output**: `data/microwave_dataset/`
- **Images**: 10 timesteps, 4 cameras each
- **Resolution**: ~591×281 pixels per camera view (split from 1182×562 composite)
- **Note**: Zarr data contains placeholder data (1×1 pixels), so PNG samples were used

### 2. Coffee Machine Dataset (Zarr Data) ⭐ **RECOMMENDED**
- **Source**: `/home/azadaianchuk/projects/AdaManip/demo_data/rgbd_grasp_OpenCoffeeMachine_7_eps20_clock1.0/rgbd_demo_data.zarr`
- **Output**: `data/coffee_machine_dataset_full/`
- **Images**: 42 timesteps (sampled every 20 from 840 total), 3 cameras each
- **Resolution**: 512×512 pixels per camera view
- **Features**: ✅ **Real RGB data**, ✅ **Real depth data**, ✅ **Multi-camera setup**

### 3. Coffee Machine Dataset (PNG Samples) - Legacy
- **Source**: `/home/azadaianchuk/projects/AdaManip/demo_data/rgbd_grasp_OpenCoffeeMachine_7_eps20_clock1.0/png_samples`
- **Output**: `data/coffee_machine_dataset/`
- **Images**: 10 timesteps, 4 cameras each (split from composite)
- **Resolution**: 590×886 pixels per camera view

## Dataset Structure Created

All datasets follow the D³Fields format:
```
dataset_name/
├── camera_0/
│   ├── color/
│   │   ├── 0.png
│   │   ├── 1.png
│   │   └── ...
│   ├── depth/
│   │   ├── 0.png
│   │   ├── 1.png
│   │   └── ...
│   ├── camera_extrinsics.npy  # 4×4 transformation matrix
│   └── camera_params.npy      # [fx, fy, cx, cy]
├── camera_1/
├── camera_2/
└── camera_3/ (if applicable)
```

## Key Improvements with Zarr Data

### ✅ **Real Multi-Camera Data**
- Coffee machine dataset has **3 actual cameras** (not split composite images)
- Each camera provides a genuine different viewpoint
- **512×512 resolution** - much higher quality than split composites

### ✅ **Actual Depth Data**
- Real depth measurements from simulation/sensors
- Depth values in meters, converted to millimeters for PNG storage
- No more dummy depth data based on grayscale intensity

### ✅ **Temporal Consistency**
- 840 total timesteps available (sampled to 42 for efficiency)
- Consistent camera poses across time
- Better for tracking and temporal analysis

## Current Status & Next Steps

### ✅ **Completed**
- Multi-camera RGB extraction from zarr
- Real depth data extraction
- Proper camera parameter estimation (fx=400, fy=400, cx=256, cy=256 for 512×512 images)
- Example camera extrinsics (3 cameras positioned around scene)

### ⚠️ **Still Need Attention**
1. **Camera Extrinsics**: Using example poses - replace with actual camera positions if available
2. **Camera Parameters**: Estimated values - fine-tune if you have calibration data
3. **Depth Value Handling**: Some invalid depth values cause warnings (NaN/inf values)

## How to Use with D³Fields

### Step 1: Create PCA Model
```bash
# Modify scripts/prepare_pca.py with your object type
python scripts/prepare_pca.py
```

### Step 2: Run D³Fields Visualization
```bash
python vis_repr_custom.py \
    --data_path data/coffee_machine_dataset_full \
    --pca_path pca_model/coffee_machine.pkl \
    --query_texts "coffee machine" \
    --query_thresholds 0.3 \
    --x_lower -0.5 --x_upper 0.5 \
    --y_lower -0.5 --y_upper 0.5 \
    --z_lower -0.3 --z_upper 0.3
```

## Conversion Scripts Created

### 1. `convert_adamanip_zarr_to_d3fields.py` ⭐ **NEW & RECOMMENDED**
- Extracts actual multi-camera RGB and depth data from zarr files
- Handles real depth measurements
- Supports sampling and frame limiting
- Proper camera parameter estimation

### 2. `convert_adamanip_to_d3fields.py`
- General script for zarr and image directory modes
- Fallback for various data formats

### 3. `convert_png_samples_to_d3fields.py`
- For PNG sample files when zarr data is not available
- Splits composite images into camera views

## Usage Examples

### Convert Zarr Data (Recommended)
```bash
python convert_adamanip_zarr_to_d3fields.py \
    --source /path/to/data.zarr \
    --output data/my_dataset \
    --sample_every 20 \
    --max_frames 100
```

### Convert PNG Samples (Fallback)
```bash
python convert_png_samples_to_d3fields.py \
    --source /path/to/png_samples \
    --output data/my_dataset \
    --num_cameras 4
```

## Data Quality Comparison

| Dataset | Source | Cameras | Resolution | RGB Quality | Depth Quality |
|---------|--------|---------|------------|-------------|---------------|
| Coffee Machine (Zarr) | Real multi-cam | 3 | 512×512 | ✅ Real | ✅ Real |
| Coffee Machine (PNG) | Split composite | 4 | 590×886 | ⚠️ Split | ❌ Dummy |
| Microwave (PNG) | Split composite | 4 | 591×281 | ⚠️ Split | ❌ Dummy |

## Recommendations

1. **Use zarr data when available** - Much better quality and real multi-camera setup
2. **Coffee machine dataset is ideal** for testing D³Fields due to real depth data
3. **Check for more zarr datasets** in your AdaManip data
4. **Fine-tune camera parameters** if you have calibration data
5. **Test with D³Fields** to validate the conversion quality

## Files Created
- `data/coffee_machine_dataset_full/` - **BEST**: Real multi-camera zarr data (42 frames, 3 cameras)
- `data/coffee_machine_dataset_zarr/` - Test zarr data (10 frames, 3 cameras)
- `data/coffee_machine_dataset/` - PNG samples (10 frames, 4 cameras)
- `data/microwave_dataset/` - PNG samples (10 frames, 4 cameras)
- `convert_adamanip_zarr_to_d3fields.py` - **NEW**: Zarr-specific conversion script
- `convert_adamanip_to_d3fields.py` - General conversion script
- `convert_png_samples_to_d3fields.py` - PNG-specific conversion script 