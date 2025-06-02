# PCA Precomputation for AdaManip Objects

This document explains how to create proper PCA models for each object type in the AdaManip dataset using DINOv2 features extracted from masked image patches.

## Overview

The system extracts DINOv2 features from object regions (defined by masks) across all environments and creates object-specific PCA models for feature visualization. This replaces the dummy PCA models with proper, data-driven ones.

## Files

- `scripts/our_precompute_pca.py` - Main script for PCA precomputation
- `scripts/test_pca_precompute.py` - Test script for verification
- `scripts/visualize_pca.py` - Analysis and visualization tool
- `main.py` - Updated to use proper PCA models (with fallback to dummy)

## Object Types

The system processes the following object types from the AdaManip dataset:

- `bottle` (from OpenBottle tasks)
- `door` (from OpenDoor tasks)
- `safe` (from OpenSafe tasks)
- `coffee_machine` (from OpenCoffeeMachine tasks)
- `window` (from OpenWindow tasks)
- `pressure_cooker` (from OpenPressureCooker tasks)
- `pen` (from OpenPen tasks)
- `lamp` (from OpenLamp tasks)
- `microwave` (from OpenMicroWave tasks)

## Usage

### Basic Usage (Recommended)

Run PCA precomputation for all object types using the wrapper script:

```bash
python run_pca_precompute.py
```

Or run directly from scripts folder:

```bash
python scripts/our_precompute_pca.py
```

### Advanced Options

```bash
python scripts/our_precompute_pca.py \
    --data_root /path/to/adamanip_d3fields \
    --output_dir pca_model \
    --model_name dinov2_vitl14 \
    --device cuda \
    --target_size 224 \
    --max_samples 100 \
    --object_types bottle door safe
```

#### Parameters

- `--data_root`: Path to the adamanip_d3fields directory (default: hardcoded path)
- `--output_dir`: Output directory for PCA models (default: `pca_model`)
- `--model_name`: DINOv2 model to use (choices: `dinov2_vits14`, `dinov2_vitb14`, `dinov2_vitl14`, `dinov2_vitg14`)
- `--device`: Device to run on (default: `cuda`)
- `--target_size`: Target image size for processing (default: 224)
- `--max_samples`: Maximum samples per object type (default: 50)
- `--object_types`: Specific object types to process (default: all)

### Testing

Test the system with a subset of data using the wrapper:

```bash
python test_pca.py
```

Or run directly from scripts folder:

```bash
python scripts/test_pca_precompute.py
```

### Visualization

Visualize and analyze created PCA models:

```bash
python scripts/visualize_pca.py --object_name bottle
```

## How It Works

### 1. Data Collection
- Iterates through all environments for each task
- Loads color images and corresponding masks
- Extracts object regions using auto-detected main object ID (largest area)
- Crops to bounding box with padding
- Collects up to `max_samples` per object type

### 2. Feature Extraction
- Processes images in batches using DINOv2
- Resizes images to target size (224x224)
- Extracts patch-level features using `forward_features()`
- Creates binary masks for foreground/background patches

### 3. PCA Training
- Combines features from all samples of an object type
- Uses only foreground patches (where mask = main object)
- Fits PCA with 3 components for RGB visualization
- Saves explained variance statistics

### 4. Model Saving
- Saves PCA model as pickle file: `{object_name}_{model_name}.pkl`
- Includes metadata: explained variance, sample count, etc.
- Creates summary text file with statistics

## Output Structure

```
pca_model/
├── bottle_dinov2_vitl14.pkl           # PCA model for bottles
├── bottle_dinov2_vitl14_summary.txt   # Summary statistics
├── door_dinov2_vitl14.pkl             # PCA model for doors
├── door_dinov2_vitl14_summary.txt     # Summary statistics
├── ...
└── pca_results_dinov2_vitl14.json     # Overall results summary
```

## Integration with Main Pipeline

The `main.py` script has been updated to automatically load proper PCA models:

```python
# In create_meshes_and_save()
object_name = OBJECT_TYPE_MAPPING.get(task, task.lower())
pca_model = create_simple_pca_model(object_name)
```

The system:
1. Tries to load the proper PCA model for the object type
2. Falls back to dummy PCA if model not found
3. Uses the PCA for feature mesh visualization

## Requirements

- PyTorch with CUDA support
- torchvision
- scikit-learn
- OpenCV (cv2)
- PIL/Pillow
- NumPy

## Expected Performance

- **Processing time**: ~5-10 minutes per object type (depends on samples)
- **Memory usage**: ~2-4GB GPU memory for DINOv2-L
- **Storage**: ~1-5MB per PCA model

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use smaller DINOv2 model
2. **No samples found**: Check data paths and mask files
3. **PCA fails**: Ensure sufficient foreground features are extracted

### Debug Tips

- Run with `--max_samples 10` for quick testing
- Check the summary files for statistics
- Use `scripts/test_pca_precompute.py` for debugging

## Example Usage Workflow

1. **Precompute PCA models** (using wrapper script):
   ```bash
   python run_pca_precompute.py
   ```

   Or with custom parameters:
   ```bash
   python scripts/our_precompute_pca.py --max_samples 100
   ```

2. **Verify models were created**:
   ```bash
   ls pca_model/
   ```

3. **Test the PCA system** (optional):
   ```bash
   python test_pca.py
   ```

4. **Run D3Fields with proper PCA**:
   ```bash
   python main.py --use_gt_masks --tasks OpenBottle --max_envs 1
   ```

5. **Check feature mesh quality** in the output PLY files

The feature meshes should now show more meaningful, object-specific color patterns compared to the random dummy PCA. 