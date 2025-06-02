# PCA Integration Summary

This document summarizes the complete integration of the PCA precomputation system after moving files to the `scripts/` folder.

## ✅ Files Structure

```
d3fields/
├── main.py                           # Updated with OBJECT_TYPE_MAPPING and PCA loading
├── run_pca_precompute.py             # Wrapper to run PCA from main directory
├── test_pca.py                       # Wrapper to test PCA from main directory  
├── test_integration.py               # Integration test script
├── scripts/
│   ├── our_precompute_pca.py         # Main PCA precomputation script
│   ├── test_pca_precompute.py        # PCA test script
│   ├── visualize_pca.py              # PCA analysis and visualization
│   └── PCA_PRECOMPUTE_README.md      # Updated documentation
└── pca_model/                        # Output directory for PCA models (created when run)
    ├── bottle_dinov2_vitl14.pkl
    ├── door_dinov2_vitl14.pkl
    └── ...
```

## 🔧 Integration Points

### 1. Object Type Mapping
- **Location**: Defined in `main.py` (lines 32-43)
- **Purpose**: Maps task names to object categories
- **Usage**: Used by both main pipeline and PCA scripts

```python
OBJECT_TYPE_MAPPING = {
    'OpenBottle': 'bottle',
    'OpenDoor': 'door', 
    'OpenSafe': 'safe',
    'OpenCoffeeMachine': 'coffee_machine',
    'OpenWindow': 'window',
    'OpenPressureCooker': 'pressure_cooker',
    'OpenPen': 'pen',
    'OpenLamp': 'lamp',
    'OpenMicroWave': 'microwave'
}
```

### 2. PCA Model Loading
- **Location**: `main.py` function `create_simple_pca_model()` (lines ~380-410)
- **Behavior**: 
  1. Tries to load proper PCA model from `pca_model/{object_name}_{model_name}.pkl`
  2. Falls back to dummy PCA model if not found
- **Integration**: Called automatically during mesh creation in `create_meshes_and_save()`

### 3. Auto-Object Detection
- **Location**: `scripts/our_precompute_pca.py`
- **Feature**: Automatically detects main object ID in masks (largest area)
- **Benefit**: Works with varying object IDs (2, 7, etc.) instead of hardcoded ID=1

## 🚀 Usage Instructions

### Option 1: Using Wrapper Scripts (Recommended)

```bash
# 1. Precompute PCA models
python run_pca_precompute.py

# 2. Test the system (optional)  
python test_pca.py

# 3. Run D3Fields with proper PCA
python main.py --use_gt_masks --tasks OpenBottle --max_envs 1
```

### Option 2: Direct Script Execution

```bash
# 1. Precompute PCA models
python scripts/our_precompute_pca.py --max_samples 100

# 2. Test the system
python scripts/test_pca_precompute.py

# 3. Visualize results
python scripts/visualize_pca.py --object_name bottle
```

## 🎯 Key Benefits

1. **Proper PCA Models**: Real DINOv2 features instead of dummy models
2. **Object-Specific**: Different PCA per object type for better visualization
3. **Automatic Integration**: Main pipeline loads proper models automatically
4. **Flexible Object Detection**: Works with any object ID in masks
5. **Easy to Use**: Wrapper scripts for convenient execution
6. **Comprehensive**: Covers all 9 AdaManip object types

## 📋 Integration Checklist

- ✅ Files moved to `scripts/` folder
- ✅ `main.py` has `OBJECT_TYPE_MAPPING` defined
- ✅ `main.py` has updated `create_simple_pca_model()` function
- ✅ Wrapper scripts created for easy execution
- ✅ Documentation updated for new file locations
- ✅ Auto-object detection implemented
- ✅ Integration test script created
- ✅ JSON serialization fix applied (resolves numpy int64 errors)

## 🔧 Recent Fixes

### JSON Serialization Fix
- **Issue**: `TypeError: Object of type int64 is not JSON serializable` when saving PCA results
- **Cause**: NumPy types (`int64`, `float64`, `ndarray`) are not JSON serializable by default
- **Solution**: Added `convert_to_json_serializable()` helper function that converts:
  - `np.integer` → `int`
  - `np.floating` → `float`  
  - `np.ndarray` → `list`
  - Recursively handles nested dictionaries and lists
- **Location**: `scripts/our_precompute_pca.py` (lines ~17-30)
- **Usage**: Applied automatically when saving PCA results to JSON

## 🔍 Verification

To verify the integration works:

1. **Check file structure**: All scripts in `scripts/` folder
2. **Run integration test**: `python test_integration.py` (requires PyTorch)
3. **Test PCA creation**: `python run_pca_precompute.py`
4. **Verify PCA loading**: Run main pipeline and check for "Loaded proper PCA model" messages

## 🎉 Result

The PCA system is now fully integrated and will:
- Generate proper, object-specific PCA models
- Automatically load them during D3Fields processing
- Fall back gracefully to dummy models if needed
- Provide better feature mesh visualizations with meaningful colors

The integration maintains backward compatibility while providing significantly improved feature visualization quality. 