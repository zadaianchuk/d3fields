# D3Fields Processing for AdaManip Dataset

This repository provides a complete pipeline for processing the AdaManip D3Fields dataset using the D3Fields fusion system. The main script (`main.py`) computes D3Fields representations for all objects in the dataset.

## 🚀 Quick Start

### Basic Usage
```bash
# Process all tasks with default settings
python main.py

# Quick test with 2 tasks
python main.py --quick_test

# Process specific tasks
python main.py --tasks OpenBottle OpenDoor --max_envs 2

# Use CPU instead of GPU
python main.py --device cpu
```

### With PyTorch/GPU (Recommended)
```bash
# Install dependencies
pip install torch torchvision

# Run with GPU acceleration
python main.py --device cuda:0 --max_envs 5
```

## 📁 Core Files

- **`main.py`** - Main processing script for D3Fields computation
- **`d3fields_data_loader.py`** - Streamlined data loader
- **`analyze_results.py`** - Results analysis and visualization
- **`d3fields_utils.py`** - Utility functions

## ⚙️ Features

### Main Processing (`main.py`)
- **🎯 Task-Specific Queries** - Optimized text queries for each manipulation task
- **📦 Complete Pipeline** - Data loading → Fusion → Segmentation → Feature extraction
- **🛡️ Robust Error Handling** - Graceful failure handling with detailed logs
- **🎭 Simulation Mode** - Works without PyTorch for testing
- **📊 Progress Tracking** - Real-time progress and success statistics
- **💾 Comprehensive Output** - Point clouds, features, metadata, and visualizations

### Output Structure
```
d3fields_results/
├── processing_results.json          # Overall results summary
├── OpenBottle/
│   └── grasp_env_0/
│       ├── d3fields_metadata.json   # Environment metadata
│       ├── camera_0_color.png       # Original images
│       ├── camera_1_color.png
│       ├── pointcloud_instance_1.npy # Point clouds per instance
│       ├── pointcloud_instance_2.npy
│       └── pointcloud_instance_3.npy
└── OpenDoor/...
```

## 🎯 Task Configuration

Each manipulation task uses specialized queries:

| Task | Queries |
|------|---------|
| **OpenBottle** | bottle, cap, lid, robotic arm, gripper |
| **OpenDoor** | door, handle, doorknob, robotic arm, gripper |
| **OpenSafe** | safe, door, handle, lock, robotic arm, gripper |
| **OpenCoffeeMachine** | coffee machine, lid, top, robotic arm, gripper |
| **OpenWindow** | window, handle, frame, robotic arm, gripper |
| **OpenPressureCooker** | pressure cooker, lid, handle, robotic arm, gripper |

## 📊 Analysis

### View Results
```bash
# Analyze processing results
python analyze_results.py

# Analyze specific results directory
python analyze_results.py --results_dir my_results
```

### Example Output
```
📊 Overall Results:
   Total environments: 18
   ✅ Successful: 18
   ❌ Failed: 0
   📈 Success rate: 100.0%
```

## 🔧 Command Line Options

```bash
python main.py [OPTIONS]

Options:
  --output_dir DIR        Output directory (default: d3fields_results)
  --max_envs N           Max environments per type (default: 3)
  --env_types TYPE       Environment types: grasp, manip (default: both)
  --device DEVICE        Device: cuda:0, cpu (default: cuda:0)
  --tasks TASK [TASK...] Specific tasks to process (default: all)
  --quick_test           Quick test mode (1 env per task)
  --simulation           Force simulation mode
```

## 🎭 Simulation Mode

When PyTorch is not available, the system runs in simulation mode:
- ✅ Tests complete pipeline functionality
- ✅ Generates realistic dummy data
- ✅ Validates data loading and saving
- ⚠️ Does not compute actual D3Fields

## 🛠️ Dependencies

### Required
- `numpy`
- `opencv-python`
- `Pillow`
- `pathlib`

### Optional (for real processing)
- `torch` + `torchvision`
- `matplotlib` (for visualization)

### GPU Requirements
- CUDA-compatible GPU (recommended)
- 6GB+ GPU memory for full processing

## 📈 Performance

### Processing Speed
- **GPU**: ~2-5 environments/minute
- **CPU**: ~0.5-1 environments/minute
- **Simulation**: ~10-20 environments/minute

### Memory Usage
- **GPU Memory**: ~4-6GB
- **System Memory**: ~2-4GB

## 🔍 Workspace Configuration

Default workspace boundaries (adjust in `main.py`):
```python
DEFAULT_BOUNDARIES = {
    'x_lower': -0.6, 'x_upper': 0.6,
    'y_lower': -0.6, 'y_upper': 0.6,
    'z_lower': 0.0, 'z_upper': 1.2
}
```

## 📋 Example Workflows

### Full Dataset Processing
```bash
# Process entire dataset with moderate settings
python main.py --max_envs 3 --env_types grasp manip --device cuda:0

# High-throughput processing
python main.py --max_envs 10 --device cuda:0

# CPU-only processing
python main.py --max_envs 2 --device cpu
```

### Selective Processing
```bash
# Process only bottle and door tasks
python main.py --tasks OpenBottle OpenDoor --device cuda:0

# Process only grasp environments
python main.py --env_types grasp --max_envs 5

# Single task, single environment
python main.py --tasks OpenBottle --max_envs 1 --env_types grasp
```

### Development/Testing
```bash
# Quick functionality test
python main.py --quick_test

# Simulation mode test
python main.py --simulation --quick_test

# Debug single task
python main.py --tasks OpenBottle --max_envs 1 --env_types grasp
```
