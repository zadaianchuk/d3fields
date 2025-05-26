#!/usr/bin/env python3

try:
    print("Testing basic imports...")
    import torch
    print(f"PyTorch {torch.__version__} imported successfully")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    print("Testing fusion module...")
    from fusion import Fusion
    print("Fusion module imported successfully")
    
    print("Testing visualization module...")
    import vis_repr
    print("Visualization module imported successfully")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc() 