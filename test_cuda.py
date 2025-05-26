#!/usr/bin/env python3

import torch
import sys

print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU count:", torch.cuda.device_count())
    print("GPU name:", torch.cuda.get_device_name(0))

# Test basic tensor operations
try:
    x = torch.randn(3, 3)
    if torch.cuda.is_available():
        x_gpu = x.cuda()
        print("GPU tensor test successful")
    print("Basic PyTorch test successful")
except Exception as e:
    print("Error:", e) 