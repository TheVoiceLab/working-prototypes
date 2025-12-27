import torch
import sys
import os

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA Runtime initialized: {torch.cuda.is_initialized()}")

# Check if the driver is actually visible to the OS
try:
    import subprocess
    nvm = subprocess.check_output(["nvidia-smi"]).decode()
    print("NVIDIA-SMI: Success")
except:
    print("NVIDIA-SMI: Failed (Driver path not in System PATH)")

print(f"CUDA Available: {torch.cuda.is_available()}")