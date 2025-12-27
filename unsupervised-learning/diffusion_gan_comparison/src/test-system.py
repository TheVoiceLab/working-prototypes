import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA compiled with PyTorch: {torch.version.cuda}")
print(f"Is CUDA available: {torch.cuda.is_available()}")