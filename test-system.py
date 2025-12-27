import torch
import time
import numpy as np


def run_test():
    print("-" * 30)
    print("RTX 5090 SYSTEM CHECK")
    print("-" * 30)

    # 1. Hardware & Driver Check
    is_available = torch.cuda.is_available()
    print(f"CUDA Available: {is_available}")

    if not is_available:
        print("ERROR: CUDA not detected. Check your drivers and requirements.txt installation.")
        return

    device_name = torch.cuda.get_device_name(0)
    cuda_version = torch.version.cuda
    arch_list = torch.cuda.get_arch_list()

    print(f"GPU Model: {device_name}")
    print(f"PyTorch CUDA Version: {cuda_version}")
    print(f"Supported Architectures: {arch_list}")

    # 2. Memory Check
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Total VRAM: {total_mem:.2f} GB")

    # 3. Floating Point Performance Test (FP16/Tensor Cores)
    print("\nRunning Tensor Core Stress Test (FP16 MatMul)...")
    size = 10000  # Large matrix size
    a = torch.randn(size, size, device='cuda', dtype=torch.float16)
    b = torch.randn(size, size, device='cuda', dtype=torch.float16)

    # Warmup
    for _ in range(10):
        _ = torch.matmul(a, b)

    torch.cuda.synchronize()
    start_time = time.time()

    iterations = 50
    for _ in range(iterations):
        _ = torch.matmul(a, b)

    torch.cuda.synchronize()
    end_time = time.time()

    avg_time = (end_time - start_time) / iterations
    tflops = (2 * size ** 3) / (avg_time * 1e12)

    print(f"Average FP16 MatMul Time: {avg_time * 1000:.2f} ms")
    print(f"Estimated Throughput: {tflops:.2f} TFLOPS")

    # 4. BF16 Verification (Crucial for Blackwell)
    print("\nChecking BF16 Compatibility...")
    try:
        c = torch.randn(100, 100, device='cuda', dtype=torch.bfloat16)
        print("BF16 is supported and functional.")
    except Exception as e:
        print(f"BF16 Test Failed: {e}")

    print("-" * 30)
    print("TEST COMPLETE")


if __name__ == "__main__":
    run_test()