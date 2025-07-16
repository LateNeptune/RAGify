#!/usr/bin/env python3
"""
Quick CUDA availability check
"""

import torch
import subprocess
import sys

def check_cuda_installation():
    print("🔍 CUDA Installation Check")
    print("=" * 40)
    
    # Check PyTorch CUDA
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available in PyTorch: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version (PyTorch): {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("❌ CUDA not available in PyTorch")
        
        # Check if NVIDIA driver is installed
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ NVIDIA driver is installed")
                print("❌ But PyTorch was installed without CUDA support")
                print("\n🔧 To fix this, reinstall PyTorch with CUDA:")
                print("pip uninstall torch torchvision")
                print("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
            else:
                print("❌ NVIDIA driver not found")
        except FileNotFoundError:
            print("❌ nvidia-smi not found - NVIDIA driver not installed")
            print("\n🔧 To fix this:")
            print("1. Install NVIDIA GPU drivers")
            print("2. Install CUDA toolkit")
            print("3. Reinstall PyTorch with CUDA support")

def check_easyocr_gpu():
    print("\n🖼️ EasyOCR GPU Check")
    print("=" * 40)
    
    try:
        import easyocr
        
        # Try to create reader with GPU
        try:
            reader = easyocr.Reader(['en'], gpu=True)
            print("✅ EasyOCR can use GPU")
        except Exception as e:
            print(f"❌ EasyOCR GPU error: {e}")
            
            # Try CPU fallback
            try:
                reader = easyocr.Reader(['en'], gpu=False)
                print("✅ EasyOCR works with CPU")
            except Exception as e2:
                print(f"❌ EasyOCR CPU error: {e2}")
                
    except ImportError:
        print("❌ EasyOCR not installed")

if __name__ == "__main__":
    check_cuda_installation()
    check_easyocr_gpu()
