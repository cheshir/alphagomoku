#!/usr/bin/env python3
"""
Check available devices and PyTorch configuration.
Useful for debugging training issues on different platforms.
"""

import sys
import torch


def check_cuda():
    """Check CUDA availability and details"""
    print("=" * 60)
    print("CUDA (NVIDIA GPU) Check")
    print("=" * 60)

    if torch.cuda.is_available():
        print("✅ CUDA is available!")
        print(f"   Device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"   Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"      Total memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"      Compute capability: {props.major}.{props.minor}")
        print(f"   Current device: {torch.cuda.current_device()}")
    else:
        print("❌ CUDA is not available")
        print("   Possible reasons:")
        print("   - PyTorch installed without CUDA support")
        print("   - NVIDIA drivers not installed")
        print("   - No NVIDIA GPU in system")
        print("   - (Colab) GPU runtime not selected")
    print()


def check_mps():
    """Check MPS (Apple Silicon) availability"""
    print("=" * 60)
    print("MPS (Apple Silicon) Check")
    print("=" * 60)

    if torch.backends.mps.is_available():
        print("✅ MPS is available!")
        print("   Running on Apple Silicon (M1/M2/M3)")
    else:
        print("❌ MPS is not available")
        print("   This is normal if not running on Apple Silicon")
    print()


def check_pytorch():
    """Check PyTorch configuration"""
    print("=" * 60)
    print("PyTorch Configuration")
    print("=" * 60)

    print(f"PyTorch version: {torch.__version__}")
    print(f"Python version: {sys.version}")

    # Check if CUDA was built into PyTorch
    cuda_available = hasattr(torch.version, 'cuda') and torch.version.cuda is not None
    print(f"CUDA built: {'✅ Yes' if cuda_available else '❌ No'}")
    if cuda_available:
        print(f"   CUDA version: {torch.version.cuda}")

    print()


def test_device():
    """Test creating tensors on different devices"""
    print("=" * 60)
    print("Device Test")
    print("=" * 60)

    # Test CPU
    try:
        x = torch.randn(100, 100).to('cpu')
        print("✅ CPU works")
    except Exception as e:
        print(f"❌ CPU failed: {e}")

    # Test CUDA
    if torch.cuda.is_available():
        try:
            x = torch.randn(100, 100).to('cuda')
            print("✅ CUDA works")
            # Test computation
            y = x @ x.T
            print(f"   CUDA computation works (result shape: {y.shape})")
        except Exception as e:
            print(f"❌ CUDA failed: {e}")

    # Test MPS
    if torch.backends.mps.is_available():
        try:
            x = torch.randn(100, 100).to('mps')
            print("✅ MPS works")
            # Test computation
            y = x @ x.T
            print(f"   MPS computation works (result shape: {y.shape})")
        except Exception as e:
            print(f"❌ MPS failed: {e}")

    print()


def recommend_action():
    """Recommend what to do based on detection results"""
    print("=" * 60)
    print("Recommendation")
    print("=" * 60)

    if torch.cuda.is_available():
        print("✅ Your system is ready for CUDA training!")
        print("   Use: --device cuda")
        print("   Or: --device auto (will auto-select CUDA)")
    elif torch.backends.mps.is_available():
        print("✅ Your system is ready for MPS training!")
        print("   Use: --device mps")
        print("   Or: --device auto (will auto-select MPS)")
    else:
        print("⚠️  No GPU detected, will use CPU")
        print("   Use: --device cpu")
        print("   Or: --device auto (will auto-select CPU)")
        print()
        print("If you're on Google Colab:")
        print("   1. Go to Runtime > Change runtime type")
        print("   2. Select 'T4 GPU' or other GPU")
        print("   3. Click Save")
        print("   4. Restart the runtime")
        print("   5. Reinstall requirements if needed:")
        print("      !pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print()
        print("If you have an NVIDIA GPU but CUDA is not detected:")
        print("   Reinstall PyTorch with CUDA support:")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")

    print()


def main():
    print("\n" + "=" * 60)
    print("AlphaGomoku Device Check")
    print("=" * 60)
    print()

    check_pytorch()
    check_cuda()
    check_mps()
    test_device()
    recommend_action()

    print("=" * 60)
    print("Check complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
