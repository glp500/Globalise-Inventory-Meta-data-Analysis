#!/usr/bin/env python3
"""
Quick GPU/CUDA test for the VOC classifier
"""

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

try:
    import torch
    print("✅ PyTorch imported successfully")

    # Check CUDA
    print(f"🔍 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"📊 CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"💾 CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    # Check MPS (Apple Metal)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("🍎 Apple MPS (Metal) available: True")
    else:
        print("🍎 Apple MPS (Metal) available: False")

    # Test device selection
    if torch.cuda.is_available():
        device = "cuda"
        print(f"🚀 Selected device: {device}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        print(f"🚀 Selected device: {device}")
    else:
        device = "cpu"
        print(f"🚀 Selected device: {device}")

    # Quick tensor test
    print("\n🧪 Testing tensor operations...")
    x = torch.randn(100, 100).to(device)
    y = torch.randn(100, 100).to(device)
    z = torch.matmul(x, y)
    print(f"✅ Tensor test successful on {device}")
    print(f"📈 Result shape: {z.shape}, device: {z.device}")

    if device == "cuda":
        print(f"💾 GPU memory used: {torch.cuda.memory_allocated() / 1e6:.1f}MB")
        torch.cuda.empty_cache()
        print("🧹 GPU cache cleared")

except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n🏁 GPU test complete")