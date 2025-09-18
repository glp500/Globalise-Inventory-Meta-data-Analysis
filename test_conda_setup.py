#!/usr/bin/env python3
"""
Test conda environment setup for VOC classifier
"""

import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all required packages can be imported."""
    print("🧪 Testing package imports...")

    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")

        import torchvision
        print(f"✅ TorchVision: {torchvision.__version__}")

        from PIL import Image
        print(f"✅ Pillow: {Image.__version__}")

        import transformers
        print(f"✅ Transformers: {transformers.__version__}")

        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_conda_info():
    """Test conda environment information."""
    print("\n🔍 Conda environment info...")

    try:
        # Get conda info
        result = subprocess.run(['conda', 'info', '--json'],
                              capture_output=True, text=True, check=True)
        import json
        info = json.loads(result.stdout)

        print(f"✅ Conda version: {info['conda_version']}")
        print(f"✅ Active environment: {info['active_prefix_name']}")
        print(f"✅ Python version: {info['python_version']}")

        return True
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ Conda info failed: {e}")
        return False

def test_gpu_support():
    """Test GPU support in conda environment."""
    print("\n🚀 Testing GPU support...")

    try:
        import torch

        print(f"🔍 CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"📊 CUDA devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")

        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("🍎 Apple MPS available: True")
        else:
            print("🍎 Apple MPS available: False")

        return True
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False

def test_classifier_import():
    """Test that classifier can be imported."""
    print("\n📋 Testing classifier import...")

    try:
        # Test basic import without instantiation
        import sys
        from pathlib import Path

        # Add current directory to path
        current_dir = Path(__file__).parent
        sys.path.insert(0, str(current_dir))

        # Import classifier module
        import classify
        print("✅ Classifier module imported successfully")

        # Test model shortcuts
        shortcuts = classify.POPULAR_MODELS
        print(f"✅ Model shortcuts available: {len(shortcuts)}")

        return True
    except Exception as e:
        print(f"❌ Classifier import failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Conda Environment Setup for VOC Classifier\n")

    tests = [
        ("Package Imports", test_imports),
        ("Conda Environment", test_conda_info),
        ("GPU Support", test_gpu_support),
        ("Classifier Import", test_classifier_import)
    ]

    results = []

    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)

    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
        if success:
            passed += 1

    print(f"\n🎯 Tests passed: {passed}/{len(tests)}")

    if passed == len(tests):
        print("🎉 All tests passed! Conda environment is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())