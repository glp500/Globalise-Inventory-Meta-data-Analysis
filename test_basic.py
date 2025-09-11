#!/usr/bin/env python3
"""
Quick test script for basic functionality.
"""

import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test if core modules can be imported."""
    print("Testing imports...")
    try:
        from utils.image_utils import load_image, get_image_info
        print("✅ Image utils imported successfully")
        
        from core.page_detector import detect_two_page_spread, detect_foldout
        print("✅ Page detector imported successfully")
        
        from core.classifier import VOCPageClassifier
        print("✅ Classifier imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_image_loading():
    """Test basic image loading."""
    print("\nTesting image loading...")
    
    # Import here after path setup
    from utils.image_utils import load_image, get_image_info
    
    # Find test images
    image_files = list(Path('data/input').glob('*.jpg'))
    if not image_files:
        print("❌ No test images found in data/input")
        return False
    
    test_image = str(image_files[0])
    print(f"Testing with: {Path(test_image).name}")
    
    try:
        # Test image loading
        image = load_image(test_image)
        if image is not None:
            print("✅ Image loading successful")
            print(f"   Shape: {image.shape}")
            
            # Test image info
            info = get_image_info(test_image)
            if info:
                print("✅ Image info extraction successful")
                print(f"   Dimensions: {info.get('width', 'N/A')}x{info.get('height', 'N/A')}")
                print(f"   Aspect ratio: {info.get('aspect_ratio', 'N/A'):.2f}")
                print(f"   File size: {info.get('file_size_bytes', 0) / (1024*1024):.1f} MB")
            return True
        else:
            print("❌ Image loading returned None")
            return False
            
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return False

def test_page_detection():
    """Test page detection functionality."""
    print("\nTesting page detection...")
    
    from core.page_detector import detect_two_page_spread, detect_foldout, analyze_page_structure
    
    # Find test images
    image_files = list(Path('data/input').glob('*.jpg'))
    if not image_files:
        print("❌ No test images found")
        return False
    
    test_image = str(image_files[0])
    
    try:
        # Test two-page detection
        spread_result = detect_two_page_spread(test_image)
        print("✅ Two-page detection successful")
        print(f"   Is two-page: {spread_result.get('is_two_page', False)}")
        print(f"   Confidence: {spread_result.get('confidence', 0):.2f}")
        
        # Test foldout detection
        foldout_result = detect_foldout(test_image)
        print("✅ Foldout detection successful")
        print(f"   Is foldout: {foldout_result.get('is_foldout', False)}")
        print(f"   Aspect ratio: {foldout_result.get('aspect_ratio', 0):.2f}")
        
        # Test complete analysis
        analysis = analyze_page_structure(test_image)
        print("✅ Page structure analysis successful")
        print(f"   Page type: {analysis.get('page_type', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in page detection: {e}")
        return False

def test_classifier_init():
    """Test classifier initialization (without model loading)."""
    print("\nTesting classifier initialization...")
    
    try:
        from core.classifier import VOCPageClassifier
        
        # Test initialization (this might fail at model loading)
        classifier = VOCPageClassifier('src/config/settings.yaml')
        print("✅ Classifier initialized")
        print(f"   Categories available: {len(classifier.CATEGORIES)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Classifier initialization error: {e}")
        print("   This is expected if PyTorch/model loading has issues")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("VOC CLASSIFIER - BASIC FUNCTIONALITY TEST")
    print("=" * 60)
    
    results = {
        'imports': test_imports(),
        'image_loading': test_image_loading(),
        'page_detection': test_page_detection(),
        'classifier_init': test_classifier_init()
    }
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title():<20} {status}")
    
    passed = sum(results.values())
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic functionality tests passed!")
        return True
    else:
        print("⚠️  Some tests failed - see details above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)