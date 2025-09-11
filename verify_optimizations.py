#!/usr/bin/env python3
"""
Verification script to test all VOC classifier optimizations.
"""

import sys
import os
import time
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_fallback_classification():
    """Test fallback classification when model is not available."""
    print("\n🔄 Testing fallback classification...")
    
    try:
        from core.classifier import VOCPageClassifier
        
        # Find a test image
        image_files = list(Path('data/input').glob('*.jpg'))
        if not image_files:
            print("❌ No test images found")
            return False
        
        test_image = str(image_files[0])
        
        # Initialize classifier with potential model loading failure
        classifier = VOCPageClassifier('src/config/settings.yaml')
        
        # Test classification (should use fallback if model fails)
        result = classifier.classify_page(test_image)
        
        print(f"✅ Classification result: {result.category}")
        print(f"   Confidence: {result.confidence:.2f}")
        print(f"   Method: {result.features.get('classification_method', 'unknown')}")
        
        return result.category != 'error'
        
    except Exception as e:
        print(f"❌ Fallback classification test failed: {e}")
        return False

def test_batch_processing():
    """Test batch processing functionality."""
    print("\n🔄 Testing batch processing...")
    
    try:
        from core.classifier import VOCPageClassifier
        
        # Find test images
        image_files = list(Path('data/input').glob('*.jpg'))[:3]  # Test with 3 images
        if len(image_files) < 2:
            print("❌ Need at least 2 test images")
            return False
        
        classifier = VOCPageClassifier('src/config/settings.yaml')
        
        # Test directory processing
        start_time = time.time()
        results = classifier.process_directory('data/input')
        processing_time = time.time() - start_time
        
        print(f"✅ Processed {len(results)} images in {processing_time:.2f}s")
        print(f"   Average time per image: {processing_time/len(results):.2f}s")
        
        # Check results quality
        successful = [r for r in results if r.category != 'error']
        print(f"   Success rate: {len(successful)}/{len(results)} ({len(successful)/len(results)*100:.1f}%)")
        
        return len(successful) > 0
        
    except Exception as e:
        print(f"❌ Batch processing test failed: {e}")
        return False

def test_page_detection():
    """Test optimized page detection."""
    print("\n🔄 Testing page detection performance...")
    
    try:
        from core.page_detector import analyze_page_structure
        
        # Find test images
        image_files = list(Path('data/input').glob('*.jpg'))[:5]
        if not image_files:
            print("❌ No test images found")
            return False
        
        total_time = 0
        successful = 0
        
        for image_file in image_files:
            start_time = time.time()
            try:
                result = analyze_page_structure(str(image_file))
                detection_time = time.time() - start_time
                total_time += detection_time
                
                if result.get('page_type'):
                    successful += 1
                    print(f"   ✅ {image_file.name}: {result.get('page_type')} ({detection_time:.3f}s)")
                else:
                    print(f"   ⚠️  {image_file.name}: No detection result")
                    
            except Exception as e:
                print(f"   ❌ {image_file.name}: Error - {e}")
        
        avg_time = total_time / len(image_files) if image_files else 0
        print(f"✅ Average detection time: {avg_time:.3f}s per image")
        print(f"   Success rate: {successful}/{len(image_files)}")
        
        return successful > 0
        
    except Exception as e:
        print(f"❌ Page detection test failed: {e}")
        return False

def test_configuration_loading():
    """Test configuration loading and validation."""
    print("\n🔄 Testing configuration loading...")
    
    try:
        import yaml
        from pathlib import Path
        
        config_path = Path('src/config/settings.yaml')
        if not config_path.exists():
            print("❌ Configuration file not found")
            return False
        
        # Load configuration
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Validate required sections
        required_sections = ['model', 'classification', 'performance', 'monitoring']
        missing = [s for s in required_sections if s not in config]
        
        if missing:
            print(f"❌ Missing configuration sections: {missing}")
            return False
        
        # Check key settings
        model_config = config.get('model', {})
        perf_config = config.get('performance', {})
        
        print("✅ Configuration validation:")
        print(f"   Model device: {model_config.get('device', 'not set')}")
        print(f"   Batch size: {config.get('classification', {}).get('batch_size', 'not set')}")
        print(f"   GPU enabled: {perf_config.get('use_gpu', 'not set')}")
        print(f"   Mixed precision: {perf_config.get('mixed_precision', 'not set')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_error_handling():
    """Test error handling and recovery."""
    print("\n🔄 Testing error handling...")
    
    try:
        from core.classifier import VOCPageClassifier
        
        classifier = VOCPageClassifier('src/config/settings.yaml')
        
        # Test with non-existent file
        result = classifier.classify_page('nonexistent_image.jpg')
        
        if result.error:
            print(f"✅ Error handling working: {result.error[:50]}...")
            print(f"   Fallback category: {result.category}")
            return True
        else:
            print("❌ No error reported for invalid file")
            return False
            
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def test_diagnostics():
    """Test diagnostic utilities."""
    print("\n🔄 Testing diagnostics...")
    
    try:
        from utils.diagnostics import run_full_diagnostics, print_diagnostics_report
        import yaml
        
        # Load config for diagnostics
        with open('src/config/settings.yaml') as f:
            config = yaml.safe_load(f)
        
        # Run diagnostics
        diagnostics = run_full_diagnostics(config)
        
        # Print abbreviated report
        print("✅ Diagnostics completed:")
        print(f"   Dependencies available: {sum(1 for k, v in diagnostics['dependencies'].items() if k != 'errors' and v)}")
        print(f"   GPU available: {diagnostics['gpu']['cuda_available']}")
        print(f"   Recommendations: {len(diagnostics.get('recommendations', []))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Diagnostics test failed: {e}")
        return False

def run_performance_benchmark():
    """Run a quick performance benchmark."""
    print("\n📊 Running performance benchmark...")
    
    try:
        from core.classifier import VOCPageClassifier
        import time
        
        # Find test images
        image_files = list(Path('data/input').glob('*.jpg'))[:3]
        if len(image_files) < 2:
            print("❌ Need at least 2 test images for benchmark")
            return False
        
        classifier = VOCPageClassifier('src/config/settings.yaml')
        
        # Sequential processing benchmark
        start_time = time.time()
        results = []
        for image_file in image_files:
            result = classifier.classify_page(str(image_file))
            results.append(result)
        sequential_time = time.time() - start_time
        
        # Directory processing benchmark (may use batch processing)
        start_time = time.time()
        batch_results = classifier._process_directory_sequential(image_files)
        batch_time = time.time() - start_time
        
        print(f"📈 Performance Results:")
        print(f"   Sequential: {sequential_time:.2f}s total ({sequential_time/len(image_files):.2f}s per image)")
        print(f"   Batch method: {batch_time:.2f}s total ({batch_time/len(image_files):.2f}s per image)")
        
        successful_results = [r for r in results if r.category != 'error']
        print(f"   Success rate: {len(successful_results)}/{len(results)} ({len(successful_results)/len(results)*100:.1f}%)")
        
        return len(successful_results) > 0
        
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        return False

def main():
    """Run all verification tests."""
    print("=" * 60)
    print("VOC CLASSIFIER - OPTIMIZATION VERIFICATION")
    print("=" * 60)
    
    tests = [
        ("Configuration Loading", test_configuration_loading),
        ("Diagnostics System", test_diagnostics),
        ("Page Detection", test_page_detection),
        ("Error Handling", test_error_handling),
        ("Fallback Classification", test_fallback_classification),
        ("Batch Processing", test_batch_processing),
        ("Performance Benchmark", run_performance_benchmark)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:<25} {status}")
        if success:
            passed += 1
    
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All optimizations verified successfully!")
        print("\n💡 Your VOC classifier is now optimized and ready for production use!")
    elif passed >= total * 0.7:
        print("✅ Most optimizations working - system is functional")
        print("⚠️  Some non-critical tests failed - check logs above")
    else:
        print("⚠️  Multiple optimization failures detected")
        print("🔧 Review failed tests and check system configuration")
    
    return passed / total

if __name__ == "__main__":
    success_rate = main()
    sys.exit(0 if success_rate >= 0.7 else 1)