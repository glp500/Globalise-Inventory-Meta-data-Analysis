"""
System diagnostics and GPU detection utilities for VOC classifier.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def check_gpu_availability() -> Dict[str, Any]:
    """
    Check GPU availability and compatibility.
    
    Returns:
        Dictionary with GPU diagnostics information
    """
    diagnostics = {
        'torch_available': False,
        'cuda_available': False,
        'cuda_devices': 0,
        'current_device': 'CPU only',
        'gpu_memory_gb': 0,
        'errors': []
    }
    
    try:
        import torch
        diagnostics['torch_available'] = True
        
        # Check CUDA availability
        diagnostics['cuda_available'] = torch.cuda.is_available()
        diagnostics['cuda_devices'] = torch.cuda.device_count()
        
        if diagnostics['cuda_available']:
            diagnostics['current_device'] = f"cuda:{torch.cuda.current_device()}"
            # Get GPU memory info
            gpu_props = torch.cuda.get_device_properties(0)
            diagnostics['gpu_memory_gb'] = gpu_props.total_memory / (1024**3)
            diagnostics['gpu_name'] = gpu_props.name
        else:
            diagnostics['errors'].append("CUDA not available - using CPU only")
            
    except ImportError as e:
        diagnostics['errors'].append(f"PyTorch not available: {e}")
    except Exception as e:
        diagnostics['errors'].append(f"Error checking GPU: {e}")
    
    return diagnostics


def check_dependencies() -> Dict[str, Any]:
    """
    Check if all required dependencies are available.
    
    Returns:
        Dictionary with dependency status
    """
    dependencies = {
        'torch': False,
        'transformers': False,
        'opencv': False,
        'pillow': False,
        'numpy': False,
        'errors': []
    }
    
    # Check PyTorch
    try:
        import torch
        dependencies['torch'] = True
    except ImportError as e:
        dependencies['errors'].append(f"PyTorch not available: {e}")
    
    # Check Transformers
    try:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        dependencies['transformers'] = True
    except ImportError as e:
        dependencies['errors'].append(f"Transformers not available: {e}")
    
    # Check OpenCV
    try:
        import cv2
        dependencies['opencv'] = True
    except ImportError as e:
        dependencies['errors'].append(f"OpenCV not available: {e}")
    
    # Check PIL
    try:
        from PIL import Image
        dependencies['pillow'] = True
    except ImportError as e:
        dependencies['errors'].append(f"Pillow not available: {e}")
    
    # Check NumPy
    try:
        import numpy as np
        dependencies['numpy'] = True
        # Check NumPy version compatibility
        numpy_version = np.__version__
        if numpy_version.startswith('2.'):
            dependencies['errors'].append(f"NumPy 2.x detected ({numpy_version}) - may cause PyTorch compatibility issues. Consider downgrading to numpy<2")
    except ImportError as e:
        dependencies['errors'].append(f"NumPy not available: {e}")
    
    return dependencies


def check_model_path(model_name: str) -> Dict[str, Any]:
    """
    Check if model can be loaded from cache or needs downloading.
    
    Args:
        model_name: Name/path of the model
        
    Returns:
        Dictionary with model availability status
    """
    model_info = {
        'model_name': model_name,
        'cached': False,
        'cache_path': None,
        'size_gb': 0,
        'errors': []
    }
    
    try:
        from transformers import Qwen2VLForConditionalGeneration
        
        # Check if model is cached
        try:
            # This will check cache without downloading
            cache_dir = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name, 
                local_files_only=True,
                cache_dir=None
            ).config._name_or_path
            model_info['cached'] = True
            model_info['cache_path'] = cache_dir
        except Exception:
            model_info['errors'].append(f"Model {model_name} not cached - will need to download")
            
    except ImportError as e:
        model_info['errors'].append(f"Cannot check model: Transformers not available - {e}")
    except Exception as e:
        model_info['errors'].append(f"Error checking model: {e}")
    
    return model_info


def run_full_diagnostics(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Run comprehensive system diagnostics.
    
    Args:
        config: Configuration dictionary (optional)
        
    Returns:
        Complete diagnostics report
    """
    logger.info("Running system diagnostics...")
    
    diagnostics = {
        'system': {
            'platform': sys.platform,
            'python_version': sys.version,
            'python_executable': sys.executable
        },
        'gpu': check_gpu_availability(),
        'dependencies': check_dependencies(),
        'recommendations': []
    }
    
    # Check model if config provided
    if config and 'model' in config:
        model_name = config['model'].get('name', 'Qwen/Qwen2-VL-7B-Instruct')
        diagnostics['model'] = check_model_path(model_name)
    
    # Generate recommendations
    recommendations = []
    
    # GPU recommendations
    if not diagnostics['gpu']['cuda_available']:
        recommendations.append("⚠️  GPU not available - processing will be slow on CPU only")
        recommendations.append("💡 Consider using a machine with CUDA-compatible GPU for better performance")
    
    # Dependency recommendations  
    if diagnostics['dependencies']['errors']:
        recommendations.append("❌ Missing dependencies detected - install missing packages")
    
    # NumPy compatibility
    if any('NumPy 2.x' in error for error in diagnostics['dependencies']['errors']):
        recommendations.append("🔧 NumPy 2.x compatibility issue - run: pip install 'numpy<2'")
    
    # Model recommendations
    if 'model' in diagnostics and diagnostics['model']['errors']:
        recommendations.append("📥 Model not cached - first run will download ~14GB")
    
    diagnostics['recommendations'] = recommendations
    
    return diagnostics


def print_diagnostics_report(diagnostics: Dict[str, Any]):
    """Print formatted diagnostics report."""
    print("\n" + "="*60)
    print("VOC CLASSIFIER - SYSTEM DIAGNOSTICS REPORT")
    print("="*60)
    
    # System info
    print(f"\n🖥️  System: {diagnostics['system']['platform']}")
    print(f"🐍 Python: {diagnostics['system']['python_version'].split()[0]}")
    
    # GPU status
    gpu = diagnostics['gpu']
    if gpu['cuda_available']:
        print(f"🚀 GPU: {gpu.get('gpu_name', 'Available')} ({gpu['gpu_memory_gb']:.1f}GB)")
    else:
        print("⚠️  GPU: Not available (CPU only)")
    
    # Dependencies
    deps = diagnostics['dependencies']
    working_deps = sum(1 for k, v in deps.items() if k != 'errors' and v)
    total_deps = len([k for k in deps.keys() if k != 'errors'])
    print(f"📦 Dependencies: {working_deps}/{total_deps} available")
    
    # Errors
    all_errors = gpu.get('errors', []) + deps.get('errors', [])
    if 'model' in diagnostics:
        all_errors.extend(diagnostics['model'].get('errors', []))
    
    if all_errors:
        print(f"\n❌ Issues Found ({len(all_errors)}):")
        for error in all_errors[:5]:  # Show first 5 errors
            print(f"   • {error}")
        if len(all_errors) > 5:
            print(f"   ... and {len(all_errors) - 5} more")
    
    # Recommendations
    if diagnostics['recommendations']:
        print(f"\n💡 Recommendations:")
        for rec in diagnostics['recommendations']:
            print(f"   {rec}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    # Run diagnostics as standalone script
    import yaml
    
    # Try to load config
    config = None
    config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    
    # Run diagnostics
    diagnostics = run_full_diagnostics(config)
    print_diagnostics_report(diagnostics)