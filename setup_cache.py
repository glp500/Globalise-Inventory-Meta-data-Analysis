#!/usr/bin/env python3
"""
Setup script to configure HuggingFace cache to external drive.
"""

import os
from pathlib import Path

def setup_external_cache():
    """Set up HuggingFace cache on the external drive."""
    
    # Get the current project directory (which is on the external drive)
    project_dir = Path(__file__).parent
    
    # Create cache directory on the external drive
    cache_dir = project_dir / "models_cache"
    cache_dir.mkdir(exist_ok=True)
    
    # Set HuggingFace cache environment variables
    os.environ['TRANSFORMERS_CACHE'] = str(cache_dir)
    os.environ['HF_HOME'] = str(cache_dir)
    os.environ['HF_DATASETS_CACHE'] = str(cache_dir / "datasets")
    
    print(f"✅ HuggingFace cache configured to: {cache_dir}")
    print(f"✅ This will store models on your external drive instead of MacBook storage")
    
    # Create a .env file for persistent storage
    env_file = project_dir / ".env"
    with open(env_file, "w") as f:
        f.write(f"TRANSFORMERS_CACHE={cache_dir}\n")
        f.write(f"HF_HOME={cache_dir}\n")
        f.write(f"HF_DATASETS_CACHE={cache_dir / 'datasets'}\n")
    
    print(f"✅ Environment variables saved to: {env_file}")
    
    return str(cache_dir)

if __name__ == "__main__":
    cache_path = setup_external_cache()
    print(f"\n🚀 To use this cache in your scripts, run:")
    print(f"   export TRANSFORMERS_CACHE={cache_path}")
    print(f"   export HF_HOME={cache_path}")
    print(f"\nOr source the .env file before running Python scripts.")