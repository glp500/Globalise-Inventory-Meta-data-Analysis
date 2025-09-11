#!/usr/bin/env python3
"""
Test client for VOC Page Classifier server.
"""

import requests
import json
import time
from pathlib import Path
import argparse

class VOCClassifierClient:
    """Client for VOC Page Classifier API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
    
    def health_check(self):
        """Check if server is healthy."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            return response.status_code == 200, response.json()
        except Exception as e:
            return False, {"error": str(e)}
    
    def get_model_info(self):
        """Get model information."""
        try:
            response = requests.get(f"{self.base_url}/model/info", timeout=10)
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def classify_single(self, image_path: str):
        """Classify a single image."""
        try:
            with open(image_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(
                    f"{self.base_url}/classify",
                    files=files,
                    timeout=60
                )
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def classify_batch(self, image_paths: list):
        """Classify multiple images."""
        try:
            files = []
            for path in image_paths:
                files.append(('files', open(path, 'rb')))
            
            response = requests.post(
                f"{self.base_url}/classify/batch",
                files=files,
                timeout=300
            )
            
            # Close all file handles
            for _, f in files:
                f.close()
                
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def run_diagnostics(self):
        """Run server diagnostics."""
        try:
            response = requests.get(f"{self.base_url}/diagnostics", timeout=30)
            return response.json()
        except Exception as e:
            return {"error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="Test VOC Page Classifier API")
    parser.add_argument("--url", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--image", help="Single image to classify")
    parser.add_argument("--batch", help="Directory containing images for batch processing")
    parser.add_argument("--health", action="store_true", help="Check server health")
    parser.add_argument("--info", action="store_true", help="Get model information")
    parser.add_argument("--diagnostics", action="store_true", help="Run diagnostics")
    
    args = parser.parse_args()
    
    client = VOCClassifierClient(args.url)
    
    print(f"🔗 Connecting to VOC Classifier at: {args.url}")
    
    # Health check first
    healthy, health_info = client.health_check()
    if not healthy:
        print(f"❌ Server health check failed: {health_info}")
        return
    
    print(f"✅ Server is healthy: {health_info.get('status', 'unknown')}")
    
    if args.health:
        print("\n📊 Health Status:")
        print(json.dumps(health_info, indent=2))
    
    if args.info:
        print("\n🔍 Model Information:")
        model_info = client.get_model_info()
        print(json.dumps(model_info, indent=2))
    
    if args.diagnostics:
        print("\n🔧 Running Diagnostics...")
        diagnostics = client.run_diagnostics()
        print(json.dumps(diagnostics, indent=2))
    
    if args.image:
        print(f"\n🖼️  Classifying single image: {args.image}")
        if Path(args.image).exists():
            start_time = time.time()
            result = client.classify_single(args.image)
            processing_time = time.time() - start_time
            
            if "error" not in result:
                print(f"✅ Classification completed in {processing_time:.2f}s")
                print(f"   Category: {result.get('category', 'unknown')}")
                print(f"   Confidence: {result.get('confidence', 0):.2f}")
                print(f"   Page Type: {result.get('page_type', 'unknown')}")
                print(f"   Method: {result.get('processing_method', 'unknown')}")
                if result.get('error'):
                    print(f"   Warning: {result['error']}")
            else:
                print(f"❌ Classification failed: {result['error']}")
        else:
            print(f"❌ Image file not found: {args.image}")
    
    if args.batch:
        print(f"\n📦 Processing batch from directory: {args.batch}")
        batch_dir = Path(args.batch)
        if batch_dir.exists() and batch_dir.is_dir():
            # Find image files
            image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
            image_files = [
                str(f) for f in batch_dir.rglob('*')
                if f.suffix.lower() in image_extensions and f.is_file()
            ]
            
            if not image_files:
                print(f"❌ No image files found in {args.batch}")
                return
            
            print(f"Found {len(image_files)} images to process...")
            
            # Limit batch size for API
            if len(image_files) > 20:
                print(f"⚠️  Limiting to first 20 images (API limit)")
                image_files = image_files[:20]
            
            start_time = time.time()
            result = client.classify_batch(image_files)
            processing_time = time.time() - start_time
            
            if "error" not in result:
                print(f"✅ Batch processing completed in {processing_time:.2f}s")
                summary = result.get('summary', {})
                print(f"   Processed: {result.get('batch_size', 0)} images")
                print(f"   Successful: {summary.get('successful', 0)}")
                print(f"   Failed: {summary.get('failed', 0)}")
                print(f"   Avg Confidence: {summary.get('average_confidence', 0):.2f}")
                
                # Show detailed results
                if result.get('results'):
                    print("\n📋 Detailed Results:")
                    for i, res in enumerate(result['results'][:5]):  # Show first 5
                        print(f"   {i+1}. {res.get('image_path', 'unknown')}")
                        print(f"      Category: {res.get('category', 'unknown')}")
                        print(f"      Confidence: {res.get('confidence', 0):.2f}")
                        if res.get('error'):
                            print(f"      Error: {res['error']}")
                    
                    if len(result['results']) > 5:
                        print(f"   ... and {len(result['results']) - 5} more")
            else:
                print(f"❌ Batch processing failed: {result['error']}")
        else:
            print(f"❌ Directory not found: {args.batch}")
    
    if not any([args.health, args.info, args.diagnostics, args.image, args.batch]):
        print("\n💡 Usage examples:")
        print(f"  python test_client.py --health")
        print(f"  python test_client.py --info")
        print(f"  python test_client.py --image data/input/sample.jpg")
        print(f"  python test_client.py --batch data/input/")
        print(f"  python test_client.py --diagnostics")

if __name__ == "__main__":
    main()