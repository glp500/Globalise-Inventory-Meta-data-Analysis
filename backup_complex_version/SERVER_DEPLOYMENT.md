# VOC Page Classifier - Server Deployment Guide

## Overview
This guide explains how to deploy the VOC Page Classifier as a web service that can run on a GPU-enabled server while being accessed from your local machine.

## 🚀 Quick Start

### Option 1: Direct Python Server (Recommended for Testing)
```bash
# 1. Install server dependencies
pip install -r requirements_server.txt

# 2. Configure cache for external drive
python setup_cache.py

# 3. Start the server
python server.py
```

The API will be available at `http://localhost:8000`

### Option 2: Docker Deployment (Recommended for Production)
```bash
# 1. Build and start with Docker Compose
docker-compose up --build

# 2. Access API at http://localhost:8000
```

## 📋 Server Requirements

### Minimum Requirements
- **CPU**: 4+ cores
- **RAM**: 8GB+ (16GB recommended)
- **Storage**: 50GB+ free space for model cache
- **GPU**: Optional but highly recommended (NVIDIA with 8GB+ VRAM)

### Recommended Server Specs
- **CPU**: 8+ cores (Intel Xeon or AMD EPYC)
- **RAM**: 32GB+ 
- **GPU**: NVIDIA RTX 4090, A100, or similar (24GB+ VRAM)
- **Storage**: NVMe SSD with 100GB+ free

## 🔧 Configuration

### 1. Model Caching to External Drive
The setup automatically configures HuggingFace to cache models on your external drive:

```bash
# Run this to configure cache location
python setup_cache.py
```

This creates:
- `models_cache/` directory on your external drive
- `.env` file with cache configuration
- Environment variables for HuggingFace

### 2. GPU Configuration
Edit `src/config/settings.yaml`:
```yaml
model:
  device: "cuda"  # Change from "cpu" to "cuda"
  torch_dtype: "float16"  # Use half-precision for memory efficiency

performance:
  use_gpu: true
  mixed_precision: true
```

### 3. Server Configuration
Environment variables:
```bash
export HOST=0.0.0.0       # Listen on all interfaces
export PORT=8000          # Server port
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
```

## 📡 API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Single Image Classification
```bash
curl -X POST "http://localhost:8000/classify" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/image.jpg"
```

### Batch Classification
```bash
curl -X POST "http://localhost:8000/classify/batch" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg"
```

### Model Information
```bash
curl http://localhost:8000/model/info
```

### System Diagnostics  
```bash
curl http://localhost:8000/diagnostics
```

## 🐳 Docker Deployment

### Basic Deployment
```bash
# Start the service
docker-compose up -d

# View logs
docker-compose logs -f voc-classifier

# Stop the service
docker-compose down
```

### GPU-Enabled Deployment
Make sure you have:
1. NVIDIA Docker runtime installed
2. GPU drivers on the host

```bash
# Start with GPU support
docker-compose up -d

# Verify GPU access
docker-compose exec voc-classifier nvidia-smi
```

## 🌐 Remote Access Configuration

### 1. SSH Tunneling (Secure)
On your local machine:
```bash
# Forward server port to local port
ssh -L 8000:localhost:8000 user@your-server-ip
```

Then access `http://localhost:8000` locally.

### 2. Direct Access (Less Secure)
Configure firewall on server:
```bash
# Open port 8000
sudo ufw allow 8000/tcp
```

Access directly at `http://your-server-ip:8000`

### 3. Nginx Reverse Proxy (Production)
```bash
# Start with Nginx
docker-compose --profile production up -d
```

## 📊 Monitoring and Performance

### Server Metrics
The API provides performance metrics:
```bash
curl http://localhost:8000/diagnostics
```

### Log Files
- Server logs: `logs/server.log`
- Docker logs: `docker-compose logs`

### Performance Tuning

#### For High Throughput
Edit `src/config/settings.yaml`:
```yaml
classification:
  batch_size: 32        # Increase for more GPU memory
  parallel_workers: 4   # CPU parallelization
  
performance:
  mixed_precision: true
  memory_limit_gb: 16
```

#### For Low Memory Systems
```yaml
model:
  torch_dtype: "float16"
  
classification:
  batch_size: 8
  
performance:
  memory_limit_gb: 8
```

## 🐛 Troubleshooting

### Common Issues

#### 1. "No space left on device"
```bash
# Check disk space
df -h

# Clear model cache if needed
rm -rf models_cache/*

# Use external drive cache
python setup_cache.py
```

#### 2. "CUDA out of memory"
```bash
# Reduce batch size in settings.yaml
classification:
  batch_size: 4  # Reduce from default
```

#### 3. "Model not found"
```bash
# Pre-download model manually
python -c "
from transformers import Qwen2VLForConditionalGeneration
model = Qwen2VLForConditionalGeneration.from_pretrained('Qwen/Qwen2-VL-7B-Instruct')
"
```

#### 4. NumPy compatibility errors
```bash
# Install compatible NumPy version
pip install 'numpy<2.0' --force-reinstall
```

### Performance Issues
1. **Slow startup**: Model downloading - wait for completion
2. **High memory usage**: Reduce batch_size or use CPU
3. **Low throughput**: Enable GPU, increase batch_size

## 📱 Client Examples

### Python Client
```python
import requests

# Single image classification
with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/classify',
        files={'file': f}
    )
    result = response.json()
    print(f"Category: {result['category']}")
    print(f"Confidence: {result['confidence']:.2f}")
```

### JavaScript Client
```javascript
// Upload file classification
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8000/classify', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log('Category:', data.category);
    console.log('Confidence:', data.confidence);
});
```

### cURL Batch Processing
```bash
#!/bin/bash
# Process all images in directory
for image in *.jpg; do
    echo "Processing $image..."
    curl -X POST "http://localhost:8000/classify" \
      -H "Content-Type: multipart/form-data" \
      -F "file=@$image" \
      -o "result_$image.json"
done
```

## 🔒 Security Considerations

1. **Authentication**: Add API keys for production
2. **Rate Limiting**: Implement request limits
3. **HTTPS**: Use TLS certificates for production
4. **Input Validation**: Server validates file types
5. **File Cleanup**: Temporary files are auto-deleted

## 📈 Scaling Options

### Horizontal Scaling
```yaml
# docker-compose.yml
services:
  voc-classifier:
    deploy:
      replicas: 3  # Run 3 instances
```

### Load Balancer
Use Nginx or cloud load balancer to distribute requests.

### Cloud Deployment
Deploy to:
- AWS EC2 with GPU instances
- Google Cloud Platform with GPU
- Azure with NVIDIA VMs
- RunPod, Vast.ai, or similar GPU cloud services

## 💡 Tips for Optimal Performance

1. **Pre-warm the model**: Send a test image on startup
2. **Use persistent storage**: Mount external drive for model cache
3. **Monitor GPU memory**: Adjust batch size based on available VRAM
4. **Enable async processing**: For high-volume workloads
5. **Use SSD storage**: For faster model loading and image processing

This server deployment allows you to:
- Run the heavy model on a GPU server
- Access it remotely from your local machine
- Process large batches efficiently
- Scale up for production workloads