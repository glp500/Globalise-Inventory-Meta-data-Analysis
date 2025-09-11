#!/usr/bin/env python3
"""
FastAPI server for VOC Page Classifier.
Allows remote access to the classification model via REST API.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import sys
import tempfile
import logging
from pathlib import Path
from typing import List, Optional
import json
import asyncio
import aiofiles

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.classifier import VOCPageClassifier, ClassificationResult
from utils.diagnostics import run_full_diagnostics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/server.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="VOC Page Classifier API",
    description="REST API for classifying VOC (Dutch East India Company) historical documents",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global classifier instance
classifier = None
server_config = {}

@app.on_event("startup")
async def startup_event():
    """Initialize the classifier on startup."""
    global classifier, server_config
    
    logger.info("Starting VOC Page Classifier Server...")
    
    # Load configuration
    config_path = "src/config/settings.yaml"
    try:
        classifier = VOCPageClassifier(config_path)
        logger.info("✅ Classifier initialized successfully")
        
        # Load server configuration
        import yaml
        with open(config_path) as f:
            server_config = yaml.safe_load(f)
            
    except Exception as e:
        logger.error(f"❌ Failed to initialize classifier: {e}")
        # Don't exit - allow server to start with limited functionality
        classifier = None

@app.get("/")
async def root():
    """Root endpoint with server information."""
    return {
        "service": "VOC Page Classifier API",
        "version": "1.0.0",
        "status": "online" if classifier else "model_unavailable",
        "endpoints": {
            "classify_single": "/classify",
            "classify_batch": "/classify/batch", 
            "health": "/health",
            "diagnostics": "/diagnostics",
            "model_info": "/model/info"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if classifier:
        model_available = classifier.qwen_interface and classifier.qwen_interface.is_available()
        return {
            "status": "healthy",
            "classifier_initialized": True,
            "model_available": model_available,
            "fallback_classification": "available"
        }
    else:
        return {
            "status": "degraded",
            "classifier_initialized": False,
            "model_available": False,
            "message": "Running in limited mode - check logs"
        }

@app.post("/classify")
async def classify_single_image(file: UploadFile = File(...)):
    """
    Classify a single image.
    
    Args:
        file: Image file to classify
        
    Returns:
        Classification result with category, confidence, and features
    """
    if not classifier:
        raise HTTPException(status_code=503, detail="Classifier not available")
    
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Classify the image
        result = classifier.classify_page(temp_path)
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        # Convert ClassificationResult to dict for JSON response
        response = {
            "image_path": file.filename,
            "category": result.category,
            "confidence": result.confidence,
            "page_type": result.page_type,
            "features": result.features,
            "timestamp": result.timestamp,
            "error": result.error,
            "processing_method": result.features.get('classification_method', 'unknown')
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error classifying image {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.post("/classify/batch")
async def classify_batch_images(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Classify multiple images in batch.
    
    Args:
        files: List of image files to classify
        
    Returns:
        List of classification results
    """
    if not classifier:
        raise HTTPException(status_code=503, detail="Classifier not available")
    
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Batch size limited to 50 images")
    
    try:
        temp_files = []
        results = []
        
        # Save all files temporarily
        for file in files:
            if not file.content_type or not file.content_type.startswith('image/'):
                continue
                
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_files.append((temp_file.name, file.filename))
        
        # Process batch
        for temp_path, original_name in temp_files:
            result = classifier.classify_page(temp_path)
            
            response = {
                "image_path": original_name,
                "category": result.category,
                "confidence": result.confidence,
                "page_type": result.page_type,
                "features": result.features,
                "timestamp": result.timestamp,
                "error": result.error,
                "processing_method": result.features.get('classification_method', 'unknown')
            }
            results.append(response)
        
        # Clean up temporary files
        for temp_path, _ in temp_files:
            try:
                os.unlink(temp_path)
            except:
                pass
        
        return {
            "batch_size": len(results),
            "results": results,
            "summary": {
                "successful": len([r for r in results if r["category"] != "error"]),
                "failed": len([r for r in results if r["category"] == "error"]),
                "average_confidence": sum(r["confidence"] for r in results) / len(results) if results else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Error in batch classification: {e}")
        raise HTTPException(status_code=500, detail=f"Batch classification failed: {str(e)}")

@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model."""
    if not classifier:
        return {"status": "no_classifier", "model_available": False}
    
    if classifier.qwen_interface:
        model_info = classifier.qwen_interface.get_model_info()
        return {
            "classifier_available": True,
            "model_info": model_info,
            "categories": classifier.CATEGORIES,
            "config": {
                "device": server_config.get('model', {}).get('device', 'unknown'),
                "batch_size": server_config.get('classification', {}).get('batch_size', 'unknown')
            }
        }
    else:
        return {
            "classifier_available": True,
            "model_available": False,
            "fallback_mode": True,
            "categories": classifier.CATEGORIES
        }

@app.get("/diagnostics")
async def run_diagnostics():
    """Run system diagnostics."""
    try:
        from utils.diagnostics import run_full_diagnostics
        import yaml
        
        with open("src/config/settings.yaml") as f:
            config = yaml.safe_load(f)
        
        diagnostics = run_full_diagnostics(config)
        
        return {
            "diagnostics": diagnostics,
            "timestamp": classifier._get_timestamp() if classifier else None
        }
        
    except Exception as e:
        logger.error(f"Diagnostics failed: {e}")
        return {
            "error": f"Diagnostics failed: {str(e)}",
            "basic_info": {
                "classifier_available": classifier is not None,
                "server_status": "running"
            }
        }

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    # Server configuration
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
    
    logger.info(f"Starting server on {HOST}:{PORT}")
    
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info",
        access_log=True,
        reload=False  # Set to True for development
    )