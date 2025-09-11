# VOC Page Classifier Server Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONPATH=/app/src
ENV TRANSFORMERS_CACHE=/app/models_cache
ENV HF_HOME=/app/models_cache
ENV PYTHONUNBUFFERED=1

# Create cache directories
RUN mkdir -p /app/models_cache /app/logs /app/data/input /app/data/output

# Copy requirements first (for better caching)
COPY requirements_server.txt requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_server.txt

# Copy source code
COPY src/ ./src/
COPY server.py ./
COPY setup_cache.py ./

# Create non-root user
RUN useradd -m -u 1000 classifier && \
    chown -R classifier:classifier /app

USER classifier

# Configure cache
RUN python setup_cache.py

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run server
CMD ["python", "server.py"]