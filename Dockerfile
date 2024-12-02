# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt')"

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/models /app/data/output

# Expose ports for different services
# Gradio
EXPOSE 7860  
# Streamlit
EXPOSE 8501  
# Prometheus metrics
EXPOSE 8000  

# Set environment variable for config path
ENV CONFIG_PATH=/app/config.yaml

# Default command (can be overridden)
CMD ["python", "src/main.py"]
