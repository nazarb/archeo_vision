FROM python:3.10-slim

# Set timezone
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Warsaw

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
# Note: Installing torch with CUDA support for GPU acceleration
RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    torch==2.1.0 \
    torchvision==0.16.0 \
    opencv-python==4.8.1.78 \
    numpy==1.24.3 \
    pillow==10.1.0 \
    rembg[gpu]==2.0.50

# Pre-download the default U2Net model to speed up first request
RUN python -c "from rembg import new_session; new_session('u2net')"

# Create directories
RUN mkdir -p /models /shared

# Copy application code
COPY app.py /app/

# Expose port
EXPOSE 8002

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]
