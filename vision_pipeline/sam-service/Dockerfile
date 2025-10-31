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
RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    torch==2.1.0 \
    torchvision==0.16.0 \
    opencv-python==4.8.1.78 \
    numpy==1.24.3 \
    pillow==10.1.0 \
    wget

# Install segment-anything separately from git
RUN pip install --no-cache-dir git+https://github.com/facebookresearch/segment-anything.git

# Create directories
RUN mkdir -p /models /shared

# Copy application code
COPY app.py /app/

# Expose port
EXPOSE 8001

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
