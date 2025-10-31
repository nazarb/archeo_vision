FROM python:3.10-slim

# Set timezone
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Warsaw

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    requests==2.31.0 \
    pydantic==2.4.2 \
    Pillow==10.1.0

# Create shared directory
RUN mkdir -p /shared

# Copy application code
COPY app.py /app/

# Expose port
EXPOSE 8080

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
