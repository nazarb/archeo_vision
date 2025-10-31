#!/bin/bash
# Setup script for Vision AI Pipeline

set -e

echo "=========================================="
echo "Vision AI Pipeline Setup"
echo "=========================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
COMPOSE_CMD=""
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
elif docker compose version &> /dev/null 2>&1; then
    COMPOSE_CMD="sudo docker compose"
else
    echo -e "${RED}Error: Docker Compose is not installed${NC}"
    echo "Please install Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

echo -e "${GREEN}✓ Docker and Docker Compose are installed${NC}"
echo "Using: $COMPOSE_CMD"

# Check for NVIDIA GPU
echo ""
echo "Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ NVIDIA GPU detected${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    GPU_MODE=true
else
    echo -e "${YELLOW}⚠ No NVIDIA GPU detected - will use CPU mode${NC}"
    echo -e "${YELLOW}  Note: CPU mode is significantly slower${NC}"
    GPU_MODE=false
fi

# Create directory structure
echo ""
echo "Creating directory structure..."
mkdir -p sam-service
mkdir -p pipeline-api
mkdir -p shared/{images,detections,results,masks,visualizations}

echo -e "${GREEN}✓ Directories created${NC}"

# Create default prompt file if it doesn't exist
if [ ! -f "shared/prompt.txt" ]; then
    echo ""
    echo "Creating default prompt file..."
    cat > shared/prompt.txt << 'EOF'
Detect all objects in this image. Return the results as a JSON array with each object having:
- "label": the object class name (e.g., "dog", "cat", "person", "car")
- "box": 8 coordinates [x1, y1, x2, y2, x3, y3, x4, y4] representing the 4 corners in pixel coordinates

Format the response as valid JSON:
```json
[
  {
    "label": "object_name",
    "box": [x1, y1, x2, y2, x3, y3, x4, y4]
  }
]
```

Be specific with labels and ensure all coordinates are in pixel values.
EOF
    echo -e "${GREEN}✓ Default prompt created at shared/prompt.txt${NC}"
else
    echo -e "${YELLOW}⚠ Prompt file already exists, skipping${NC}"
fi

# Ask about GPU mode if GPU detected
if [ "$GPU_MODE" = true ]; then
    echo ""
    read -p "Use GPU acceleration? (Y/n): " use_gpu
    use_gpu=${use_gpu:-Y}
    
    if [[ ! $use_gpu =~ ^[Yy]$ ]]; then
        echo "Configuring for CPU mode..."
        GPU_MODE=false
    fi
fi

# Modify docker-compose.yml for CPU if needed
if [ "$GPU_MODE" = false ] && [ -f "docker-compose.yml" ]; then
    echo ""
    echo -e "${YELLOW}Note: To use CPU mode, you need to manually edit docker-compose.yml${NC}"
    echo "Follow the instructions in the file to disable GPU and enable CPU mode"
fi

# Check if files exist
echo ""
echo "Checking required files..."
required_files=(
    "docker-compose.yml"
    "sam-service/Dockerfile"
    "sam-service/app.py"
    "pipeline-api/Dockerfile"
    "pipeline-api/app.py"
    "vision_client.py"
    "requirements.txt"
)

missing_files=()
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -gt 0 ]; then
    echo -e "${RED}Error: Missing required files:${NC}"
    for file in "${missing_files[@]}"; do
        echo "  - $file"
    done
    echo ""
    echo "Please ensure all files are in the correct locations"
    exit 1
fi

echo -e "${GREEN}✓ All required files present${NC}"

# Build and start services
echo ""
echo "=========================================="
echo "Building and starting services..."
echo "=========================================="

$COMPOSE_CMD build

echo ""
echo "Starting services in background..."
$COMPOSE_CMD up -d

echo ""
echo "Waiting for services to be healthy (this may take a few minutes)..."

# Wait for services
max_wait=300
elapsed=0
interval=10

while [ $elapsed -lt $max_wait ]; do
    sleep $interval
    elapsed=$((elapsed + interval))
    
    # Check health endpoint
    if curl -s http://localhost:8080/health > /dev/null 2>&1; then
        health_status=$(curl -s http://localhost:8080/health | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
        if [ "$health_status" = "healthy" ]; then
            echo -e "${GREEN}✓ All services are healthy!${NC}"
            break
        fi
    fi
    
    echo "Still waiting... (${elapsed}s / ${max_wait}s)"
done

if [ $elapsed -ge $max_wait ]; then
    echo -e "${RED}Services did not become healthy in time${NC}"
    echo "Check logs with: docker-compose logs"
    exit 1
fi

# Check if Qwen model is available
echo ""
echo "Checking for Qwen model..."
models=$(docker exec vision-ollama ollama list 2>/dev/null | grep -i qwen | wc -l)

if [ "$models" -eq 0 ]; then
    echo -e "${YELLOW}⚠ No Qwen model found${NC}"
    echo ""
    echo "You need to pull a Qwen VL model. Choose one:"
    echo "  1. qwen2-vl:7b  (Recommended for testing, ~4GB)"
    echo "  2. qwen2-vl:72b (Best quality, ~40GB)"
    echo ""
    read -p "Enter choice (1 or 2): " model_choice
    
    if [ "$model_choice" = "1" ]; then
        model="qwen2-vl:7b"
    else
        model="qwen2-vl:72b"
    fi
    
    echo ""
    echo "Pulling $model (this will take several minutes)..."
    docker exec vision-ollama ollama pull $model
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Model downloaded successfully${NC}"
    else
        echo -e "${RED}Failed to download model${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ Qwen model(s) already available${NC}"
    docker exec vision-ollama ollama list | grep -i qwen
fi

# Install Python dependencies
echo ""
echo "Installing Python client dependencies..."
if command -v pip3 &> /dev/null; then
    pip3 install -r requirements.txt
elif command -v pip &> /dev/null; then
    pip install -r requirements.txt
else
    echo -e "${YELLOW}⚠ pip not found, skipping Python dependencies${NC}"
    echo "  Install manually with: pip install -r requirements.txt"
fi

# Final status check
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Service URLs:"
echo "  - Pipeline API: http://localhost:8080"
echo "  - SAM Service:  http://localhost:8001"
echo "  - Ollama:       http://localhost:11434"
echo ""
echo "Next steps:"
echo "  1. Add images to: ./shared/images/"
echo "  2. Edit prompt (optional): ./shared/prompt.txt"
echo "  3. Run client: python vision_client.py"
echo ""
echo "Useful commands:"
echo "  - View logs:    $COMPOSE_CMD logs -f"
echo "  - Check health: curl http://localhost:8080/health"
echo "  - Stop services: $COMPOSE_CMD down"
echo "  - List models:  docker exec vision-ollama ollama list"
echo ""

if [ "$GPU_MODE" = true ]; then
    echo -e "${GREEN}GPU acceleration: ENABLED${NC}"
else
    echo -e "${YELLOW}GPU acceleration: DISABLED (CPU mode)${NC}"
fi

echo ""
echo "Happy detecting! 🚀"
