#!/bin/bash
# Test script for Vision AI Pipeline

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Detect compose command
COMPOSE_CMD=""
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
elif docker compose version &> /dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
elif sudo docker compose version &> /dev/null 2>&1; then
    COMPOSE_CMD="sudo docker compose"
else
    echo -e "${RED}Error: Docker Compose is not installed${NC}"
    exit 1
fi

echo "=========================================="
echo "Vision AI Pipeline Test Suite"
echo "=========================================="
echo "Using: $COMPOSE_CMD"

# Test 1: Check if services are running
echo -e "\n${BLUE}Test 1: Checking if services are running...${NC}"
if sudo docker ps | grep -q vision-ollama && \
   sudo docker ps | grep -q vision-sam && \
   sudo docker ps | grep -q vision-pipeline; then
    echo -e "${GREEN}✓ All services are running${NC}"
else
    echo -e "${RED}✗ Some services are not running${NC}"
    echo "Start with: $COMPOSE_CMD up -d"
    exit 1
fi

# Test 2: Health check endpoints
echo -e "\n${BLUE}Test 2: Checking health endpoints...${NC}"

# Ollama health
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Ollama is responding${NC}"
else
    echo -e "${RED}✗ Ollama is not responding${NC}"
    exit 1
fi

# SAM health
sam_health=$(curl -s http://localhost:8001/health | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
if [ "$sam_health" = "healthy" ]; then
    echo -e "${GREEN}✓ SAM service is healthy${NC}"
else
    echo -e "${RED}✗ SAM service is not healthy${NC}"
    exit 1
fi

# Pipeline health
pipeline_health=$(curl -s http://localhost:8080/health | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
if [ "$pipeline_health" = "healthy" ] || [ "$pipeline_health" = "degraded" ]; then
    echo -e "${GREEN}✓ Pipeline API is responding${NC}"
else
    echo -e "${RED}✗ Pipeline API is not healthy${NC}"
    exit 1
fi

# Test 3: Check for Qwen models
echo -e "\n${BLUE}Test 3: Checking for Qwen models...${NC}"
qwen_models=$(docker exec vision-ollama ollama list 2>/dev/null | grep -i qwen || true)
if [ -n "$qwen_models" ]; then
    echo -e "${GREEN}✓ Qwen models found:${NC}"
    echo "$qwen_models"
else
    echo -e "${YELLOW}⚠ No Qwen models found${NC}"
    echo "Pull a model with: docker exec vision-ollama ollama pull qwen2-vl:7b"
fi

# Test 4: Check directories
echo -e "\n${BLUE}Test 4: Checking directory structure...${NC}"
required_dirs=(
    "shared/images"
    "shared/detections"
    "shared/results"
    "shared/masks"
    "shared/visualizations"
)

all_exist=true
for dir in "${required_dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo -e "${GREEN}✓ $dir${NC}"
    else
        echo -e "${RED}✗ $dir (missing)${NC}"
        all_exist=false
    fi
done

if [ "$all_exist" = false ]; then
    echo -e "${YELLOW}Creating missing directories...${NC}"
    mkdir -p shared/{images,detections,results,masks,visualizations}
fi

# Test 5: Check for images
echo -e "\n${BLUE}Test 5: Checking for test images...${NC}"
image_count=$(find shared/images -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) 2>/dev/null | wc -l)
if [ "$image_count" -gt 0 ]; then
    echo -e "${GREEN}✓ Found $image_count image(s)${NC}"
    find shared/images -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) -exec basename {} \;
else
    echo -e "${YELLOW}⚠ No images found in shared/images/${NC}"
    echo "Add images to test the pipeline"
fi

# Test 6: Check prompt file
echo -e "\n${BLUE}Test 6: Checking prompt file...${NC}"
if [ -f "shared/prompt.txt" ]; then
    prompt_size=$(wc -c < shared/prompt.txt)
    if [ "$prompt_size" -gt 10 ]; then
        echo -e "${GREEN}✓ Prompt file exists (${prompt_size} bytes)${NC}"
    else
        echo -e "${YELLOW}⚠ Prompt file is too small${NC}"
    fi
else
    echo -e "${RED}✗ Prompt file not found${NC}"
    echo "Create one at: shared/prompt.txt"
fi

# Test 7: Check Python dependencies
echo -e "\n${BLUE}Test 7: Checking Python dependencies...${NC}"
if python3 -c "import cv2, numpy, PIL, requests" 2>/dev/null; then
    echo -e "${GREEN}✓ Python dependencies installed${NC}"
else
    echo -e "${YELLOW}⚠ Some Python dependencies missing${NC}"
    echo "Install with: pip install -r requirements.txt"
fi

# Test 8: Test API endpoint (if test image exists)
echo -e "\n${BLUE}Test 8: Testing detection API...${NC}"
test_image=$(find shared/images -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) 2>/dev/null | head -n 1)

if [ -n "$test_image" ] && [ -n "$qwen_models" ]; then
    echo "Using test image: $(basename $test_image)"
    echo "Sending test request (this may take 10-30 seconds)..."
    
    # Encode image to base64
    image_base64=$(base64 -w 0 "$test_image")
    
    # Create JSON payload
    json_payload=$(cat <<EOF
{
  "image_base64": "$image_base64",
  "prompt": "Detect all objects in this image. Return JSON with label and box [x1,y1,x2,y2,x3,y3,x4,y4]."
}
EOF
)
    
    # Send request
    response=$(curl -s -X POST http://localhost:8080/detect \
        -H "Content-Type: application/json" \
        -d "$json_payload" \
        --max-time 60 || echo '{"error":"timeout"}')
    
    if echo "$response" | grep -q '"success":true'; then
        count=$(echo "$response" | grep -o '"count":[0-9]*' | cut -d':' -f2)
        echo -e "${GREEN}✓ Detection API working! Detected $count object(s)${NC}"
    elif echo "$response" | grep -q 'timeout'; then
        echo -e "${YELLOW}⚠ Request timed out (model may still be loading)${NC}"
    else
        echo -e "${YELLOW}⚠ Detection returned no results${NC}"
        echo "Response: $(echo $response | head -c 200)"
    fi
else
    echo -e "${YELLOW}⚠ Skipping (no test image or model)${NC}"
fi

# Summary
echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo ""
echo "Services Status:"
$COMPOSE_CMD ps
echo ""
echo "To run the full pipeline:"
echo "  python vision_client.py"
echo ""
echo "To view logs:"
echo "  $COMPOSE_CMD logs -f"
echo ""
echo "To restart services:"
echo "  $COMPOSE_CMD restart"
echo ""
