#!/bin/bash
# Quick rebuild script after fixes

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
    COMPOSE_CMD="sudo docker-compose"
elif docker compose version &> /dev/null 2>&1; then
    COMPOSE_CMD="sudo docker compose"
else
    COMPOSE_CMD="sudo docker compose"
fi

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}Rebuilding Vision AI Pipeline${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""
echo "Using command: $COMPOSE_CMD"
echo ""

# Step 1: Stop and clean
echo -e "${YELLOW}Step 1: Stopping existing containers...${NC}"
$COMPOSE_CMD down 2>/dev/null || true

# Step 2: Remove old images (optional)
read -p "Remove old images to force complete rebuild? (y/N): " remove_images
if [[ $remove_images =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Removing old images...${NC}"
    docker rmi vision-pipeline-sam-service 2>/dev/null || true
    docker rmi vision-pipeline-pipeline-api 2>/dev/null || true
fi

# Step 3: Build
echo ""
echo -e "${YELLOW}Step 2: Building services (this may take 5-10 minutes)...${NC}"
echo ""

if $COMPOSE_CMD build --no-cache; then
    echo ""
    echo -e "${GREEN}✓ Build successful!${NC}"
else
    echo ""
    echo -e "${RED}✗ Build failed${NC}"
    echo ""
    echo "Common issues:"
    echo "  1. Network connectivity (downloading packages)"
    echo "  2. Disk space (run: df -h)"
    echo "  3. Docker daemon not running (run: sudo systemctl start docker)"
    echo ""
    echo "Check logs above for specific errors"
    exit 1
fi

# Step 4: Start services
echo ""
echo -e "${YELLOW}Step 3: Starting services...${NC}"

if $COMPOSE_CMD up -d; then
    echo ""
    echo -e "${GREEN}✓ Services started!${NC}"
else
    echo ""
    echo -e "${RED}✗ Failed to start services${NC}"
    exit 1
fi

# Step 5: Wait for health checks
echo ""
echo -e "${YELLOW}Step 4: Waiting for services to be healthy...${NC}"
echo "This may take 1-2 minutes..."
echo ""

sleep 10

# Check each service
max_wait=120
elapsed=0
interval=5

while [ $elapsed -lt $max_wait ]; do
    # Check Ollama
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        ollama_status="✓"
    else
        ollama_status="⏳"
    fi
    
    # Check SAM
    if curl -s http://localhost:8001/health > /dev/null 2>&1; then
        sam_status="✓"
    else
        sam_status="⏳"
    fi
    
    # Check Pipeline
    if curl -s http://localhost:8080/health > /dev/null 2>&1; then
        pipeline_status="✓"
    else
        pipeline_status="⏳"
    fi
    
    echo -ne "\rOllama: $ollama_status  SAM: $sam_status  Pipeline: $pipeline_status  (${elapsed}s)"
    
    # Check if all are ready
    if [[ "$ollama_status" == "✓" && "$sam_status" == "✓" && "$pipeline_status" == "✓" ]]; then
        echo ""
        echo ""
        echo -e "${GREEN}✓ All services are healthy!${NC}"
        break
    fi
    
    sleep $interval
    elapsed=$((elapsed + interval))
done

if [ $elapsed -ge $max_wait ]; then
    echo ""
    echo ""
    echo -e "${YELLOW}⚠ Services started but may still be initializing${NC}"
    echo "Check status with: $COMPOSE_CMD ps"
    echo "View logs with: $COMPOSE_CMD logs -f"
fi

# Step 6: Show status
echo ""
echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}Service Status${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

$COMPOSE_CMD ps

echo ""
echo -e "${GREEN}Next steps:${NC}"
echo "  1. Check logs: $COMPOSE_CMD logs -f"
echo "  2. Pull Qwen model: docker exec vision-ollama ollama pull qwen2-vl:7b"
echo "  3. Add images: cp your_images/* shared/images/"
echo "  4. Run client: python vision_client.py"
echo ""
echo "Service URLs:"
echo "  - Pipeline API: http://localhost:8080"
echo "  - SAM Service:  http://localhost:8001"
echo "  - Ollama:       http://localhost:11434"
echo ""
