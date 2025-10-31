# Quick Start Guide

## 5-Minute Setup

### 1. Clone and Setup

```bash
# Download all files to a directory
mkdir vision-pipeline && cd vision-pipeline

# Make setup script executable
chmod +x setup.sh

# Run automated setup
./setup.sh
```

The setup script will:
- Create directory structure
- Build Docker images
- Start all services
- Download SAM model
- Prompt you to download Qwen model

### 2. Add Test Images

```bash
# Copy your images
cp /path/to/images/*.jpg shared/images/

# Or download a test image
wget -O shared/images/test.jpg https://example.com/image.jpg
```

### 3. Process Images

```bash
# Install Python dependencies (if not done by setup)
pip install -r requirements.txt

# Run the client
python vision_client.py
```

### 4. Check Results

```bash
# View detection results
ls shared/detections/

# View segmentation results
ls shared/results/
ls shared/masks/
ls shared/visualizations/
```

## Manual Setup (Without Script)

### 1. Start Services

```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# Watch logs
docker-compose logs -f
```

### 2. Download Qwen Model

```bash
# For testing (7B model, ~4GB)
docker exec vision-ollama ollama pull qwen2-vl:7b

# For production (72B model, ~40GB)
docker exec vision-ollama ollama pull qwen2-vl:72b
```

### 3. Check Health

```bash
curl http://localhost:8080/health
```

Should return:
```json
{
  "status": "healthy",
  "services": {
    "ollama": {"status": "healthy", "models": 1},
    "sam": {"status": "healthy", "model_loaded": true}
  }
}
```

### 4. Process Images

```bash
pip install -r requirements.txt
python vision_client.py
```

## CPU-Only Mode

If you don't have an NVIDIA GPU:

### Option 1: Docker Compose Override

Create `docker-compose.override.yml`:

```yaml
version: '3.8'

services:
  ollama:
    deploy: {}
    environment:
      - TZ=Europe/Warsaw
      - OLLAMA_HOST=0.0.0.0:11434
      - OLLAMA_NUM_GPU=0

  sam-service:
    deploy: {}
    environment:
      - TZ=Europe/Warsaw
      - SAM_MODEL_TYPE=vit_b
      - FORCE_CPU=true

  pipeline-api:
    # No changes needed
```

Then run:
```bash
docker-compose up -d
```

### Option 2: Edit docker-compose.yml

1. Remove all `deploy.resources.reservations` sections
2. Set `OLLAMA_NUM_GPU=0` for ollama service
3. Set `FORCE_CPU=true` for sam-service
4. Consider using smaller models (vit_b for SAM)

## Testing

Run the test suite:

```bash
chmod +x test_pipeline.sh
./test_pipeline.sh
```

## Common Issues

### Services won't start

```bash
# Check Docker
docker ps

# Check logs
docker-compose logs ollama
docker-compose logs sam-service

# Restart
docker-compose restart
```

### Out of memory

Use smaller models:
```bash
# Smaller Qwen model
docker exec vision-ollama ollama pull qwen2-vl:7b

# Smaller SAM model (edit docker-compose.yml)
SAM_MODEL_TYPE=vit_b
```

### No detections

1. Check prompt format in `shared/prompt.txt`
2. Verify model is loaded: `docker exec vision-ollama ollama list`
3. Try simpler prompt
4. Check image quality

### Slow processing

- Enable GPU acceleration
- Use smaller models
- Reduce image resolution
- Process in batches

## Example Prompts

### General Detection
```
Detect all objects in this image. Return JSON with label and box [x1,y1,x2,y2,x3,y3,x4,y4].
```

### Specific Objects
```
Detect all people and vehicles. Return JSON array with:
- label: person, car, truck, or bus
- box: 8 coordinates in pixels
```

### Detailed Detection
```
Analyze this image and detect:
- All animals (specify species)
- All vehicles (specify type)
- All buildings or structures

Return JSON with label (be specific) and box coordinates.
```

## Performance Tips

### GPU Mode (Fastest)
- Use qwen2-vl:72b for best quality
- Use SAM vit_h for best segmentation
- Expect 8-24 seconds per image (3 objects)

### CPU Mode (Slower)
- Use qwen2-vl:7b
- Use SAM vit_b
- Expect 2-10 minutes per image
- Process fewer images at once

### Batch Processing
```bash
# Process multiple folders
for dir in folder1 folder2 folder3; do
  python vision_client.py --shared-path ./data/$dir
done
```

## Next Steps

1. ✓ Services running
2. ✓ Model downloaded
3. ✓ Images processed
4. → Integrate into your workflow
5. → Customize prompts for your use case
6. → Scale with multiple instances

## API Usage

### Detection Only
```python
import requests
import base64

with open("image.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

response = requests.post("http://localhost:8080/detect", json={
    "image_base64": img_b64,
    "prompt": "Detect all objects"
})

print(response.json())
```

### Full Pipeline
```python
response = requests.post("http://localhost:8080/pipeline", json={
    "image_base64": img_b64,
    "prompt": "Detect all objects"
})

result = response.json()
detections = result["detection"]["boxes"]
segmentations = result["segmentation"]["results"]
```

## Resources

- **Ollama**: https://ollama.ai/
- **Qwen 2.5 VL**: https://github.com/QwenLM/Qwen2-VL
- **SAM**: https://github.com/facebookresearch/segment-anything
- **Docker**: https://docs.docker.com/

## Support

Check logs for detailed error messages:
```bash
docker-compose logs -f
```

Test individual services:
```bash
# Test Ollama
curl http://localhost:11434/api/tags

# Test SAM
curl http://localhost:8001/health

# Test Pipeline
curl http://localhost:8080/health
```
