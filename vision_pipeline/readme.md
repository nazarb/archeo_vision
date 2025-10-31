# Vision AI Pipeline with Qwen 2.5 VL + SAM

A complete Docker-based vision AI pipeline that combines Ollama's Qwen 2.5 VL (72B) for object detection with Meta's Segment Anything Model (SAM) for instance segmentation.

## Features

- **Object Detection**: Qwen 2.5 VL detects objects with 4-point bounding boxes
- **Instance Segmentation**: SAM converts 4-point boxes to precise segmentation masks
- **Unified API**: Coordinated pipeline with health checks
- **GPU Support**: NVIDIA CUDA acceleration with CPU fallback
- **Automated Processing**: Python client processes entire image folders
- **Rich Visualizations**: Colored masks, bounding boxes, and labels

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Ollama    │────▶│  Pipeline    │────▶│     SAM     │
│ Qwen 2.5 VL │     │     API      │     │   Service   │
│  :11434     │     │    :8080     │     │    :8001    │
└─────────────┘     └──────────────┘     └─────────────┘
       │                    │                     │
       └────────────────────┴─────────────────────┘
                            │
                    ┌───────▼────────┐
                    │  Vision Client │
                    │   (Python)     │
                    └────────────────┘
```

## Quick Start

### 1. Setup Project Structure

```bash
mkdir -p vision-pipeline/{sam-service,pipeline-api,shared/{images,detections,results,masks,visualizations}}
cd vision-pipeline

# Copy all files to appropriate directories
# - docker-compose.yml → ./
# - sam-service/Dockerfile and app.py → ./sam-service/
# - pipeline-api/Dockerfile and app.py → ./pipeline-api/
# - vision_client.py → ./
# - requirements.txt → ./
```

### 2. Create Prompt File

```bash
cat > shared/prompt.txt << 'EOF'
Detect all objects in this image. Return the results as a JSON array with each object having:
- "label": the object class name
- "box": 8 coordinates [x1, y1, x2, y2, x3, y3, x4, y4] representing the 4 corners in pixel coordinates

Format the response as:
```json
[
  {
    "label": "object_name",
    "box": [x1, y1, x2, y2, x3, y3, x4, y4]
  }
]
```
EOF
```

### 3. Add Images

```bash
# Copy your images to the shared/images folder
cp /path/to/your/images/* shared/images/
```

### 4. Start Services (GPU Mode)

```bash
# Build and start all services
docker-compose up -d

# Watch logs
docker-compose logs -f
```

**First Run**: Ollama will automatically download Qwen 2.5 VL (72B) model (~40GB). This may take 15-30 minutes depending on your connection.

### 5. Pull Qwen Model (Required on First Run)

```bash
# Wait for Ollama to be ready
docker exec -it vision-ollama ollama pull qwen2-vl:72b

# Or use a smaller model for testing
docker exec -it vision-ollama ollama pull qwen2-vl:7b
```

### 6. Check Health

```bash
curl http://localhost:8080/health
```

Expected response:
```json
{
  "status": "healthy",
  "services": {
    "ollama": {"status": "healthy", "models": 1},
    "sam": {"status": "healthy", "model_loaded": true}
  }
}
```

### 7. Process Images

```bash
# Install client dependencies
pip install -r requirements.txt

# Process all images
python vision_client.py

# With custom settings
python vision_client.py \
  --pipeline-url http://localhost:8080 \
  --sam-url http://localhost:8001 \
  --shared-path ./shared \
  --model qwen2-vl:72b
```

## CPU-Only Mode

To run without GPU:

1. Edit `docker-compose.yml`:
   - Comment out all `deploy.resources.reservations` sections
   - Uncomment CPU-only environment variables

2. For Ollama:
```yaml
environment:
  - OLLAMA_NUM_GPU=0
```

3. For SAM:
```yaml
environment:
  - FORCE_CPU=true
```

4. Restart:
```bash
docker-compose down
docker-compose up -d
```

**Note**: CPU mode is significantly slower. Consider using smaller models like `qwen2-vl:7b`.

## Output Structure

```
shared/
├── images/                    # Input images
├── detections/               # Detection stage outputs
│   ├── image_detection.json  # Raw detection results
│   └── image_detection.jpg   # Visualization with 4-point boxes
├── results/                  # Final pipeline results
│   └── image_result.json     # Combined detection + segmentation
├── masks/                    # Individual segmentation masks
│   ├── image_mask_0_dog.png
│   └── image_mask_1_cat.png
└── visualizations/           # Final visualizations
    └── image_segmentation.jpg # Colored masks + boxes + labels
```

## API Endpoints

### Pipeline API (port 8080)

#### GET /health
Check status of all services.

#### POST /detect
Detect objects using Qwen VL.

**Request:**
```json
{
  "image_base64": "base64_encoded_image",
  "prompt": "Detect all objects...",
  "model": "qwen2-vl:72b"
}
```

**Response:**
```json
{
  "success": true,
  "raw_response": "exact Qwen output",
  "boxes": [
    {
      "x1": 100, "y1": 150,
      "x2": 300, "y2": 150,
      "x3": 300, "y3": 400,
      "x4": 100, "y4": 400,
      "label": "dog"
    }
  ],
  "count": 1,
  "image_size": {"width": 800, "height": 600},
  "model_used": "qwen2-vl:72b"
}
```

#### POST /pipeline
Full detection + segmentation pipeline.

### SAM Service (port 8001)

#### GET /health
Check SAM service status.

#### POST /segment
Segment objects with 4-point boxes.

**Request:**
```json
{
  "image_base64": "base64_encoded_image",
  "boxes": [
    {
      "x1": 100, "y1": 150,
      "x2": 300, "y2": 150,
      "x3": 300, "y3": 400,
      "x4": 100, "y4": 400,
      "label": "dog"
    }
  ]
}
```

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "id": 0,
      "label": "dog",
      "confidence": 0.98,
      "box": {...},
      "mask_base64": "base64_png_mask",
      "polygons": [[[x1, y1], [x2, y2], ...]],
      "area": 15234
    }
  ]
}
```

## Configuration

### Environment Variables

**Ollama:**
- `OLLAMA_HOST`: Bind address (default: 0.0.0.0:11434)
- `OLLAMA_NUM_GPU`: Number of GPUs (0 for CPU)

**SAM Service:**
- `SAM_MODEL_TYPE`: vit_h (best), vit_l, or vit_b
- `FORCE_CPU`: Set to "true" for CPU mode
- `CUDA_VISIBLE_DEVICES`: GPU device ID

**Pipeline API:**
- `OLLAMA_URL`: Ollama service URL
- `SAM_URL`: SAM service URL
- `DEFAULT_MODEL`: Default Qwen model

### SAM Model Sizes

| Model | Size | Quality | Speed |
|-------|------|---------|-------|
| vit_h | 2.4GB | Best | Slow |
| vit_l | 1.2GB | Good | Medium |
| vit_b | 375MB | Decent | Fast |

Change in `docker-compose.yml`:
```yaml
environment:
  - SAM_MODEL_TYPE=vit_l  # or vit_b
```

## Troubleshooting

### Service won't start

```bash
# Check logs
docker-compose logs ollama
docker-compose logs sam-service
docker-compose logs pipeline-api

# Restart specific service
docker-compose restart ollama
```

### Out of memory

- Use smaller Qwen model: `qwen2-vl:7b`
- Use smaller SAM model: `vit_b`
- Reduce image resolution
- Switch to CPU mode

### No objects detected

- Check prompt format in `shared/prompt.txt`
- Verify Qwen model supports vision: `docker exec vision-ollama ollama list`
- Test with simpler prompt
- Check image quality and size

### Slow processing

- Use GPU acceleration
- Use smaller models
- Process fewer images at once
- Reduce image resolution

### GPU not detected

```bash
# Check NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Install nvidia-docker2
sudo apt-get install nvidia-docker2
sudo systemctl restart docker
```

## Advanced Usage

### Custom Prompts

Create specific detection prompts for your use case:

```text
Detect all people and vehicles in this street scene.
Return JSON with "label" (person/car/truck/bus) and "box" (8 coordinates).
```

### Batch Processing

```bash
# Process multiple folders
for folder in dataset1 dataset2 dataset3; do
  python vision_client.py --shared-path ./data/$folder
done
```

### API Integration

```python
import requests
import base64

# Encode image
with open("image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# Call pipeline
response = requests.post("http://localhost:8080/detect", json={
    "image_base64": image_b64,
    "prompt": "Detect all objects",
    "model": "qwen2-vl:72b"
})

print(response.json())
```

## Performance

Approximate processing times (RTX 3090):

| Stage | Time | Notes |
|-------|------|-------|
| Detection (Qwen 72B) | 5-15s | Depends on image complexity |
| Segmentation (SAM vit_h) | 1-3s | Per object detected |
| Total (3 objects) | 8-24s | Full pipeline |

CPU mode: 10-50x slower

## Data Format

### 4-Point Box Format

All boxes use 8 coordinates representing 4 corners:
```
[x1, y1, x2, y2, x3, y3, x4, y4]
```

Typical rectangular box:
```
(x1,y1)────(x2,y2)
   │          │
(x4,y4)────(x3,y3)
```

Coordinates are in **pixel space** (not normalized).

## License

This project combines multiple components:
- Ollama: MIT License
- Qwen 2.5 VL: Apache 2.0
- SAM: Apache 2.0
- FastAPI: MIT License

## Credits

- **Qwen 2.5 VL**: Alibaba Cloud
- **SAM**: Meta AI Research
- **Ollama**: Ollama Team

## Support

For issues and questions:
1. Check logs: `docker-compose logs -f`
2. Verify health: `curl http://localhost:8080/health`
3. Test individual services
4. Check GPU availability: `nvidia-smi`

---

**Warsaw Timezone**: All services use `Europe/Warsaw` timezone by default.
