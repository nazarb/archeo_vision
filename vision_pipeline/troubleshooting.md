# Troubleshooting Guide

## Service Issues

### Services Won't Start

**Symptom**: `docker-compose up` fails or services exit immediately

**Solutions**:

1. Check Docker daemon:
```bash
sudo systemctl status docker
sudo systemctl start docker
```

2. Check ports are available:
```bash
# Check if ports are in use
sudo lsof -i :11434  # Ollama
sudo lsof -i :8001   # SAM
sudo lsof -i :8080   # Pipeline

# Kill processes if needed
sudo kill -9 <PID>
```

3. Check logs:
```bash
docker-compose logs ollama
docker-compose logs sam-service
docker-compose logs pipeline-api
```

4. Rebuild containers:
```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### GPU Not Detected

**Symptom**: Services start but don't use GPU

**Solutions**:

1. Check NVIDIA driver:
```bash
nvidia-smi
```

2. Check NVIDIA Docker runtime:
```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

3. Install nvidia-docker2:
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

4. Verify Docker can see GPU:
```bash
docker info | grep -i runtime
# Should show: Runtimes: nvidia runc
```

### Out of Memory Errors

**Symptom**: Services crash with OOM errors

**Solutions**:

1. Use smaller models:
```bash
# Qwen
docker exec vision-ollama ollama pull qwen2.5-vl:7b  # Smaller alternative to qwen3-vl:8b
```

2. Change SAM model size in `docker-compose.yml`:
```yaml
environment:
  - SAM_MODEL_TYPE=vit_b  # smallest, fastest
  # - SAM_MODEL_TYPE=vit_l  # medium
  # - SAM_MODEL_TYPE=vit_h  # largest, best quality
```

3. Process fewer images at once:
```bash
# Process one at a time
for img in shared/images/*.jpg; do
  python vision_client.py --shared-path shared/single/
done
```

4. Reduce image resolution:
```python
from PIL import Image

img = Image.open("large.jpg")
img = img.resize((1024, 768))  # Reduce size
img.save("resized.jpg")
```

5. Switch to CPU mode (frees GPU memory)

## Detection Issues

### No Objects Detected

**Symptom**: Detection returns empty results or count: 0

**Solutions**:

1. Check prompt format:
```text
# Good prompt
Detect all objects in this image. Return JSON with label and box [x1,y1,x2,y2,x3,y3,x4,y4].

# Bad prompt (too vague)
Find things
```

2. Verify model is loaded:
```bash
docker exec vision-ollama ollama list
# Should show qwen3-vl model
```

3. Test with simpler prompt:
```text
List all objects you can see in this image.
```

4. Check image quality:
- Is the image too dark/bright?
- Is resolution too low?
- Is the file corrupted?

5. Check raw response:
```bash
# Look at detection JSON
cat shared/detections/*_detection.json | jq .raw_response
```

### Wrong Object Labels

**Symptom**: Objects detected but labels are generic or wrong

**Solutions**:

1. Be more specific in prompt:
```text
# Instead of:
Detect objects

# Use:
Detect and classify objects. Be specific:
- Animals: specify species (dog, cat, bird, etc.)
- Vehicles: specify type (car, truck, bus, motorcycle)
- People: label as "person"
```

2. Use better quality model:
```bash
# Use the default qwen3-vl:8b model
docker exec vision-ollama ollama pull qwen3-vl:8b
```

3. Provide examples in prompt:
```text
Detect objects and return JSON like:
[
  {"label": "golden_retriever", "box": [...]},
  {"label": "red_car", "box": [...]}
]
Be specific with breeds, colors, and types.
```

### Incorrect Bounding Boxes

**Symptom**: Boxes are in wrong positions or wrong size

**Solutions**:

1. Verify coordinates are in pixels (not normalized):
```python
# Check detection JSON
import json
with open('shared/detections/image_detection.json') as f:
    data = json.load(f)
    box = data['boxes'][0]
    print(f"Box coords: {box}")
    # Should be large numbers (pixels), not 0-1 range
```

2. Check image size matches:
```python
from PIL import Image
img = Image.open('shared/images/test.jpg')
print(f"Image size: {img.size}")  # Compare with detection.image_size
```

3. Test with different prompt format:
```text
Detect objects. For each object, return the bounding box as 4 corner points 
in pixel coordinates: [top-left-x, top-left-y, top-right-x, top-right-y, 
bottom-right-x, bottom-right-y, bottom-left-x, bottom-left-y]
```

## Segmentation Issues

### Segmentation Fails After Detection

**Symptom**: Detection works but segmentation errors

**Solutions**:

1. Check SAM service health:
```bash
curl http://localhost:8001/health
```

2. Verify SAM model loaded:
```bash
docker-compose logs sam-service | grep "initialized"
```

3. Check if boxes are valid:
```python
# Boxes should have all 8 coordinates
import json
with open('shared/detections/image_detection.json') as f:
    data = json.load(f)
    for box in data['boxes']:
        assert all(k in box for k in ['x1','y1','x2','y2','x3','y3','x4','y4'])
```

4. Restart SAM service:
```bash
docker-compose restart sam-service
docker-compose logs -f sam-service
```

### Poor Segmentation Quality

**Symptom**: Masks are inaccurate or incomplete

**Solutions**:

1. Use better SAM model:
```yaml
# In docker-compose.yml
environment:
  - SAM_MODEL_TYPE=vit_h  # Best quality
```

2. Improve detection boxes:
- More accurate boxes → better segmentation
- Adjust detection prompt for tighter boxes

3. Check box-to-mask conversion:
```python
# Boxes should tightly fit objects
# Review detection visualizations in shared/detections/
```

### Missing Masks

**Symptom**: Some objects don't get segmentation masks

**Solutions**:

1. Check SAM confidence threshold:
```python
# In sam-service/app.py, masks with low confidence may be filtered
# Check segmentation results JSON
with open('shared/results/image_result.json') as f:
    data = json.load(f)
    for obj in data['segmentation']['results']:
        print(f"{obj['label']}: confidence {obj['confidence']}")
```

2. Verify all boxes are processed:
```bash
# Compare counts
cat shared/detections/image_detection.json | jq .count
cat shared/results/image_result.json | jq '.segmentation.results | length'
```

## Performance Issues

### Very Slow Processing

**Symptom**: Each image takes minutes or hangs

**Solutions**:

1. Check if using GPU:
```bash
# During processing, check GPU usage
nvidia-smi -l 1
```

2. Use smaller models:
```bash
# Qwen 2.5-VL 7B (smaller alternative)
docker exec vision-ollama ollama pull qwen2.5-vl:7b

# SAM vit_b instead of vit_h
# Edit docker-compose.yml SAM_MODEL_TYPE=vit_b
```

3. Reduce image size:
```python
# Resize before processing
from PIL import Image
img = Image.open('large.jpg')
img.thumbnail((1920, 1080))  # Max dimensions
img.save('resized.jpg')
```

4. Check network latency:
```bash
# If using remote Docker host
time curl http://localhost:8080/health
```

5. Monitor resources:
```bash
# CPU/Memory usage
docker stats

# Disk I/O
iostat -x 1
```

### Timeout Errors

**Symptom**: Requests timeout before completing

**Solutions**:

1. Increase timeout in client:
```python
# In vision_client.py
response = requests.post(url, json=payload, timeout=600)  # 10 minutes
```

2. Process fewer objects per image:
```text
# In prompt
Detect only the 3 largest objects in this image.
```

3. Check service logs for bottlenecks:
```bash
docker-compose logs -f --tail=100
```

## API Issues

### Connection Refused

**Symptom**: Cannot connect to API endpoints

**Solutions**:

1. Check services are running:
```bash
docker-compose ps
```

2. Check port bindings:
```bash
docker port vision-pipeline
docker port vision-sam
docker port vision-ollama
```

3. Test from inside container:
```bash
docker exec vision-pipeline curl http://localhost:8080/health
```

4. Check firewall:
```bash
sudo ufw status
sudo ufw allow 8080/tcp
sudo ufw allow 8001/tcp
sudo ufw allow 11434/tcp
```

### JSON Parse Errors

**Symptom**: "Failed to parse JSON" or similar errors

**Solutions**:

1. Check Qwen response format:
```bash
# Look at raw_response in detection JSON
cat shared/detections/*_detection.json | jq -r .raw_response
```

2. Update prompt to enforce JSON:
```text
IMPORTANT: Return ONLY valid JSON, no explanatory text.
Format: [{"label": "object", "box": [x1,y1,x2,y2,x3,y3,x4,y4]}]
```

3. Handle markdown code blocks:
```python
# Parser already handles ```json ... ```
# Check pipeline-api logs for parsing errors
docker-compose logs pipeline-api | grep -i "parse"
```

### Rate Limiting / Too Many Requests

**Symptom**: Requests fail after several images

**Solutions**:

1. Add delays between requests:
```python
import time
for image_path in image_files:
    process_image(image_path)
    time.sleep(2)  # 2 second delay
```

2. Process in smaller batches:
```bash
# Instead of all at once
ls shared/images/*.jpg | head -10 | xargs -I {} cp {} batch1/
python vision_client.py --shared-path batch1
```

## Client Issues

### Python Dependencies Missing

**Symptom**: ImportError when running client

**Solutions**:

1. Install requirements:
```bash
pip install -r requirements.txt

# Or with specific versions
pip install opencv-python==4.8.1.78 numpy==1.24.3 Pillow==10.1.0 requests==2.31.0
```

2. Use virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

3. Check Python version:
```bash
python --version  # Should be 3.8+
```

### Cannot Read/Write Files

**Symptom**: Permission denied or file not found errors

**Solutions**:

1. Check permissions:
```bash
ls -la shared/
chmod -R 755 shared/
```

2. Check paths:
```bash
# Ensure correct structure
tree shared/
```

3. Run with correct working directory:
```bash
cd /path/to/vision-pipeline
python vision_client.py
```

### Visualizations Not Created

**Symptom**: No images in shared/visualizations/

**Solutions**:

1. Check OpenCV installation:
```python
import cv2
print(cv2.__version__)
```

2. Check file write permissions:
```bash
touch shared/visualizations/test.txt
rm shared/visualizations/test.txt
```

3. Check for errors in client output:
```bash
python vision_client.py 2>&1 | tee client.log
grep -i error client.log
```

## Data Issues

### Images Not Found

**Symptom**: Client says "No images found"

**Solutions**:

1. Check image extensions:
```bash
ls shared/images/
# Supported: .jpg, .jpeg, .png, .bmp, .webp
```

2. Check file permissions:
```bash
ls -la shared/images/
chmod 644 shared/images/*
```

3. Use absolute path:
```bash
python vision_client.py --shared-path /full/path/to/shared
```

### Corrupted Output Files

**Symptom**: JSON files are empty or invalid

**Solutions**:

1. Check disk space:
```bash
df -h
```

2. Check for write errors:
```bash
docker-compose logs pipeline-api | grep -i "write\|save"
```

3. Validate JSON:
```bash
python -m json.tool shared/results/image_result.json
```

## Docker Issues

### Build Failures

**Symptom**: docker-compose build fails

**Solutions**:

1. Check Docker version:
```bash
docker --version  # Should be 20.10+
docker-compose --version  # Should be 1.29+
```

2. Clear build cache:
```bash
docker-compose build --no-cache
```

3. Check network connectivity:
```bash
docker pull python:3.10-slim
```

4. Increase build memory:
```json
// In Docker Desktop settings or daemon.json
{
  "memory": "8G"
}
```

### Volume Mount Issues

**Symptom**: Changes to shared/ not reflected in container

**Solutions**:

1. Check mount paths:
```bash
docker inspect vision-pipeline | grep -A 10 Mounts
```

2. Restart containers:
```bash
docker-compose down
docker-compose up -d
```

3. Use absolute paths in docker-compose.yml:
```yaml
volumes:
  - /full/path/to/shared:/shared
```

## Getting Help

### Collect Diagnostic Information

```bash
# System info
uname -a
docker --version
docker-compose --version
nvidia-smi  # If using GPU

# Service status
docker-compose ps
curl http://localhost:8080/health

# Logs (last 100 lines)
docker-compose logs --tail=100 > logs.txt

# Resource usage
docker stats --no-stream

# Disk space
df -h
```

### Enable Debug Logging

Edit service app.py files:

```python
# Change logging level
logging.basicConfig(level=logging.DEBUG)
```

Rebuild and restart:
```bash
docker-compose build
docker-compose up -d
docker-compose logs -f
```

### Test Individual Components

```bash
# Test Ollama directly
curl -X POST http://localhost:11434/api/generate \
  -d '{"model": "qwen3-vl:8b", "prompt": "Hello"}'

# Test SAM directly
curl http://localhost:8001/health

# Test Pipeline health
curl http://localhost:8080/health
```

### Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| "CUDA out of memory" | GPU memory full | Use smaller models or CPU mode |
| "Model not found" | Qwen not pulled | `ollama pull qwen3-vl:8b` |
| "Connection refused" | Service not running | `docker-compose up -d` |
| "Timeout" | Request too slow | Increase timeout, use smaller models |
| "JSON parse error" | Invalid response | Improve prompt, check raw_response |
| "Permission denied" | File access issue | Check permissions with `chmod` |
| "No such file" | Missing file/directory | Run setup script or create manually |

## Still Having Issues?

1. Run test script: `./test_pipeline.sh`
2. Check all service logs: `docker-compose logs`
3. Verify setup: Review README.md and QUICKSTART.md
4. Start fresh: `docker-compose down -v && ./setup.sh`

Remember to check logs for detailed error messages - they usually point to the exact problem!
