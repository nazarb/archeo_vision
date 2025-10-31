# Archaeological Vision Pipeline

Specialized system for detecting and labeling pottery pieces in archaeological documentation images.

## Purpose

This pipeline automatically:
1. Detects pottery pieces in images
2. Reads text labels on each piece
3. Identifies the main catalog label (yellow sticky note)
4. Generates structured JSON output with bounding boxes

## Quick Start

### 1. Setup Directory Structure

```bash
mkdir -p archeo-shared/{images,results,visualizations}
```

### 2. Add Images

```bash
cp your_pottery_images/* archeo-shared/images/
```

### 3. Ensure Services Are Running

```bash
# Check services
sudo docker compose ps

# If not running
sudo docker compose up -d
```

### 4. Run the Pipeline

```bash
python archeo_vision_client.py --model qwen2.5-vl:7b
```

## Output Format

The pipeline generates JSON files in this exact format:

```json
{
  "main_label": "2025-B2164-A3",
  "image_path": "archeo-shared/images/pottery_001.jpg",
  "image_width": 2560,
  "image_height": 975,
  "annotations": [
    {
      "bbox_2d": [88, 97, 252, 222],
      "label": "D25B2164A3-4"
    },
    {
      "bbox_2d": [274, 134, 472, 294],
      "label": "D25B2164A3-1"
    }
  ]
}
```

## Directory Structure

```
archeo-shared/
├── images/              # Input: Your pottery photos
├── results/             # Output: JSON files with annotations
├── visualizations/      # Output: Images with bounding boxes
└── prompt.txt          # Detection prompt (auto-created)
```

## Command Line Options

```bash
python archeo_vision_client.py \
  --pipeline-url http://localhost:8080 \
  --shared-path ./archeo-shared \
  --model qwen2.5-vl:7b
```

### Available Models

- `qwen2.5-vl:7b` - Faster, good for testing
- `qwen2-vl:72b` - More accurate, slower
- `llava:13b` - Alternative model

## Output Files

For each input image `pottery_001.jpg`:

### JSON Result (`results/pottery_001_pottery.json`)
```json
{
  "main_label": "catalog number from yellow note",
  "image_path": "path/to/image",
  "image_width": 2560,
  "image_height": 975,
  "annotations": [
    {
      "bbox_2d": [left, top, right, bottom],
      "label": "text on pottery"
    }
  ]
}
```

### Visualization (`visualizations/pottery_001_labeled.jpg`)
- Bounding boxes around each pottery piece
- Labels displayed above each box
- Main catalog label shown at top
- Count of detected pieces

## Tips for Best Results

### Image Quality
- Use good lighting (avoid shadows)
- Ensure text on pottery is clearly visible
- Include the yellow catalog label card
- Use the scale bar for reference

### Label Readability
- Write labels clearly on pottery
- Use high contrast (dark on light background)
- Ensure labels are not obscured

### Image Composition
- Place pottery pieces with space between them
- Keep scale bar visible
- Position yellow label card prominently
- Use neutral background (gray/white)

## Troubleshooting

### No Detections
- Check if prompt.txt exists in archeo-shared/
- Verify image quality and contrast
- Try different model: `--model qwen2-vl:72b`

### Wrong Labels
- Ensure text is clearly visible
- Check image resolution (2560x975 or higher recommended)
- Verify lighting conditions

### Bounding Boxes Incorrect
- Check if model reported correct image size in JSON
- Pipeline automatically scales coordinates
- Review logs for scaling information

## Integration with Docker Services

The archaeological pipeline uses the same Docker infrastructure:

```bash
# Check service health
curl http://localhost:8080/health

# View logs
sudo docker compose logs -f pipeline-api

# Restart if needed
sudo docker compose restart
```

## Batch Processing

Process multiple images:

```bash
# Add all images
cp batch_of_images/* archeo-shared/images/

# Process all at once
python archeo_vision_client.py

# Results appear in archeo-shared/results/
```

## Advanced Usage

### Custom Prompt

Edit `archeo-shared/prompt.txt` to customize detection:

```text
You are provided with an image of size {width} x {height}...
[your custom instructions]
```

The `{width}` and `{height}` placeholders are automatically replaced with actual image dimensions.

### Processing Single Image

```python
from pathlib import Path
from archeo_vision_client import ArcheoVisionClient

client = ArcheoVisionClient()
prompt = client.load_prompt()
client.process_image(Path("archeo-shared/images/pottery.jpg"), prompt)
```

## Expected Performance

- **Processing time**: 5-15 seconds per image (7B model)
- **Accuracy**: High for clear labels and well-lit images
- **GPU memory**: ~5-8GB (7B model), ~40GB (72B model)

## File Organization Tool

After processing images, use the organizer to rename and organize files by their catalog labels.

### Usage

```bash
# Basic organization
python archeo_file_organizer.py

# With index creation
python archeo_file_organizer.py --create-index
```

### What It Does

1. **Reads JSON files** from `archeo-shared/results/`
2. **Extracts main_label** (e.g., "2025-B2164-A3")
3. **Finds matching image** in `archeo-shared/images/`
4. **Copies both files** to `archeo-shared/final/` with new names:
   - Image: `2025-B2164-A3.jpg`
   - JSON: `2025-B2164-A3.json`
5. **Updates image_path** in JSON to reflect new location

### Example

**Before:**
```
archeo-shared/
├── images/
│   └── IMG_20250128_143022.jpg
└── results/
    └── IMG_20250128_143022_pottery.json
```

**After:**
```
archeo-shared/
└── final/
    ├── 2025-B2164-A3.jpg        # Renamed image
    ├── 2025-B2164-A3.json       # Renamed JSON with updated path
    ├── index.json                # Optional: Collection index
    └── index.txt                 # Optional: Human-readable listing
```

### Index Files

Use `--create-index` to generate:

**index.json:**
```json
{
  "total_files": 5,
  "pottery_collections": [
    {
      "main_label": "2025-B2164-A3",
      "image_file": "2025-B2164-A3.jpg",
      "json_file": "2025-B2164-A3.json",
      "pottery_count": 6,
      "labels": ["D25B2164A3-1", "D25B2164A3-2", ...]
    }
  ]
}
```

**index.txt:**
```
Archaeological Pottery Collection Index
============================================================

1. 2025-B2164-A3
   Image: 2025-B2164-A3.jpg
   Pottery pieces: 6
   Labels: D25B2164A3-1, D25B2164A3-2, ...
```

## Complete Workflow

### 1. Process Images

```bash
python archeo_vision_client.py --model qwen2.5-vl:7b
```

### 2. Organize Files

```bash
python archeo_file_organizer.py --create-index
```

### 3. Results

All files are now organized in `archeo-shared/final/` with catalog-based naming!
