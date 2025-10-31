import os
import re
import json
import logging
import base64
from typing import List, Dict, Any, Optional
import requests
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Vision AI Pipeline API")

# Configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
SAM_URL = os.getenv("SAM_URL", "http://sam-service:8001")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "qwen3-vl:8b")


class DetectionBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    x3: float
    y3: float
    x4: float
    y4: float
    label: str


class DetectionRequest(BaseModel):
    image_base64: str
    prompt: str
    model: Optional[str] = None


class DetectionResponse(BaseModel):
    success: bool
    raw_response: str
    boxes: List[DetectionBox]
    count: int
    image_size: Dict[str, int]
    model_used: str
    error: str = ""


class PipelineResponse(BaseModel):
    success: bool
    detection: DetectionResponse
    segmentation: Dict[str, Any]
    error: str = ""


def get_available_models() -> List[str]:
    """Get list of available models from Ollama"""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        if response.status_code == 200:
            data = response.json()
            models = [model["name"] for model in data.get("models", [])]
            return models
        return []
    except Exception as e:
        logger.error(f"Failed to get models: {e}")
        return []


def find_qwen_model(requested_model: str) -> str:
    """Find best matching Qwen model"""
    available_models = get_available_models()
    
    # Try exact match first
    if requested_model in available_models:
        return requested_model
    
    # Try to find any Qwen VL model
    qwen_models = [m for m in available_models if "qwen" in m.lower() and "vl" in m.lower()]
    
    if qwen_models:
        logger.warning(f"Model {requested_model} not found. Using {qwen_models[0]} instead.")
        return qwen_models[0]
    
    # Try to find any Qwen model
    qwen_models = [m for m in available_models if "qwen" in m.lower()]
    
    if qwen_models:
        logger.warning(f"Model {requested_model} not found. Using {qwen_models[0]} instead.")
        return qwen_models[0]
    
    # Return requested model anyway (will fail later with clear error)
    logger.error(f"No Qwen models found. Available: {available_models}")
    return requested_model


def extract_json_from_markdown(text: str) -> str:
    """Extract JSON from markdown code blocks"""
    # Try to find JSON in markdown code blocks
    patterns = [
        r"```json\s*(.*?)\s*```",  # ```json ... ```
        r"```\s*(.*?)\s*```",       # ``` ... ```
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    # Return original text if no markdown blocks found
    return text.strip()


def parse_qwen_response(response_text: str, image_width: int, image_height: int) -> List[DetectionBox]:
    """
    Parse Qwen/LLaVa response to extract boxes with 4 points
    Handles both normalized (0-1) YOLO format and pixel coordinates
    """
    boxes = []
    
    try:
        # Extract JSON from markdown if present
        json_text = extract_json_from_markdown(response_text)
        
        # Try to parse as JSON
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse as JSON, trying to extract JSON objects")
            # Try to find JSON-like structures
            json_matches = re.findall(r'\{[^{}]*"box"[^{}]*\}', json_text)
            if json_matches:
                for match in json_matches:
                    try:
                        obj = json.loads(match)
                        if "box" in obj:
                            data = [obj]
                            break
                    except:
                        continue
                else:
                    raise ValueError("No valid JSON found")
            else:
                raise ValueError("No valid JSON structure found")
        
        # Handle different response formats
        if isinstance(data, dict):
            # Check for pottery/annotations format
            if "annotations" in data:
                data = data["annotations"]
            elif "box" in data or "bbox" in data:
                data = [data]
            elif "objects" in data:
                data = data["objects"]
            elif "detections" in data:
                data = data["detections"]
            else:
                for value in data.values():
                    if isinstance(value, list):
                        data = value
                        break
        
        if not isinstance(data, list):
            logger.warning(f"Unexpected data format: {type(data)}")
            return boxes
        
        # Parse each detection
        for item in data:
            if not isinstance(item, dict):
                continue
            
            # Get label
            label = item.get("label", item.get("name", item.get("class", "object")))
            
            # Get box coordinates
            box = item.get("box", item.get("bbox", item.get("bounding_box")))
            
            if not box or not isinstance(box, list):
                continue
            
            # Determine if coordinates are normalized (YOLO format) or pixels
            # Check if all values are between 0 and 1 (or slightly above 1)
            is_normalized = all(0 <= abs(val) <= 2 for val in box)
            
            if len(box) == 4:
                # Determine format by checking if values suggest center-based or corner-based
                # YOLO: [x_center, y_center, width, height] - width and height are typically < 1
                # Corners: [x_min, y_min, x_max, y_max] - x_max > x_min, y_max > y_min
                
                if is_normalized and max(box) <= 1.5:
                    # Normalized coordinates (0-1 range)
                    # Check if it's YOLO format or corners format
                    
                    # If box[2] < box[0] or box[3] < box[1], it's likely YOLO format (width/height)
                    # If box[2] > box[0] and box[3] > box[1], it's corners format
                    
                    if box[2] > box[0] and box[3] > box[1]:
                        # Normalized corners format: [x_min, y_min, x_max, y_max]
                        x_min = box[0] * image_width
                        y_min = box[1] * image_height
                        x_max = box[2] * image_width
                        y_max = box[3] * image_height
                        
                        logger.info(f"Converted normalized corners: {box} -> pixels: [{x_min:.1f}, {y_min:.1f}, {x_max:.1f}, {y_max:.1f}]")
                    else:
                        # YOLO format: [x_center, y_center, width, height]
                        x_center, y_center, w, h = box
                        
                        # Convert to pixel coordinates
                        x_center_px = x_center * image_width
                        y_center_px = y_center * image_height
                        w_px = w * image_width
                        h_px = h * image_height
                        
                        # Convert to corner coordinates
                        x_min = x_center_px - w_px / 2
                        y_min = y_center_px - h_px / 2
                        x_max = x_center_px + w_px / 2
                        y_max = y_center_px + h_px / 2
                        
                        logger.info(f"Converted YOLO format: {box} -> corners: [{x_min:.1f}, {y_min:.1f}, {x_max:.1f}, {y_max:.1f}]")
                else:
                    # Corner format in pixels: [x_min, y_min, x_max, y_max]
                    x_min, y_min, x_max, y_max = box
                    logger.info(f"Using pixel corners: [{x_min:.1f}, {y_min:.1f}, {x_max:.1f}, {y_max:.1f}]")
                
                # Create 4-point box from corners
                boxes.append(DetectionBox(
                    x1=float(x_min),
                    y1=float(y_min),
                    x2=float(x_max),
                    y2=float(y_min),
                    x3=float(x_max),
                    y3=float(y_max),
                    x4=float(x_min),
                    y4=float(y_max),
                    label=str(label)
                ))
                
            elif len(box) == 8:
                # 8-point format [x1, y1, x2, y2, x3, y3, x4, y4]
                if is_normalized:
                    # Denormalize
                    box = [
                        box[0] * image_width, box[1] * image_height,
                        box[2] * image_width, box[3] * image_height,
                        box[4] * image_width, box[5] * image_height,
                        box[6] * image_width, box[7] * image_height
                    ]
                
                boxes.append(DetectionBox(
                    x1=float(box[0]),
                    y1=float(box[1]),
                    x2=float(box[2]),
                    y2=float(box[3]),
                    x3=float(box[4]),
                    y3=float(box[5]),
                    x4=float(box[6]),
                    y4=float(box[7]),
                    label=str(label)
                ))
        
    except Exception as e:
        logger.error(f"Error parsing response: {e}", exc_info=True)
    
    return boxes


def call_ollama_vision(image_base64: str, prompt: str, model: str) -> tuple[str, str]:
    """Call Ollama vision model"""
    try:
        # Find available model
        model_to_use = find_qwen_model(model)
        
        payload = {
            "model": model_to_use,
            "prompt": prompt,
            "images": [image_base64],
            "stream": False
        }
        
        logger.info(f"Calling Ollama with model: {model_to_use}")
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            timeout=300
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Ollama error: {response.text}"
            )
        
        result = response.json()
        return result.get("response", ""), model_to_use
        
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Ollama request timed out")
    except Exception as e:
        logger.error(f"Ollama error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ollama error: {str(e)}")


def call_sam_service(image_base64: str, boxes: List[DetectionBox]) -> Dict[str, Any]:
    """Call SAM service for segmentation"""
    try:
        payload = {
            "image_base64": image_base64,
            "boxes": [box.dict() for box in boxes]
        }
        
        logger.info(f"Calling SAM service with {len(boxes)} boxes")
        response = requests.post(
            f"{SAM_URL}/segment",
            json=payload,
            timeout=120
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"SAM error: {response.text}"
            )
        
        return response.json()
        
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="SAM request timed out")
    except Exception as e:
        logger.error(f"SAM error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"SAM error: {str(e)}")


@app.get("/health")
async def health():
    """Health check for all services"""
    services = {}
    
    # Check Ollama
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        services["ollama"] = {
            "status": "healthy" if response.status_code == 200 else "unhealthy",
            "models": len(response.json().get("models", [])) if response.status_code == 200 else 0
        }
    except Exception as e:
        services["ollama"] = {"status": "unhealthy", "error": str(e)}
    
    # Check SAM
    try:
        response = requests.get(f"{SAM_URL}/health", timeout=5)
        services["sam"] = response.json() if response.status_code == 200 else {"status": "unhealthy"}
    except Exception as e:
        services["sam"] = {"status": "unhealthy", "error": str(e)}
    
    all_healthy = all(
        s.get("status") == "healthy" 
        for s in services.values()
    )
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "services": services
    }


@app.post("/detect", response_model=DetectionResponse)
async def detect(request: DetectionRequest):
    """
    Detect objects using Qwen VL model
    
    Returns boxes with 4 points in pixel coordinates
    """
    try:
        # Decode image to get dimensions
        import base64
        from PIL import Image
        import io
        
        image_bytes = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        image_width, image_height = image.size
        
        # Call Ollama
        model = request.model or DEFAULT_MODEL
        raw_response, model_used = call_ollama_vision(
            request.image_base64,
            request.prompt,
            model
        )
        
        # Parse response
        boxes = parse_qwen_response(raw_response, image_width, image_height)
        
        return DetectionResponse(
            success=True,
            raw_response=raw_response,
            boxes=boxes,
            count=len(boxes),
            image_size={"width": image_width, "height": image_height},
            model_used=model_used
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection error: {e}", exc_info=True)
        return DetectionResponse(
            success=False,
            raw_response="",
            boxes=[],
            count=0,
            image_size={"width": 0, "height": 0},
            model_used="",
            error=str(e)
        )


@app.post("/pipeline", response_model=PipelineResponse)
async def pipeline(request: DetectionRequest):
    """
    Full pipeline: detection + segmentation
    """
    try:
        # Step 1: Detection
        detection_result = await detect(request)
        
        if not detection_result.success or detection_result.count == 0:
            return PipelineResponse(
                success=False,
                detection=detection_result,
                segmentation={},
                error="No objects detected"
            )
        
        # Step 2: Segmentation
        segmentation_result = call_sam_service(
            request.image_base64,
            detection_result.boxes
        )
        
        return PipelineResponse(
            success=True,
            detection=detection_result,
            segmentation=segmentation_result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        return PipelineResponse(
            success=False,
            detection=DetectionResponse(
                success=False,
                raw_response="",
                boxes=[],
                count=0,
                image_size={"width": 0, "height": 0},
                model_used="",
                error=str(e)
            ),
            segmentation={},
            error=str(e)
        )


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Vision AI Pipeline API",
        "status": "running",
        "endpoints": ["/health", "/detect", "/pipeline"],
        "ollama_url": OLLAMA_URL,
        "sam_url": SAM_URL,
        "default_model": DEFAULT_MODEL
    }
