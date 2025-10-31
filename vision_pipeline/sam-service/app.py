import os
import base64
import io
import logging
from typing import List, Dict, Any
import numpy as np
import cv2
import torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import wget

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SAM Segmentation Service")

# Global SAM model
sam_model = None
sam_predictor = None

# SAM model URLs
SAM_MODELS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}


class Box4Point(BaseModel):
    """Bounding box with 4 points [x1, y1, x2, y2, x3, y3, x4, y4]"""
    x1: float
    y1: float
    x2: float
    y2: float
    x3: float
    y3: float
    x4: float
    y4: float
    label: str = ""


class SegmentRequest(BaseModel):
    image_base64: str
    boxes: List[Box4Point]


class SegmentResponse(BaseModel):
    success: bool
    results: List[Dict[str, Any]]
    error: str = ""


def download_sam_model(model_type: str = "vit_h") -> str:
    """Download SAM model if not present"""
    model_path = f"/models/sam_{model_type}.pth"
    
    if os.path.exists(model_path):
        logger.info(f"SAM model already exists at {model_path}")
        return model_path
    
    if model_type not in SAM_MODELS:
        raise ValueError(f"Unknown model type: {model_type}")
    
    logger.info(f"Downloading SAM model {model_type}...")
    url = SAM_MODELS[model_type]
    
    try:
        wget.download(url, model_path)
        logger.info(f"\nSAM model downloaded to {model_path}")
        return model_path
    except Exception as e:
        logger.error(f"Failed to download SAM model: {e}")
        raise


def initialize_sam():
    """Initialize SAM model"""
    global sam_model, sam_predictor
    
    model_type = os.getenv("SAM_MODEL_TYPE", "vit_h")
    force_cpu = os.getenv("FORCE_CPU", "false").lower() == "true"
    
    logger.info(f"Initializing SAM with model type: {model_type}")
    
    # Download model if needed
    model_path = download_sam_model(model_type)
    
    # Import SAM
    from segment_anything import sam_model_registry, SamPredictor
    
    # Determine device
    if force_cpu:
        device = "cpu"
        logger.info("Using CPU (forced)")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
    
    # Load model
    sam_model = sam_model_registry[model_type](checkpoint=model_path)
    sam_model.to(device=device)
    sam_predictor = SamPredictor(sam_model)
    
    logger.info("SAM model initialized successfully")


def box4_to_xyxy(box: Box4Point) -> List[float]:
    """Convert 4-point box to [x_min, y_min, x_max, y_max]"""
    xs = [box.x1, box.x2, box.x3, box.x4]
    ys = [box.y1, box.y2, box.y3, box.y4]
    return [min(xs), min(ys), max(xs), max(ys)]


def mask_to_polygon(mask: np.ndarray) -> List[List[List[float]]]:
    """Convert binary mask to polygon coordinates"""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    polygons = []
    for contour in contours:
        if len(contour) >= 3:  # Valid polygon needs at least 3 points
            polygon = contour.reshape(-1, 2).tolist()
            polygons.append(polygon)
    
    return polygons


def mask_to_base64(mask: np.ndarray) -> str:
    """Convert binary mask to base64 PNG"""
    mask_img = (mask * 255).astype(np.uint8)
    pil_img = Image.fromarray(mask_img)
    
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    buffer.seek(0)
    
    return base64.b64encode(buffer.read()).decode()


@app.on_event("startup")
async def startup_event():
    """Initialize SAM on startup"""
    try:
        initialize_sam()
    except Exception as e:
        logger.error(f"Failed to initialize SAM: {e}")
        raise


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": sam_model is not None,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }


@app.post("/segment", response_model=SegmentResponse)
async def segment(request: SegmentRequest):
    """
    Segment objects in image using SAM with 4-point boxes
    
    Args:
        request: Contains base64 image and list of 4-point boxes
        
    Returns:
        Segmentation results with masks, polygons, and metadata
    """
    try:
        if sam_predictor is None:
            raise HTTPException(status_code=503, detail="SAM model not initialized")
        
        # Decode image
        image_bytes = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image.convert("RGB"))
        
        # Set image for SAM
        sam_predictor.set_image(image_np)
        
        results = []
        
        for idx, box in enumerate(request.boxes):
            # Convert 4-point box to xyxy format
            box_xyxy = box4_to_xyxy(box)
            box_array = np.array(box_xyxy)
            
            # Predict mask
            masks, scores, _ = sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box_array[None, :],
                multimask_output=False
            )
            
            # Get best mask
            mask = masks[0]
            confidence = float(scores[0])
            
            # Convert mask to polygon
            polygons = mask_to_polygon(mask)
            
            # Convert mask to base64
            mask_base64 = mask_to_base64(mask)
            
            # Calculate area
            area = int(np.sum(mask))
            
            result = {
                "id": idx,
                "label": box.label,
                "confidence": confidence,
                "box": {
                    "x1": box.x1,
                    "y1": box.y1,
                    "x2": box.x2,
                    "y2": box.y2,
                    "x3": box.x3,
                    "y3": box.y3,
                    "x4": box.x4,
                    "y4": box.y4,
                },
                "mask_base64": mask_base64,
                "polygons": polygons,
                "area": area
            }
            
            results.append(result)
        
        return SegmentResponse(
            success=True,
            results=results
        )
        
    except Exception as e:
        logger.error(f"Segmentation error: {e}", exc_info=True)
        return SegmentResponse(
            success=False,
            results=[],
            error=str(e)
        )


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "SAM Segmentation Service",
        "status": "running",
        "endpoints": ["/health", "/segment"]
    }
