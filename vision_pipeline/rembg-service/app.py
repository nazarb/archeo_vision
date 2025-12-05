import os
import base64
import io
import logging
from typing import Optional, Tuple, List
from enum import Enum

import torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rembg import remove, new_session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Background Removal Service")

# Global rembg sessions cache
rembg_sessions = {}


class ModelName(str, Enum):
    """Available rembg models"""
    u2net = "u2net"
    u2netp = "u2netp"
    u2net_human_seg = "u2net_human_seg"
    u2net_cloth_seg = "u2net_cloth_seg"
    silueta = "silueta"
    isnet_general_use = "isnet-general-use"
    isnet_anime = "isnet-anime"
    sam = "sam"


class RemoveBackgroundRequest(BaseModel):
    """Request model for background removal"""
    image_base64: str
    model: ModelName = ModelName.u2net
    alpha_matting: bool = False
    alpha_matting_foreground_threshold: int = 240
    alpha_matting_background_threshold: int = 10
    alpha_matting_erode_size: int = 10
    bgcolor: Optional[Tuple[int, int, int, int]] = None  # RGBA, None = transparent


class RemoveBackgroundResponse(BaseModel):
    """Response model for background removal"""
    success: bool
    image_base64: str = ""
    original_size: Tuple[int, int] = (0, 0)
    error: str = ""


class BatchRemoveRequest(BaseModel):
    """Request model for batch background removal"""
    images_base64: List[str]
    model: ModelName = ModelName.u2net
    alpha_matting: bool = False
    bgcolor: Optional[Tuple[int, int, int, int]] = None


class BatchRemoveResponse(BaseModel):
    """Response model for batch background removal"""
    success: bool
    images_base64: List[str] = []
    errors: List[str] = []


def get_session(model_name: str):
    """Get or create a rembg session for the specified model"""
    if model_name not in rembg_sessions:
        logger.info(f"Creating new session for model: {model_name}")
        rembg_sessions[model_name] = new_session(model_name)
    return rembg_sessions[model_name]


def process_image(
    image_base64: str,
    model_name: str,
    alpha_matting: bool = False,
    alpha_matting_foreground_threshold: int = 240,
    alpha_matting_background_threshold: int = 10,
    alpha_matting_erode_size: int = 10,
    bgcolor: Optional[Tuple[int, int, int, int]] = None
) -> Tuple[str, Tuple[int, int]]:
    """
    Process a single image and remove its background.

    Returns:
        Tuple of (base64_result, (width, height))
    """
    # Decode input image
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_bytes))
    original_size = image.size

    # Convert to RGB if necessary
    if image.mode not in ('RGB', 'RGBA'):
        image = image.convert('RGB')

    # Get session for the model
    session = get_session(model_name)

    # Remove background
    result = remove(
        image,
        session=session,
        alpha_matting=alpha_matting,
        alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
        alpha_matting_background_threshold=alpha_matting_background_threshold,
        alpha_matting_erode_size=alpha_matting_erode_size,
        bgcolor=bgcolor
    )

    # Encode result to base64 PNG
    buffer = io.BytesIO()
    result.save(buffer, format="PNG")
    buffer.seek(0)
    result_base64 = base64.b64encode(buffer.read()).decode()

    return result_base64, original_size


@app.on_event("startup")
async def startup_event():
    """Initialize default model on startup"""
    try:
        logger.info("Initializing default U2Net model...")
        get_session("u2net")

        # Log GPU availability
        if torch.cuda.is_available():
            logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA version: {torch.version.cuda}")
        else:
            logger.info("Running on CPU")

        logger.info("Background removal service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        raise


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "loaded_models": list(rembg_sessions.keys())
    }


@app.post("/remove", response_model=RemoveBackgroundResponse)
async def remove_background(request: RemoveBackgroundRequest):
    """
    Remove background from a single image.

    Args:
        request: Contains base64 image and processing options

    Returns:
        Base64 PNG image with background removed
    """
    try:
        result_base64, original_size = process_image(
            image_base64=request.image_base64,
            model_name=request.model.value,
            alpha_matting=request.alpha_matting,
            alpha_matting_foreground_threshold=request.alpha_matting_foreground_threshold,
            alpha_matting_background_threshold=request.alpha_matting_background_threshold,
            alpha_matting_erode_size=request.alpha_matting_erode_size,
            bgcolor=request.bgcolor
        )

        return RemoveBackgroundResponse(
            success=True,
            image_base64=result_base64,
            original_size=original_size
        )

    except Exception as e:
        logger.error(f"Background removal error: {e}", exc_info=True)
        return RemoveBackgroundResponse(
            success=False,
            error=str(e)
        )


@app.post("/remove/batch", response_model=BatchRemoveResponse)
async def remove_background_batch(request: BatchRemoveRequest):
    """
    Remove background from multiple images.

    Args:
        request: Contains list of base64 images and processing options

    Returns:
        List of base64 PNG images with backgrounds removed
    """
    results = []
    errors = []

    for idx, image_base64 in enumerate(request.images_base64):
        try:
            result_base64, _ = process_image(
                image_base64=image_base64,
                model_name=request.model.value,
                alpha_matting=request.alpha_matting,
                bgcolor=request.bgcolor
            )
            results.append(result_base64)
            errors.append("")
        except Exception as e:
            logger.error(f"Error processing image {idx}: {e}")
            results.append("")
            errors.append(str(e))

    return BatchRemoveResponse(
        success=all(e == "" for e in errors),
        images_base64=results,
        errors=errors
    )


@app.get("/models")
async def list_models():
    """List available background removal models"""
    return {
        "models": [
            {
                "name": "u2net",
                "description": "General purpose model, good balance of speed and quality"
            },
            {
                "name": "u2netp",
                "description": "Lightweight version of U2Net, faster but less accurate"
            },
            {
                "name": "u2net_human_seg",
                "description": "Optimized for human segmentation"
            },
            {
                "name": "u2net_cloth_seg",
                "description": "Optimized for clothing segmentation"
            },
            {
                "name": "silueta",
                "description": "Fast model optimized for silhouette extraction"
            },
            {
                "name": "isnet-general-use",
                "description": "High quality general purpose model"
            },
            {
                "name": "isnet-anime",
                "description": "Optimized for anime/cartoon images"
            },
            {
                "name": "sam",
                "description": "Segment Anything Model - high quality but slower"
            }
        ],
        "default": "u2net"
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Background Removal Service",
        "status": "running",
        "gpu_enabled": torch.cuda.is_available(),
        "endpoints": ["/health", "/remove", "/remove/batch", "/models"]
    }
