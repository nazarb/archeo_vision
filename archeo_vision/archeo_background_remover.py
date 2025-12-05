#!/usr/bin/env python3
"""
Archaeological Background Remover
Removes backgrounds from archaeological artifact images for cleaner analysis and documentation

Supports both local processing (rembg/grabcut) and GPU-accelerated Docker service.
Includes scale bar overlay for archaeological documentation.
Uses Ollama vision models to automatically detect scale rulers and measure artifacts.
"""

import os
import io
import re
import json
import base64
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests

try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ArtifactMeasurement:
    """Measurement results from Ollama vision analysis"""
    success: bool = False
    # Scale bar detection
    scale_detected: bool = False
    scale_bar_length_cm: float = 0.0  # Length of the scale bar in cm
    # Image measurements in cm (from LLM using scale bar)
    image_longest_side_cm: float = 0.0  # Longest side of the image in cm
    # Artifact measurements in cm
    artifact_length_cm: float = 0.0
    artifact_width_cm: float = 0.0
    # Calculated calibration (computed outside LLM)
    pixels_per_cm: float = 0.0
    # Raw response for debugging
    raw_response: str = ""
    error: str = ""


@dataclass
class ScaleBarConfig:
    """Configuration for scale bar overlay"""
    enabled: bool = False
    length_cm: float = 5.0  # Length in centimeters
    pixels_per_cm: float = 100.0  # Calibration: pixels per centimeter
    position: str = "bottom-right"  # Position: bottom-left, bottom-right, top-left, top-right
    color: Tuple[int, int, int, int] = (0, 0, 0, 255)  # RGBA color for bar
    text_color: Tuple[int, int, int, int] = (0, 0, 0, 255)  # RGBA color for text
    bar_height: int = 10  # Height of the scale bar in pixels
    margin: int = 20  # Margin from image edge
    show_text: bool = True  # Show measurement text
    font_size: int = 24  # Font size for text
    auto_calibrate: bool = False  # Auto-detect scale from image using Ollama
    show_artifact_dimensions: bool = False  # Show artifact L x W on image


class ArcheoBackgroundRemover:
    """
    Removes backgrounds from archaeological artifact images.

    Supports multiple background removal methods:
    - api: Use GPU-accelerated Docker service (recommended for batch processing)
    - rembg: Local AI-based removal using U2Net
    - grabcut: OpenCV GrabCut algorithm (fallback)
    """

    SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']

    def __init__(
        self,
        shared_path: str = "./archeo-shared",
        method: str = "auto",
        model_name: str = "u2net",
        alpha_matting: bool = False,
        background_color: Tuple[int, int, int, int] = (0, 0, 0, 0),
        api_url: str = "http://localhost:8002",
        scale_bar: Optional[ScaleBarConfig] = None,
        pipeline_url: str = "http://localhost:8080",
        vision_model: str = "gemma3:27b"
    ):
        """
        Initialize the background remover.

        Args:
            shared_path: Path to shared directory containing images
            method: Background removal method ('api', 'rembg', 'grabcut', or 'auto')
            model_name: Model to use (u2net, u2netp, u2net_human_seg, etc.)
            alpha_matting: Enable alpha matting for better edges (slower)
            background_color: RGBA color for background (default: transparent)
            api_url: URL of the rembg Docker service
            scale_bar: Scale bar configuration (None to disable)
            pipeline_url: URL of the vision pipeline API for Ollama (for auto-calibration)
            vision_model: Ollama vision model for measurements (default: gemma3:27b)
        """
        self.shared_path = Path(shared_path)
        self.method = method
        self.model_name = model_name
        self.alpha_matting = alpha_matting
        self.background_color = background_color
        self.api_url = api_url
        self.scale_bar = scale_bar or ScaleBarConfig()
        self.pipeline_url = pipeline_url
        self.vision_model = vision_model

        # Setup directories
        self.images_dir = self.shared_path / "final"
        self.output_dir = self.shared_path / "no_background"
        self.measurements_dir = self.shared_path / "measurements"

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.scale_bar.auto_calibrate:
            self.measurements_dir.mkdir(parents=True, exist_ok=True)

        # Initialize rembg session if needed for local processing
        self.rembg_session = None
        self._api_available = None

        # Determine actual method to use
        self._resolved_method = self._resolve_method()

        if self._resolved_method == "rembg":
            self._init_rembg_session()

    def _check_api_available(self) -> bool:
        """Check if the rembg API service is available"""
        if self._api_available is not None:
            return self._api_available

        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            self._api_available = response.status_code == 200
            if self._api_available:
                health = response.json()
                logger.info(f"API service available (GPU: {health.get('gpu_available', False)})")
        except requests.exceptions.RequestException:
            self._api_available = False

        return self._api_available

    def _resolve_method(self) -> str:
        """Resolve the actual method to use based on availability"""
        if self.method == "grabcut":
            return "grabcut"

        if self.method == "api":
            if self._check_api_available():
                return "api"
            logger.warning("API service not available, falling back to local processing")

        if self.method == "auto":
            # Try API first, then local rembg, then grabcut
            if self._check_api_available():
                return "api"
            if REMBG_AVAILABLE:
                return "rembg"
            return "grabcut"

        if self.method == "rembg":
            if REMBG_AVAILABLE:
                return "rembg"
            logger.error("rembg requested but not installed. Install with: pip install rembg")
            return "grabcut"

        return "grabcut"

    def _init_rembg_session(self):
        """Initialize rembg session for faster batch processing"""
        try:
            self.rembg_session = new_session(self.model_name)
            logger.info(f"Initialized rembg session with model: {self.model_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize rembg session: {e}")
            self.rembg_session = None

    def encode_image(self, image: Image.Image) -> str:
        """Encode PIL Image to base64 string"""
        buffer = io.BytesIO()
        # Save as PNG to preserve quality
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode()

    def decode_image(self, image_base64: str) -> Image.Image:
        """Decode base64 string to PIL Image"""
        image_bytes = base64.b64decode(image_base64)
        return Image.open(io.BytesIO(image_bytes))

    def encode_image_file(self, image_path: Path) -> str:
        """Encode image file to base64 string"""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode()

    def measure_artifact(self, image_path: Path) -> ArtifactMeasurement:
        """
        Use Ollama vision model to detect scale bar and measure artifact in centimeters.

        The LLM only provides measurements in cm using the visible scale bar.
        We then calculate pixels_per_cm by comparing the LLM's longest_side_cm
        with the actual image dimensions in pixels.

        Args:
            image_path: Path to the image file

        Returns:
            ArtifactMeasurement with cm measurements and calculated pixels_per_cm
        """
        logger.info(f"Measuring artifact in: {image_path.name}")

        # Load image to get dimensions
        image = cv2.imread(str(image_path))
        if image is None:
            return ArtifactMeasurement(
                success=False,
                error=f"Failed to load image: {image_path}"
            )

        image_height, image_width = image.shape[:2]
        longest_side_pixels = max(image_width, image_height)

        # Encode image for API
        image_base64 = self.encode_image_file(image_path)

        # Simplified measurement prompt - only ask for CM values, not pixels
        measurement_prompt = """You are an expert archaeological measurement system. Be EXTREMELY PRECISE.

## TASK: Measure this image using the visible scale bar

### Step 1: FIND AND READ THE SCALE BAR CAREFULLY
Look for the scale bar/ruler on the LEFT, BOTTOM, or RIGHT side of the artifact.
- Count the EXACT centimeter markings visible on the scale bar
- Look for millimeter subdivisions between centimeter marks
- The scale bar may NOT be exactly 10cm - it could be 9cm, 9.5cm, 8cm, 12cm, etc.
- Report the PRECISE length to one decimal place (e.g., 9.3 cm, not "about 10 cm")
- Do NOT assume the scale is 10cm - READ the actual markings!

### Step 2: MEASURE THE IMAGE
Using the scale bar as your reference, measure:
- The LONGEST SIDE of the entire image in centimeters
- This is the full width OR height of the photo (whichever is larger)
- Be precise to one decimal place

### Step 3: MEASURE THE ARTIFACT
Using the scale bar, measure the archaeological artifact:
- LENGTH: The longest dimension of the artifact in centimeters
- WIDTH: The perpendicular dimension of the artifact in centimeters
- Be precise to one decimal place

## OUTPUT FORMAT (JSON only, no other text)
```json
{
  "scale_bar": {
    "detected": true,
    "length_cm": 9.3,
    "position": "left",
    "description": "Scale bar showing 9cm plus 3mm markings"
  },
  "image": {
    "longest_side_cm": 24.8,
    "orientation": "horizontal"
  },
  "artifact": {
    "detected": true,
    "length_cm": 11.5,
    "width_cm": 7.2,
    "description": "ceramic pottery fragment"
  }
}
```

CRITICAL PRECISION RULES:
- READ the actual scale bar markings - do NOT assume 10cm or any round number
- Count centimeter lines AND millimeter subdivisions carefully
- Report all measurements to ONE DECIMAL PLACE (e.g., 9.3 cm, 24.8 cm)
- If scale shows only millimeters, convert precisely (93mm = 9.3cm)
- Set "detected": false if you cannot find the scale bar"""

        # Call the pipeline API
        try:
            payload = {
                "image_base64": image_base64,
                "prompt": measurement_prompt,
                "model": self.vision_model
            }

            response = requests.post(
                f"{self.pipeline_url}/detect",
                json=payload,
                timeout=300
            )
            response.raise_for_status()
            result = response.json()

            if not result.get("success"):
                return ArtifactMeasurement(
                    success=False,
                    error=result.get("error", "Detection failed"),
                    raw_response=result.get("raw_response", "")
                )

            # Parse the response and calculate pixels_per_cm
            raw_response = result.get("raw_response", "")
            return self._parse_measurement_response(raw_response, longest_side_pixels)

        except requests.exceptions.RequestException as e:
            logger.error(f"Measurement API request failed: {e}")
            return ArtifactMeasurement(
                success=False,
                error=f"API request failed: {e}"
            )

    def _parse_measurement_response(
        self,
        raw_response: str,
        longest_side_pixels: int
    ) -> ArtifactMeasurement:
        """
        Parse the Ollama response and calculate pixels_per_cm.

        The LLM provides measurements in cm only. We calculate pixels_per_cm by:
        pixels_per_cm = longest_side_pixels / longest_side_cm
        """
        measurement = ArtifactMeasurement(raw_response=raw_response)

        try:
            # Extract JSON from markdown code blocks if present
            json_match = re.search(r'```json\s*(.*?)\s*```', raw_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON object directly
                json_match = re.search(r'\{[\s\S]*\}', raw_response)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    measurement.error = "No JSON found in response"
                    return measurement

            data = json.loads(json_str)

            # Parse scale bar data
            scale_data = data.get("scale_bar", {})
            if scale_data.get("detected", False):
                measurement.scale_detected = True
                measurement.scale_bar_length_cm = float(scale_data.get("length_cm", 0))

            # Parse image measurements
            image_data = data.get("image", {})
            measurement.image_longest_side_cm = float(image_data.get("longest_side_cm", 0))

            # Calculate pixels_per_cm using actual image dimensions
            if measurement.image_longest_side_cm > 0:
                measurement.pixels_per_cm = longest_side_pixels / measurement.image_longest_side_cm
                logger.info(f"  Image longest side: {measurement.image_longest_side_cm:.1f} cm = {longest_side_pixels} px")
                logger.info(f"  Calculated: {measurement.pixels_per_cm:.1f} pixels/cm")
            else:
                measurement.error = "LLM did not provide image longest side measurement"
                return measurement

            # Parse artifact data (already in cm from LLM)
            artifact_data = data.get("artifact", {})
            if artifact_data.get("detected", False):
                measurement.artifact_length_cm = float(artifact_data.get("length_cm", 0))
                measurement.artifact_width_cm = float(artifact_data.get("width_cm", 0))
                logger.info(f"  Artifact: {measurement.artifact_length_cm:.1f} x {measurement.artifact_width_cm:.1f} cm")

            measurement.success = measurement.scale_detected and measurement.pixels_per_cm > 0

        except json.JSONDecodeError as e:
            measurement.error = f"JSON parse error: {e}"
            logger.error(f"Failed to parse measurement JSON: {e}")
        except Exception as e:
            measurement.error = f"Parse error: {e}"
            logger.error(f"Failed to parse measurement response: {e}")

        return measurement

    def remove_background_api(self, image: Image.Image) -> Image.Image:
        """
        Remove background using the Docker API service.

        Args:
            image: PIL Image to process

        Returns:
            PIL Image with transparent background
        """
        # Encode image
        image_base64 = self.encode_image(image)

        # Prepare request payload
        payload = {
            "image_base64": image_base64,
            "model": self.model_name,
            "alpha_matting": self.alpha_matting,
            "bgcolor": self.background_color if self.background_color != (0, 0, 0, 0) else None
        }

        # Send request
        try:
            response = requests.post(
                f"{self.api_url}/remove",
                json=payload,
                timeout=300  # 5 minute timeout for large images
            )
            response.raise_for_status()
            result = response.json()

            if not result.get("success"):
                raise RuntimeError(f"API error: {result.get('error', 'Unknown error')}")

            # Decode result
            return self.decode_image(result["image_base64"])

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API request failed: {e}")

    def remove_background_rembg(self, image: Image.Image) -> Image.Image:
        """
        Remove background using rembg (U2Net model).

        Args:
            image: PIL Image to process

        Returns:
            PIL Image with transparent background
        """
        if not REMBG_AVAILABLE:
            raise RuntimeError("rembg is not installed")

        # Apply background removal
        result = remove(
            image,
            session=self.rembg_session,
            alpha_matting=self.alpha_matting,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=10,
            bgcolor=self.background_color
        )

        return result

    def remove_background_grabcut(self, image: Image.Image, iterations: int = 5) -> Image.Image:
        """
        Remove background using OpenCV GrabCut algorithm.

        This is a fallback method when rembg is not available.
        Uses automatic rectangle detection based on image edges.

        Args:
            image: PIL Image to process
            iterations: Number of GrabCut iterations

        Returns:
            PIL Image with transparent background
        """
        # Convert PIL to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Create initial mask
        mask = np.zeros(cv_image.shape[:2], np.uint8)

        # Background and foreground models
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        # Auto-detect rectangle (exclude border pixels)
        height, width = cv_image.shape[:2]
        margin = int(min(width, height) * 0.02)  # 2% margin
        rect = (margin, margin, width - 2 * margin, height - 2 * margin)

        # Apply GrabCut
        try:
            cv2.grabCut(cv_image, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)
        except cv2.error as e:
            logger.warning(f"GrabCut failed: {e}, returning original image")
            return image.convert('RGBA')

        # Create binary mask (foreground = 1 or 3)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

        # Apply mask to create RGBA image
        result = cv_image * mask2[:, :, np.newaxis]

        # Convert to RGBA with alpha channel
        result_rgba = cv2.cvtColor(result, cv2.COLOR_BGR2RGBA)
        result_rgba[:, :, 3] = mask2 * 255

        # Convert back to PIL
        return Image.fromarray(result_rgba)

    def remove_background(self, image: Image.Image) -> Image.Image:
        """
        Remove background using the configured method.

        Args:
            image: PIL Image to process

        Returns:
            PIL Image with background removed
        """
        if self._resolved_method == "api":
            logger.debug("Using API service for background removal")
            return self.remove_background_api(image)
        elif self._resolved_method == "rembg":
            logger.debug("Using local rembg for background removal")
            return self.remove_background_rembg(image)
        else:
            logger.debug("Using GrabCut for background removal")
            return self.remove_background_grabcut(image)

    def add_scale_bar(
        self,
        image: Image.Image,
        measurement: Optional[ArtifactMeasurement] = None
    ) -> Image.Image:
        """
        Add a scale bar overlay to the image, optionally with artifact dimensions.

        The scale bar is drawn based on the ScaleBarConfig settings.
        Uses pixels_per_cm calibration to convert real-world measurements to pixels.

        Args:
            image: PIL Image to add scale bar to
            measurement: Optional ArtifactMeasurement with artifact dimensions

        Returns:
            PIL Image with scale bar overlay
        """
        if not self.scale_bar.enabled:
            return image

        # Ensure image is RGBA for transparency support
        if image.mode != 'RGBA':
            image = image.convert('RGBA')

        # Create a copy to draw on
        result = image.copy()
        draw = ImageDraw.Draw(result)

        # Calculate scale bar dimensions
        bar_length_px = int(self.scale_bar.length_cm * self.scale_bar.pixels_per_cm)
        bar_height = self.scale_bar.bar_height
        margin = self.scale_bar.margin

        img_width, img_height = image.size

        # Try to load a font, fall back to default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                                      self.scale_bar.font_size)
        except (IOError, OSError):
            try:
                font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans.ttf",
                                          self.scale_bar.font_size)
            except (IOError, OSError):
                try:
                    font = ImageFont.truetype("arial.ttf", self.scale_bar.font_size)
                except (IOError, OSError):
                    font = ImageFont.load_default()

        # Determine position
        position = self.scale_bar.position.lower()

        if position == "bottom-right":
            x_start = img_width - margin - bar_length_px
            y_start = img_height - margin - bar_height
        elif position == "bottom-left":
            x_start = margin
            y_start = img_height - margin - bar_height
        elif position == "top-right":
            x_start = img_width - margin - bar_length_px
            y_start = margin
        elif position == "top-left":
            x_start = margin
            y_start = margin
        else:
            # Default to bottom-right
            x_start = img_width - margin - bar_length_px
            y_start = img_height - margin - bar_height

        x_end = x_start + bar_length_px
        y_end = y_start + bar_height

        # Draw the scale bar (main rectangle)
        draw.rectangle(
            [x_start, y_start, x_end, y_end],
            fill=self.scale_bar.color
        )

        # Draw end caps (vertical lines at each end)
        cap_height = bar_height * 2
        cap_y_start = y_start - (cap_height - bar_height) // 2
        cap_y_end = cap_y_start + cap_height

        # Left cap
        draw.rectangle(
            [x_start, cap_y_start, x_start + 2, cap_y_end],
            fill=self.scale_bar.color
        )

        # Right cap
        draw.rectangle(
            [x_end - 2, cap_y_start, x_end, cap_y_end],
            fill=self.scale_bar.color
        )

        # Add text label if enabled
        if self.scale_bar.show_text:
            # Format the measurement text
            if self.scale_bar.length_cm >= 1:
                text = f"{self.scale_bar.length_cm:.0f} cm"
            else:
                text = f"{self.scale_bar.length_cm * 10:.0f} mm"

            # Get text bounding box
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # Position text centered above or below the bar
            text_x = x_start + (bar_length_px - text_width) // 2

            if position.startswith("bottom"):
                text_y = y_start - text_height - 5
            else:
                text_y = y_end + 5

            # Draw text background for readability (semi-transparent white)
            padding = 3
            draw.rectangle(
                [text_x - padding, text_y - padding,
                 text_x + text_width + padding, text_y + text_height + padding],
                fill=(255, 255, 255, 200)
            )

            # Draw text
            draw.text((text_x, text_y), text, fill=self.scale_bar.text_color, font=font)

        # Add artifact dimensions if measurement is provided and enabled
        if (self.scale_bar.show_artifact_dimensions and measurement and
                measurement.artifact_length_cm > 0):
            dim_text = f"L: {measurement.artifact_length_cm:.1f} cm x W: {measurement.artifact_width_cm:.1f} cm"

            dim_bbox = draw.textbbox((0, 0), dim_text, font=font)
            dim_width = dim_bbox[2] - dim_bbox[0]
            dim_height = dim_bbox[3] - dim_bbox[1]

            # Position artifact dimensions in the opposite corner from scale bar
            if position == "bottom-right":
                dim_x = margin
                dim_y = img_height - margin - dim_height
            elif position == "bottom-left":
                dim_x = img_width - margin - dim_width
                dim_y = img_height - margin - dim_height
            elif position == "top-right":
                dim_x = margin
                dim_y = margin
            else:  # top-left
                dim_x = img_width - margin - dim_width
                dim_y = margin

            # Draw background
            padding = 5
            draw.rectangle(
                [dim_x - padding, dim_y - padding,
                 dim_x + dim_width + padding, dim_y + dim_height + padding],
                fill=(255, 255, 255, 220)
            )

            # Draw dimension text
            draw.text((dim_x, dim_y), dim_text, fill=self.scale_bar.text_color, font=font)

        logger.debug(f"Added scale bar: {self.scale_bar.length_cm} cm at {position}")
        return result

    def process_image(self, image_path: Path) -> Optional[Path]:
        """
        Process a single image: remove background and optionally add scale bar.

        If auto_calibrate is enabled, uses Ollama to detect the scale ruler
        and measure the artifact before processing.

        Args:
            image_path: Path to input image

        Returns:
            Path to output image, or None if processing failed
        """
        logger.info(f"Processing: {image_path.name}")

        measurement = None

        try:
            # Auto-calibrate scale using Ollama if enabled
            if self.scale_bar.auto_calibrate:
                logger.info("  Auto-calibrating scale from image...")
                measurement = self.measure_artifact(image_path)

                if measurement.success and measurement.pixels_per_cm > 0:
                    # Update scale bar config with calculated calibration
                    self.scale_bar.pixels_per_cm = measurement.pixels_per_cm
                    self.scale_bar.enabled = True
                    logger.info(f"  Auto-calibrated: {measurement.pixels_per_cm:.1f} px/cm")

                    # Save measurement data
                    measurement_file = self.measurements_dir / f"{image_path.stem}_measurement.json"
                    measurement_data = {
                        "image": image_path.name,
                        "scale_detected": measurement.scale_detected,
                        "scale_bar_length_cm": measurement.scale_bar_length_cm,
                        "image_longest_side_cm": measurement.image_longest_side_cm,
                        "pixels_per_cm": measurement.pixels_per_cm,
                        "artifact_length_cm": measurement.artifact_length_cm,
                        "artifact_width_cm": measurement.artifact_width_cm
                    }
                    with open(measurement_file, 'w') as f:
                        json.dump(measurement_data, f, indent=2)
                    logger.info(f"  Saved measurements: {measurement_file.name}")
                else:
                    logger.warning(f"  Auto-calibration failed: {measurement.error}")

            # Load image
            image = Image.open(image_path)

            # Convert to RGB if necessary (handle grayscale, RGBA, etc.)
            if image.mode not in ('RGB', 'RGBA'):
                image = image.convert('RGB')

            # Get original dimensions for logging
            original_size = image.size
            logger.info(f"  Image size: {original_size[0]}x{original_size[1]}")

            # Remove background
            result = self.remove_background(image)

            # Add scale bar if enabled
            if self.scale_bar.enabled:
                result = self.add_scale_bar(result, measurement)
                logger.info(f"  Added scale bar: {self.scale_bar.length_cm} cm")

            # Determine output path (always PNG for transparency support)
            output_filename = image_path.stem + "_no_bg.png"
            output_path = self.output_dir / output_filename

            # Save result
            result.save(output_path, "PNG")
            logger.info(f"  Saved: {output_path.name}")

            return output_path

        except Exception as e:
            logger.error(f"  Failed to process {image_path.name}: {e}")
            return None

    def process_all_images(self) -> Tuple[int, int]:
        """
        Process all images in the images directory.

        Returns:
            Tuple of (success_count, total_count)
        """
        # Find all images
        image_files = []
        for ext in self.SUPPORTED_EXTENSIONS:
            image_files.extend(self.images_dir.glob(f"*{ext}"))
            image_files.extend(self.images_dir.glob(f"*{ext.upper()}"))

        # Remove duplicates and sort
        image_files = sorted(set(image_files))

        if not image_files:
            logger.warning(f"No images found in {self.images_dir}")
            return 0, 0

        logger.info(f"Found {len(image_files)} images to process")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Method: {self._resolved_method}")

        # Process each image
        success_count = 0
        for i, image_path in enumerate(image_files, 1):
            logger.info(f"\n[{i}/{len(image_files)}] Processing {image_path.name}")

            result = self.process_image(image_path)
            if result:
                success_count += 1

        return success_count, len(image_files)

    def process_single(self, input_path: str, output_path: Optional[str] = None) -> Optional[Path]:
        """
        Process a single image file.

        Args:
            input_path: Path to input image
            output_path: Optional custom output path

        Returns:
            Path to output image, or None if processing failed
        """
        input_path = Path(input_path)

        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return None

        # Process the image
        result = self.process_image(input_path)

        # Move to custom output path if specified
        if result and output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result.rename(output_path)
            logger.info(f"Moved output to: {output_path}")
            return output_path

        return result


def main():
    parser = argparse.ArgumentParser(
        description="Archaeological Background Remover - Remove backgrounds from artifact images"
    )
    parser.add_argument(
        "--shared-path",
        default="./archeo-shared",
        help="Shared folder path (default: ./archeo-shared)"
    )
    parser.add_argument(
        "--input",
        "-i",
        help="Single input image path (processes one file instead of batch)"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output path for single image processing"
    )
    parser.add_argument(
        "--method",
        choices=["auto", "api", "rembg", "grabcut"],
        default="auto",
        help="Background removal method (default: auto - tries api, then rembg, then grabcut)"
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8002",
        help="URL of rembg Docker service (default: http://localhost:8002)"
    )
    parser.add_argument(
        "--model",
        default="u2net",
        choices=["u2net", "u2netp", "u2net_human_seg", "u2net_cloth_seg", "silueta", "isnet-general-use"],
        help="Model for background removal (default: u2net)"
    )
    parser.add_argument(
        "--alpha-matting",
        action="store_true",
        help="Enable alpha matting for better edges (slower)"
    )
    parser.add_argument(
        "--bg-color",
        default="transparent",
        help="Background color: 'transparent', 'white', 'black', or R,G,B,A values"
    )

    # Scale bar arguments
    parser.add_argument(
        "--scale-bar",
        action="store_true",
        help="Add a scale bar to processed images"
    )
    parser.add_argument(
        "--scale-length",
        type=float,
        default=5.0,
        help="Scale bar length in centimeters (default: 5.0)"
    )
    parser.add_argument(
        "--scale-pixels-per-cm",
        type=float,
        default=100.0,
        help="Calibration: pixels per centimeter (default: 100.0)"
    )
    parser.add_argument(
        "--scale-position",
        choices=["bottom-right", "bottom-left", "top-right", "top-left"],
        default="bottom-right",
        help="Scale bar position (default: bottom-right)"
    )
    parser.add_argument(
        "--scale-color",
        default="black",
        help="Scale bar color: 'black', 'white', or R,G,B values (default: black)"
    )
    parser.add_argument(
        "--scale-height",
        type=int,
        default=10,
        help="Scale bar height in pixels (default: 10)"
    )
    parser.add_argument(
        "--scale-margin",
        type=int,
        default=20,
        help="Scale bar margin from edge in pixels (default: 20)"
    )
    parser.add_argument(
        "--scale-no-text",
        action="store_true",
        help="Hide the measurement text label"
    )
    parser.add_argument(
        "--scale-font-size",
        type=int,
        default=24,
        help="Font size for scale bar text (default: 24)"
    )

    # Auto-calibration arguments (using Ollama vision)
    parser.add_argument(
        "--auto-calibrate",
        action="store_true",
        help="Auto-detect scale ruler and calibrate using Ollama vision model"
    )
    parser.add_argument(
        "--show-dimensions",
        action="store_true",
        help="Show artifact dimensions (L x W) on the processed image"
    )
    parser.add_argument(
        "--pipeline-url",
        default="http://localhost:8080",
        help="URL of vision pipeline API for Ollama (default: http://localhost:8080)"
    )
    parser.add_argument(
        "--vision-model",
        default="gemma3:27b",
        help="""Ollama VISION model for measurements. Must be a vision-capable model.
                Recommended models by VRAM:
                HIGH VRAM (40GB+):
                - qwen2.5vl:72b (45GB+ VRAM, best accuracy)
                MEDIUM VRAM (16-24GB):
                - gemma3:27b (17GB VRAM, excellent) [DEFAULT]
                - mistral-small3.1 (24GB VRAM, excellent vision)
                - qwen2.5vl:32b (20GB VRAM, very good)
                - llava:34b (20GB VRAM)
                LOW VRAM (6-10GB):
                - gemma3:12b (8GB VRAM, very good)
                - minicpm-v:8b (8GB VRAM, good)
                - deepseek-vl:7b (7GB VRAM, good)
                - qwen2.5vl:7b (6GB VRAM, good)
                - gemma3:4b (4GB VRAM, basic)
                NOTE: Text-only models (ERNIE, Llama-text, etc.) will NOT work!"""
    )

    args = parser.parse_args()

    # Parse background color
    if args.bg_color == "transparent":
        bg_color = (0, 0, 0, 0)
    elif args.bg_color == "white":
        bg_color = (255, 255, 255, 255)
    elif args.bg_color == "black":
        bg_color = (0, 0, 0, 255)
    else:
        try:
            parts = [int(x) for x in args.bg_color.split(",")]
            if len(parts) == 3:
                bg_color = tuple(parts) + (255,)
            elif len(parts) == 4:
                bg_color = tuple(parts)
            else:
                raise ValueError("Invalid color format")
        except ValueError:
            logger.error(f"Invalid background color: {args.bg_color}")
            logger.error("Use 'transparent', 'white', 'black', or R,G,B or R,G,B,A values")
            return

    # Parse scale bar color
    if args.scale_color == "black":
        scale_color = (0, 0, 0, 255)
    elif args.scale_color == "white":
        scale_color = (255, 255, 255, 255)
    else:
        try:
            parts = [int(x) for x in args.scale_color.split(",")]
            if len(parts) == 3:
                scale_color = tuple(parts) + (255,)
            elif len(parts) == 4:
                scale_color = tuple(parts)
            else:
                raise ValueError("Invalid color format")
        except ValueError:
            logger.error(f"Invalid scale bar color: {args.scale_color}")
            logger.error("Use 'black', 'white', or R,G,B or R,G,B,A values")
            return

    # Create scale bar configuration
    scale_bar_config = ScaleBarConfig(
        enabled=args.scale_bar or args.auto_calibrate,
        length_cm=args.scale_length,
        pixels_per_cm=args.scale_pixels_per_cm,
        position=args.scale_position,
        color=scale_color,
        text_color=scale_color,
        bar_height=args.scale_height,
        margin=args.scale_margin,
        show_text=not args.scale_no_text,
        font_size=args.scale_font_size,
        auto_calibrate=args.auto_calibrate,
        show_artifact_dimensions=args.show_dimensions
    )

    # Create remover
    remover = ArcheoBackgroundRemover(
        shared_path=args.shared_path,
        method=args.method,
        model_name=args.model,
        alpha_matting=args.alpha_matting,
        background_color=bg_color,
        api_url=args.api_url,
        scale_bar=scale_bar_config,
        pipeline_url=args.pipeline_url,
        vision_model=args.vision_model
    )

    # Print header
    logger.info("\n" + "=" * 60)
    logger.info("Archaeological Background Remover")
    logger.info("=" * 60)

    # Show method info
    if remover._resolved_method == "api":
        logger.info(f"Using: Docker API service at {args.api_url}")
    elif remover._resolved_method == "rembg":
        logger.info("Using: Local rembg (CPU)")
    else:
        logger.info("Using: OpenCV GrabCut fallback")

    # Show scale bar info
    if scale_bar_config.auto_calibrate:
        logger.info(f"Auto-calibration: ENABLED (using {args.vision_model})")
        logger.info(f"  Pipeline URL: {args.pipeline_url}")
        if args.show_dimensions:
            logger.info("  Artifact dimensions will be displayed on images")
    elif scale_bar_config.enabled:
        logger.info(f"Scale bar: {scale_bar_config.length_cm} cm ({scale_bar_config.position})")
        logger.info(f"  Calibration: {scale_bar_config.pixels_per_cm} pixels/cm")

    # Process
    if args.input:
        # Single image mode
        logger.info(f"\nProcessing single image: {args.input}")
        result = remover.process_single(args.input, args.output)

        if result:
            logger.info(f"\nBackground removed successfully!")
            logger.info(f"Output: {result}")
        else:
            logger.error("\nFailed to process image")
    else:
        # Batch mode
        logger.info(f"\nBatch processing images from: {remover.images_dir}")

        success_count, total_count = remover.process_all_images()

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info(f"Processing complete: {success_count}/{total_count} successful")
        logger.info("=" * 60)

        if success_count > 0:
            logger.info(f"\nOutput files saved to: {remover.output_dir}")
            logger.info("All output images are PNG format with transparency support")


if __name__ == "__main__":
    main()
