#!/usr/bin/env python3
"""
Vision AI Pipeline Client
Processes images through detection and segmentation pipeline
"""

import os
import sys
import json
import base64
import io
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
import requests
import cv2
import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VisionClient:
    def __init__(
        self,
        pipeline_url: str = "http://localhost:8080",
        sam_url: str = "http://localhost:8001",
        shared_path: str = "./shared",
        prompt_file: str = "prompt.txt",
        model: str = None
    ):
        self.pipeline_url = pipeline_url
        self.sam_url = sam_url
        self.shared_path = Path(shared_path)
        self.prompt_file = prompt_file
        self.model = model
        
        # Create output directories
        self.images_dir = self.shared_path / "images"
        self.detections_dir = self.shared_path / "detections"
        self.results_dir = self.shared_path / "results"
        self.masks_dir = self.shared_path / "masks"
        self.visualizations_dir = self.shared_path / "visualizations"
        
        for dir_path in [
            self.detections_dir,
            self.results_dir,
            self.masks_dir,
            self.visualizations_dir
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def load_prompt(self) -> str:
        """Load detection prompt from file"""
        prompt_path = self.shared_path / self.prompt_file
        
        if not prompt_path.exists():
            logger.warning(f"Prompt file not found: {prompt_path}")
            default_prompt = (
                "Detect all objects in this image. "
                "Return the results as a JSON array with 'label' and 'box' fields. "
                "Each 'box' should contain 8 coordinates [x1, y1, x2, y2, x3, y3, x4, y4] "
                "representing the 4 corners of the bounding box in pixel coordinates."
            )
            return default_prompt
        
        with open(prompt_path, 'r') as f:
            prompt = f.read().strip()
        
        logger.info(f"Loaded prompt from {prompt_path}")
        return prompt
    
    def encode_image(self, image_path: Path) -> str:
        """Encode image to base64"""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    
    def decode_image(self, image_base64: str) -> np.ndarray:
        """Decode base64 to numpy array"""
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    def draw_box_4points(
        self,
        image: np.ndarray,
        box: Dict[str, float],
        label: str,
        color: tuple,
        thickness: int = 2
    ):
        """Draw 4-point box on image"""
        points = np.array([
            [box['x1'], box['y1']],
            [box['x2'], box['y2']],
            [box['x3'], box['y3']],
            [box['x4'], box['y4']]
        ], dtype=np.int32)
        
        # Draw polygon
        cv2.polylines(image, [points], isClosed=True, color=color, thickness=thickness)
        
        # Draw label
        label_pos = (int(box['x1']), int(box['y1']) - 10)
        cv2.putText(
            image,
            label,
            label_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )
    
    def visualize_detection(
        self,
        image_path: Path,
        detection_result: Dict[str, Any],
        output_path: Path
    ):
        """Create visualization of detection results"""
        # Load image
        image = cv2.imread(str(image_path))
        
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return
        
        # Generate colors for each detection
        boxes = detection_result.get('boxes', [])
        colors = [
            tuple(map(int, np.random.randint(50, 255, 3).tolist()))
            for _ in boxes
        ]
        
        # Draw each box
        for box, color in zip(boxes, colors):
            self.draw_box_4points(
                image,
                box,
                box.get('label', 'object'),
                color,
                thickness=3
            )
        
        # Add info text
        info_text = f"Detected: {len(boxes)} objects"
        cv2.putText(
            image,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # Save
        cv2.imwrite(str(output_path), image)
        logger.info(f"Saved detection visualization: {output_path}")
    
    def visualize_segmentation(
        self,
        image_path: Path,
        detection_result: Dict[str, Any],
        segmentation_results: List[Dict[str, Any]],
        output_path: Path
    ):
        """Create visualization with masks and boxes"""
        # Load image
        image = cv2.imread(str(image_path))
        
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return
        
        # Create overlay
        overlay = image.copy()
        
        # Generate colors
        colors = [
            tuple(map(int, np.random.randint(50, 255, 3).tolist()))
            for _ in segmentation_results
        ]
        
        # Draw masks
        for result, color in zip(segmentation_results, colors):
            # Decode mask
            mask_base64 = result.get('mask_base64', '')
            if mask_base64:
                mask_bytes = base64.b64decode(mask_base64)
                mask_img = Image.open(io.BytesIO(mask_bytes))
                mask = np.array(mask_img) > 0
                
                # Apply colored mask
                overlay[mask] = overlay[mask] * 0.5 + np.array(color) * 0.5
        
        # Blend overlay
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Draw boxes
        boxes = detection_result.get('boxes', [])
        for box, color in zip(boxes, colors):
            self.draw_box_4points(
                image,
                box,
                f"{box.get('label', 'object')}",
                color,
                thickness=2
            )
        
        # Add info text
        info_text = f"Segmented: {len(segmentation_results)} objects"
        cv2.putText(
            image,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # Save
        cv2.imwrite(str(output_path), image)
        logger.info(f"Saved segmentation visualization: {output_path}")
    
    def detect_image(self, image_path: Path, prompt: str) -> Dict[str, Any]:
        """Run detection on image"""
        logger.info(f"Detecting objects in {image_path.name}")

        # Load image to get original dimensions
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return {"success": False, "error": "Failed to load image"}

        original_height, original_width = image.shape[:2]
        logger.info(f"Original image dimensions: {original_width}x{original_height}")

        # Resize image to 1200x900 for detection
        target_width = 1200
        target_height = 900
        resized_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
        logger.info(f"Resized image to: {target_width}x{target_height}")

        # Encode resized image to base64
        _, buffer = cv2.imencode('.jpg', resized_image)
        image_base64 = base64.b64encode(buffer).decode()

        # Prepare request
        payload = {
            "image_base64": image_base64,
            "prompt": prompt,
            "original_width": original_width,
            "original_height": original_height
        }

        if self.model:
            payload["model"] = self.model
        
        # Call API
        try:
            logger.info("Sending request to pipeline API (timeout: 1000s)")
            response = requests.post(
                f"{self.pipeline_url}/detect",
                json=payload,
                timeout=1000  # ~17 minutes - sufficient for Qwen3-vl:8b vision processing
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Detection failed: {e}")
            return {"success": False, "error": str(e)}
    
    def segment_objects(
        self,
        image_base64: str,
        boxes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run segmentation on detected objects"""
        logger.info(f"Segmenting {len(boxes)} objects")
        
        # Prepare request
        payload = {
            "image_base64": image_base64,
            "boxes": boxes
        }
        
        # Call SAM API directly
        try:
            response = requests.post(
                f"{self.sam_url}/segment",
                json=payload,
                timeout=350  # ~6 minutes - buffer over API timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Segmentation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def save_individual_masks(
        self,
        segmentation_results: List[Dict[str, Any]],
        image_name: str
    ):
        """Save individual mask images"""
        for result in segmentation_results:
            mask_base64 = result.get('mask_base64', '')
            if not mask_base64:
                continue
            
            # Create filename
            label = result.get('label', 'object')
            obj_id = result.get('id', 0)
            mask_filename = f"{image_name}_mask_{obj_id}_{label}.png"
            mask_path = self.masks_dir / mask_filename
            
            # Decode and save
            mask_bytes = base64.b64decode(mask_base64)
            with open(mask_path, 'wb') as f:
                f.write(mask_bytes)
            
            logger.info(f"Saved mask: {mask_path}")
    
    def process_image(self, image_path: Path, prompt: str):
        """Process single image through full pipeline"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {image_path.name}")
        logger.info(f"{'='*60}")
        
        image_stem = image_path.stem
        
        # Stage 1: Detection
        logger.info("Stage 1: Object Detection")
        detection_result = self.detect_image(image_path, prompt)
        
        if not detection_result.get('success', False):
            logger.error(f"Detection failed: {detection_result.get('error', 'Unknown error')}")
            return
        
        # Save detection results
        detection_json_path = self.detections_dir / f"{image_stem}_detection.json"
        with open(detection_json_path, 'w') as f:
            json.dump(detection_result, f, indent=2)
        logger.info(f"Saved detection JSON: {detection_json_path}")
        
        # Visualize detection
        detection_viz_path = self.detections_dir / f"{image_stem}_detection.jpg"
        self.visualize_detection(image_path, detection_result, detection_viz_path)
        
        # Check if any objects detected
        if detection_result.get('count', 0) == 0:
            logger.warning("No objects detected")
            return
        
        logger.info(f"Detected {detection_result['count']} objects")
        
        # Stage 2: Segmentation
        logger.info("Stage 2: Instance Segmentation")
        image_base64 = self.encode_image(image_path)
        segmentation_result = self.segment_objects(
            image_base64,
            detection_result.get('boxes', [])
        )
        
        if not segmentation_result.get('success', False):
            logger.error(f"Segmentation failed: {segmentation_result.get('error', 'Unknown error')}")
            return
        
        # Combine results
        final_results = {
            "image": image_path.name,
            "detection": detection_result,
            "segmentation": segmentation_result,
            "objects": []
        }
        
        # Create per-object results
        seg_results = segmentation_result.get('results', [])
        for seg in seg_results:
            obj_result = {
                "id": seg.get('id'),
                "label": seg.get('label'),
                "confidence": seg.get('confidence'),
                "box": seg.get('box'),
                "area": seg.get('area'),
                "polygon_count": len(seg.get('polygons', []))
            }
            final_results["objects"].append(obj_result)
        
        # Save final results
        result_json_path = self.results_dir / f"{image_stem}_result.json"
        with open(result_json_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        logger.info(f"Saved result JSON: {result_json_path}")
        
        # Save individual masks
        self.save_individual_masks(seg_results, image_stem)
        
        # Create final visualization
        viz_path = self.visualizations_dir / f"{image_stem}_segmentation.jpg"
        self.visualize_segmentation(
            image_path,
            detection_result,
            seg_results,
            viz_path
        )
        
        logger.info(f"✓ Successfully processed {image_path.name}")
    
    def process_all_images(self):
        """Process all images in the images directory"""
        # Load prompt
        prompt = self.load_prompt()
        logger.info(f"Using prompt: {prompt[:100]}...")
        
        # Find all images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(self.images_dir.glob(f"*{ext}"))
            image_files.extend(self.images_dir.glob(f"*{ext.upper()}"))
        
        if not image_files:
            logger.warning(f"No images found in {self.images_dir}")
            return
        
        logger.info(f"Found {len(image_files)} images to process")
        
        # Process each image
        success_count = 0
        for image_path in sorted(image_files):
            try:
                self.process_image(image_path, prompt)
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to process {image_path.name}: {e}", exc_info=True)
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing complete: {success_count}/{len(image_files)} successful")
        logger.info(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Vision AI Pipeline Client - Process images through detection and segmentation"
    )
    parser.add_argument(
        "--pipeline-url",
        default="http://localhost:8080",
        help="Pipeline API URL (default: http://localhost:8080)"
    )
    parser.add_argument(
        "--sam-url",
        default="http://localhost:8001",
        help="SAM service URL (default: http://localhost:8001)"
    )
    parser.add_argument(
        "--shared-path",
        default="./shared",
        help="Shared folder path (default: ./shared)"
    )
    parser.add_argument(
        "--prompt-file",
        default="prompt.txt",
        help="Prompt file name in shared folder (default: prompt.txt)"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Qwen model name (default: auto-detect)"
    )
    
    args = parser.parse_args()
    
    # Create client
    client = VisionClient(
        pipeline_url=args.pipeline_url,
        sam_url=args.sam_url,
        shared_path=args.shared_path,
        prompt_file=args.prompt_file,
        model=args.model
    )
    
    # Process all images
    client.process_all_images()


if __name__ == "__main__":
    main()
