#!/usr/bin/env python3
"""
Archaeological Vision Pipeline Client
Detects and labels pottery pieces and reads text from labels
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


class ArcheoVisionClient:
    def __init__(
        self,
        pipeline_url: str = "http://localhost:8080",
        shared_path: str = "./archeo-shared",
        prompt_file: str = "prompt.txt",
        model: str = None
    ):
        self.pipeline_url = pipeline_url
        self.shared_path = Path(shared_path)
        self.prompt_file = prompt_file
        self.model = model
        
        # Create output directories
        self.images_dir = self.shared_path / "images"
        self.results_dir = self.shared_path / "results"
        self.visualizations_dir = self.shared_path / "visualizations"
        
        for dir_path in [self.results_dir, self.visualizations_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def load_prompt(self) -> str:
        """Load detection prompt from file"""
        prompt_path = self.shared_path / self.prompt_file
        
        if not prompt_path.exists():
            logger.warning(f"Prompt file not found: {prompt_path}")
            return ""
        
        with open(prompt_path, 'r') as f:
            prompt = f.read().strip()
        
        logger.info(f"Loaded prompt from {prompt_path}")
        return prompt
    
    def encode_image(self, image_path: Path) -> str:
        """Encode image to base64"""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    
    def detect_pottery(self, image_path: Path, prompt: str) -> Dict[str, Any]:
        """Run pottery detection and OCR"""
        logger.info(f"Processing pottery image: {image_path.name}")
        
        # Load image to get dimensions
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return {"success": False, "error": "Failed to load image"}

        image_height, image_width = image.shape[:2]
        logger.info(f"Actual image dimensions: {image_width}x{image_height}")

        # Note: We no longer inject image dimensions into the prompt
        # Instead, we ask Qwen to report the dimensions it perceives
        # The server will then scale coordinates based on the difference

        # Encode image
        image_base64 = self.encode_image(image_path)

        # Prepare request
        payload = {
            "image_base64": image_base64,
            "prompt": prompt
        }
        
        if self.model:
            payload["model"] = self.model
        
        # Call API
        try:
            response = requests.post(
                f"{self.pipeline_url}/detect",
                json=payload,
                timeout=300
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Detection failed: {e}")
            return {"success": False, "error": str(e)}
    
    def parse_pottery_response(self, detection_result: Dict, image_path: Path) -> Dict:
        """Parse detection result into pottery format"""
        try:
            raw_response = detection_result.get("raw_response", "")
            
            # Try to extract JSON from raw response
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', raw_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                pottery_data = json.loads(json_str)
            else:
                # Try parsing the whole response as JSON
                pottery_data = json.loads(raw_response)
            
            # Add image metadata
            pottery_data["image_path"] = str(image_path)
            pottery_data["image_width"] = detection_result["image_size"]["width"]
            pottery_data["image_height"] = detection_result["image_size"]["height"]
            
            return pottery_data
            
        except Exception as e:
            logger.error(f"Failed to parse pottery response: {e}")
            
            # Fallback: use bounding boxes from detection
            annotations = []
            for box in detection_result.get("boxes", []):
                annotations.append({
                    "bbox_2d": [
                        int(box["x1"]),
                        int(box["y1"]),
                        int(box["x3"]),
                        int(box["y3"])
                    ],
                    "label": box.get("label", "unknown")
                })
            
            return {
                "main_label": "unknown",
                "image_path": str(image_path),
                "image_width": detection_result["image_size"]["width"],
                "image_height": detection_result["image_size"]["height"],
                "annotations": annotations
            }
    
    def visualize_pottery(self, image_path: Path, pottery_data: Dict, output_path: Path):
        """Create visualization with bounding boxes and labels"""
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return
        
        # Generate colors for each annotation
        annotations = pottery_data.get("annotations", [])
        colors = [
            tuple(map(int, np.random.randint(50, 255, 3).tolist()))
            for _ in annotations
        ]
        
        # Draw each annotation
        for annotation, color in zip(annotations, colors):
            bbox = annotation["bbox_2d"]
            label = annotation.get("label", "")
            
            # Draw rectangle
            cv2.rectangle(
                image,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                color,
                3
            )
            
            # Draw label background
            label_text = label
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                2
            )
            
            cv2.rectangle(
                image,
                (bbox[0], bbox[1] - text_height - 10),
                (bbox[0] + text_width + 10, bbox[1]),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                image,
                label_text,
                (bbox[0] + 5, bbox[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
        
        # Add title with main label
        main_label = pottery_data.get("main_label", "")
        title = f"Main Label: {main_label} | Objects: {len(annotations)}"
        cv2.putText(
            image,
            title,
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )
        
        # Save
        cv2.imwrite(str(output_path), image)
        logger.info(f"Saved visualization: {output_path}")
    
    def process_image(self, image_path: Path, prompt: str):
        """Process single pottery image"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {image_path.name}")
        logger.info(f"{'='*60}")
        
        image_stem = image_path.stem
        
        # Run detection
        detection_result = self.detect_pottery(image_path, prompt)
        
        if not detection_result.get('success', False):
            logger.error(f"Detection failed: {detection_result.get('error', 'Unknown error')}")
            return
        
        logger.info(f"Detected {detection_result.get('count', 0)} objects")
        
        # Parse into pottery format
        pottery_data = self.parse_pottery_response(detection_result, image_path)
        
        # Save result JSON
        result_json_path = self.results_dir / f"{image_stem}_pottery.json"
        with open(result_json_path, 'w', encoding='utf-8') as f:
            json.dump(pottery_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved pottery JSON: {result_json_path}")
        
        # Create visualization
        viz_path = self.visualizations_dir / f"{image_stem}_labeled.jpg"
        self.visualize_pottery(image_path, pottery_data, viz_path)
        
        # Print summary
        logger.info(f"\nSummary:")
        logger.info(f"  Main Label: {pottery_data.get('main_label', 'N/A')}")
        logger.info(f"  Pottery Pieces: {len(pottery_data.get('annotations', []))}")
        for i, ann in enumerate(pottery_data.get('annotations', []), 1):
            logger.info(f"    {i}. {ann.get('label', 'N/A')}")
        
        logger.info(f"✓ Successfully processed {image_path.name}")
    
    def process_all_images(self):
        """Process all images in the images directory"""
        prompt = self.load_prompt()
        
        if not prompt:
            logger.error("No prompt loaded. Cannot process images.")
            return
        
        # Find all images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
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
        logger.info(f"\nResults saved to:")
        logger.info(f"  - JSON files: {self.results_dir}")
        logger.info(f"  - Visualizations: {self.visualizations_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Archaeological Vision Pipeline - Pottery Detection and Labeling"
    )
    parser.add_argument(
        "--pipeline-url",
        default="http://localhost:8080",
        help="Pipeline API URL (default: http://localhost:8080)"
    )
    parser.add_argument(
        "--shared-path",
        default="./archeo-shared",
        help="Shared folder path (default: ./archeo-shared)"
    )
    parser.add_argument(
        "--prompt-file",
        default="prompt.txt",
        help="Prompt file name (default: prompt.txt)"
    )
    parser.add_argument(
        "--model",
        default="qwen3-vl:8b",
        help="Vision model to use (default: qwen3-vl:8b)"
    )
    
    args = parser.parse_args()
    
    # Create client
    client = ArcheoVisionClient(
        pipeline_url=args.pipeline_url,
        shared_path=args.shared_path,
        prompt_file=args.prompt_file,
        model=args.model
    )
    
    # Process all images
    client.process_all_images()


if __name__ == "__main__":
    main()