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
        self.img_rescaled_dir = self.shared_path / "img_rescaled"
        self.results_dir = self.shared_path / "results"
        self.visualizations_dir = self.shared_path / "visualizations"

        for dir_path in [self.img_rescaled_dir, self.results_dir, self.visualizations_dir]:
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

    def rescale_and_pad_image(self, image_path: Path) -> tuple[Path, Dict[str, Any]]:
        """
        Rescale image to 1024px width and pad to make it square (1024x1024)

        Returns:
            tuple: (rescaled_image_path, rescaling_info)
            rescaling_info contains: original_width, original_height, scale_factor,
            padding_top, padding_bottom, target_size
        """
        # Load original image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        original_height, original_width = image.shape[:2]
        target_width = 1024

        # Calculate scale factor to resize to 1024px width
        scale_factor = target_width / original_width

        # Calculate new height after scaling
        scaled_height = int(original_height * scale_factor)

        # Resize image to target width
        resized_image = cv2.resize(image, (target_width, scaled_height), interpolation=cv2.INTER_LANCZOS4)

        # Calculate padding needed to make it square (1024x1024)
        target_size = 1024
        padding_needed = target_size - scaled_height

        if padding_needed > 0:
            # Add padding to top and bottom
            padding_top = padding_needed // 2
            padding_bottom = padding_needed - padding_top

            # Generate random light color for padding (250-255 for each RGB channel)
            padding_color = [
                int(np.random.randint(250, 256)),
                int(np.random.randint(250, 256)),
                int(np.random.randint(250, 256))
            ]

            # Pad image with random light pixels
            padded_image = cv2.copyMakeBorder(
                resized_image,
                padding_top,
                padding_bottom,
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=padding_color
            )
        elif padding_needed < 0:
            # Image is taller than target, crop from center
            crop_start = (-padding_needed) // 2
            crop_end = crop_start + target_size
            padded_image = resized_image[crop_start:crop_end, :]
            padding_top = -crop_start
            padding_bottom = scaled_height - crop_end
        else:
            # Image is already square
            padded_image = resized_image
            padding_top = 0
            padding_bottom = 0

        # Save rescaled image
        rescaled_image_path = self.img_rescaled_dir / image_path.name
        cv2.imwrite(str(rescaled_image_path), padded_image)
        logger.info(f"Saved rescaled image: {rescaled_image_path}")

        # Store rescaling information
        rescaling_info = {
            "original_width": original_width,
            "original_height": original_height,
            "scale_factor": scale_factor,
            "padding_top": padding_top,
            "padding_bottom": padding_bottom,
            "target_size": target_size,
            "scaled_height": scaled_height
        }

        logger.info(f"Rescaling info: original={original_width}x{original_height}, "
                   f"scale={scale_factor:.3f}, padding_top={padding_top}, padding_bottom={padding_bottom}")

        return rescaled_image_path, rescaling_info

    def detect_pottery(self, image_path: Path, prompt: str, rescaling_info: Dict[str, Any]) -> Dict[str, Any]:
        """Run pottery detection and OCR using rescaled image"""
        logger.info(f"Processing pottery image: {image_path.name}")

        # Use rescaled image dimensions (1024x1024)
        image_width = rescaling_info["target_size"]
        image_height = rescaling_info["target_size"]
        logger.info(f"Rescaled image dimensions: {image_width}x{image_height}")

        # Inject rescaled image dimensions into prompt
        prompt_with_dims = prompt.replace("{width}", str(image_width))
        prompt_with_dims = prompt_with_dims.replace("{height}", str(image_height))

        # Encode rescaled image
        image_base64 = self.encode_image(image_path)

        # Prepare request
        payload = {
            "image_base64": image_base64,
            "prompt": prompt_with_dims
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
            result = response.json()

            # Attach rescaling info to result for later use
            result["rescaling_info"] = rescaling_info

            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"Detection failed: {e}")
            return {"success": False, "error": str(e)}

    def reconstruct_bounding_boxes(self, boxes: List[Dict], rescaling_info: Dict[str, Any]) -> List[Dict]:
        """
        Reconstruct bounding boxes from rescaled coordinates to original image coordinates

        Args:
            boxes: List of bounding boxes from QWEN (in rescaled 1024x1024 coordinates)
            rescaling_info: Dictionary containing rescaling parameters

        Returns:
            List of bounding boxes in original image coordinates
        """
        scale_factor = rescaling_info["scale_factor"]
        padding_top = rescaling_info["padding_top"]
        original_width = rescaling_info["original_width"]
        original_height = rescaling_info["original_height"]

        reconstructed_boxes = []

        for box in boxes:
            # Get coordinates from rescaled image (1024x1024)
            x1, y1 = box.get("x1", 0), box.get("y1", 0)
            x2, y2 = box.get("x2", 0), box.get("y2", 0)
            x3, y3 = box.get("x3", 0), box.get("y3", 0)
            x4, y4 = box.get("x4", 0), box.get("y4", 0)

            # Step 1: Remove padding from y-coordinates
            y1_unpadded = y1 - padding_top
            y2_unpadded = y2 - padding_top
            y3_unpadded = y3 - padding_top
            y4_unpadded = y4 - padding_top

            # Step 2: Scale back to original dimensions
            x1_original = x1 / scale_factor
            y1_original = y1_unpadded / scale_factor
            x2_original = x2 / scale_factor
            y2_original = y2_unpadded / scale_factor
            x3_original = x3 / scale_factor
            y3_original = y3_unpadded / scale_factor
            x4_original = x4 / scale_factor
            y4_original = y4_unpadded / scale_factor

            # Clamp to original image boundaries
            x1_original = max(0, min(x1_original, original_width))
            x2_original = max(0, min(x2_original, original_width))
            x3_original = max(0, min(x3_original, original_width))
            x4_original = max(0, min(x4_original, original_width))
            y1_original = max(0, min(y1_original, original_height))
            y2_original = max(0, min(y2_original, original_height))
            y3_original = max(0, min(y3_original, original_height))
            y4_original = max(0, min(y4_original, original_height))

            # Create reconstructed box
            reconstructed_box = {
                "x1": x1_original,
                "y1": y1_original,
                "x2": x2_original,
                "y2": y2_original,
                "x3": x3_original,
                "y3": y3_original,
                "x4": x4_original,
                "y4": y4_original,
                "label": box.get("label", "unknown")
            }

            reconstructed_boxes.append(reconstructed_box)

        logger.info(f"Reconstructed {len(reconstructed_boxes)} bounding boxes to original coordinates")
        return reconstructed_boxes

    def parse_pottery_response(self, detection_result: Dict, image_path: Path) -> Dict:
        """Parse detection result into pottery format"""
        # Get rescaling info
        rescaling_info = detection_result.get("rescaling_info", {})
        original_width = rescaling_info.get("original_width", detection_result["image_size"]["width"])
        original_height = rescaling_info.get("original_height", detection_result["image_size"]["height"])

        # Reconstruct bounding boxes to original coordinates
        boxes = detection_result.get("boxes", [])
        if rescaling_info:
            reconstructed_boxes = self.reconstruct_bounding_boxes(boxes, rescaling_info)
        else:
            reconstructed_boxes = boxes

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

            # If pottery_data contains annotations with bounding boxes, reconstruct them too
            if "annotations" in pottery_data:
                for annotation in pottery_data["annotations"]:
                    if "bbox_2d" in annotation and rescaling_info:
                        # Convert bbox_2d [x1, y1, x3, y3] to 4-point format
                        bbox = annotation["bbox_2d"]
                        temp_box = {
                            "x1": bbox[0], "y1": bbox[1],
                            "x2": bbox[2], "y2": bbox[1],
                            "x3": bbox[2], "y3": bbox[3],
                            "x4": bbox[0], "y4": bbox[3]
                        }
                        reconstructed = self.reconstruct_bounding_boxes([temp_box], rescaling_info)[0]
                        annotation["bbox_2d"] = [
                            int(reconstructed["x1"]),
                            int(reconstructed["y1"]),
                            int(reconstructed["x3"]),
                            int(reconstructed["y3"])
                        ]

            # Add image metadata (use original dimensions)
            pottery_data["image_path"] = str(image_path)
            pottery_data["image_width"] = original_width
            pottery_data["image_height"] = original_height
            pottery_data["raw_response"] = raw_response

            return pottery_data

        except Exception as e:
            logger.error(f"Failed to parse pottery response: {e}")

            # Fallback: use reconstructed bounding boxes from detection
            annotations = []
            for box in reconstructed_boxes:
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
                "image_width": original_width,
                "image_height": original_height,
                "raw_response": detection_result.get("raw_response", ""),
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

        # Step 1: Rescale and pad image to 1024x1024
        try:
            rescaled_image_path, rescaling_info = self.rescale_and_pad_image(image_path)
        except Exception as e:
            logger.error(f"Failed to rescale image: {e}")
            return

        # Step 2: Run detection on rescaled image
        detection_result = self.detect_pottery(rescaled_image_path, prompt, rescaling_info)

        if not detection_result.get('success', False):
            logger.error(f"Detection failed: {detection_result.get('error', 'Unknown error')}")
            return

        logger.info(f"Detected {detection_result.get('count', 0)} objects")

        # Step 3: Parse into pottery format (bounding boxes are reconstructed to original coordinates)
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
        default="qwen2.5-vl:7b",
        help="Vision model to use (default: qwen2.5-vl:7b)"
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