#!/usr/bin/env python3
"""
Archaeological File Organizer
Renames and organizes pottery images and JSON files based on main_label
"""

import os
import json
import shutil
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ArcheoFileOrganizer:
    def __init__(self, shared_path: str = "./archeo-shared"):
        self.shared_path = Path(shared_path)
        self.images_dir = self.shared_path / "images"
        self.results_dir = self.shared_path / "results"
        self.final_dir = self.shared_path / "final"
        
        # Create final directory
        self.final_dir.mkdir(parents=True, exist_ok=True)
        
    def sanitize_filename(self, text: str) -> str:
        """Convert label to safe filename"""
        # Replace invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            text = text.replace(char, '_')
        
        # Remove extra spaces and trim
        text = '_'.join(text.split())
        
        return text
    
    def find_matching_image(self, json_path: Path) -> Path:
        """Find the original image file for a JSON result"""
        # JSON filename pattern: imagename_pottery.json
        # Image filename pattern: imagename.jpg/png/etc
        
        json_stem = json_path.stem
        if json_stem.endswith('_pottery'):
            image_stem = json_stem[:-8]  # Remove '_pottery'
        else:
            image_stem = json_stem
        
        # Try common image extensions
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            image_path = self.images_dir / f"{image_stem}{ext}"
            if image_path.exists():
                return image_path
        
        logger.warning(f"Could not find matching image for {json_path.name}")
        return None
    
    def process_file_pair(self, json_path: Path) -> bool:
        """Process a JSON file and its corresponding image"""
        try:
            # Read JSON
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Get main label
            main_label = data.get('main_label', '')
            if not main_label or main_label == 'unknown':
                logger.warning(f"No valid main_label in {json_path.name}, skipping")
                return False
            
            # Find corresponding image
            image_path = self.find_matching_image(json_path)
            if not image_path:
                return False
            
            # Create safe filename from main_label
            safe_name = self.sanitize_filename(main_label)
            
            # Get image extension
            image_ext = image_path.suffix
            
            # Define new paths
            new_image_path = self.final_dir / f"{safe_name}{image_ext}"
            new_json_path = self.final_dir / f"{safe_name}.json"
            
            # Check if files already exist
            counter = 1
            while new_image_path.exists() or new_json_path.exists():
                new_image_path = self.final_dir / f"{safe_name}_{counter}{image_ext}"
                new_json_path = self.final_dir / f"{safe_name}_{counter}.json"
                counter += 1
            
            # Copy image
            shutil.copy2(image_path, new_image_path)
            logger.info(f"Copied: {image_path.name} -> {new_image_path.name}")
            
            # Update JSON with new image path
            data['image_path'] = str(new_image_path.relative_to(self.shared_path))
            
            # Save updated JSON
            with open(new_json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved: {json_path.name} -> {new_json_path.name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing {json_path.name}: {e}", exc_info=True)
            return False
    
    def organize_all(self) -> Tuple[int, int]:
        """Organize all JSON files and their corresponding images"""
        logger.info(f"Organizing files from {self.results_dir}")
        logger.info(f"Output directory: {self.final_dir}")
        
        # Find all JSON files in results
        json_files = list(self.results_dir.glob("*.json"))
        
        if not json_files:
            logger.warning(f"No JSON files found in {self.results_dir}")
            return 0, 0
        
        logger.info(f"Found {len(json_files)} JSON files to process")
        
        # Process each file
        success_count = 0
        for json_path in sorted(json_files):
            if self.process_file_pair(json_path):
                success_count += 1
        
        return success_count, len(json_files)
    
    def create_index(self):
        """Create an index file listing all organized files"""
        index_data = {
            "total_files": 0,
            "pottery_collections": []
        }
        
        # Find all JSON files in final directory
        json_files = sorted(self.final_dir.glob("*.json"))
        
        for json_path in json_files:
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                index_data["pottery_collections"].append({
                    "main_label": data.get("main_label", ""),
                    "image_file": json_path.stem + Path(data.get("image_path", "")).suffix,
                    "json_file": json_path.name,
                    "pottery_count": len(data.get("annotations", [])),
                    "labels": [ann.get("label", "") for ann in data.get("annotations", [])]
                })
            except Exception as e:
                logger.error(f"Error reading {json_path.name}: {e}")
        
        index_data["total_files"] = len(index_data["pottery_collections"])
        
        # Save index
        index_path = self.final_dir / "index.json"
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Created index file: {index_path}")
        
        # Also create a simple text listing
        listing_path = self.final_dir / "index.txt"
        with open(listing_path, 'w', encoding='utf-8') as f:
            f.write("Archaeological Pottery Collection Index\n")
            f.write("=" * 60 + "\n\n")
            
            for i, collection in enumerate(index_data["pottery_collections"], 1):
                f.write(f"{i}. {collection['main_label']}\n")
                f.write(f"   Image: {collection['image_file']}\n")
                f.write(f"   Pottery pieces: {collection['pottery_count']}\n")
                f.write(f"   Labels: {', '.join(collection['labels'])}\n")
                f.write("\n")
            
            f.write(f"\nTotal collections: {index_data['total_files']}\n")
        
        logger.info(f"Created text listing: {listing_path}")
        
        return index_data


def main():
    parser = argparse.ArgumentParser(
        description="Organize archaeological files by main_label"
    )
    parser.add_argument(
        "--shared-path",
        default="./archeo-shared",
        help="Shared folder path (default: ./archeo-shared)"
    )
    parser.add_argument(
        "--create-index",
        action="store_true",
        help="Create index files after organizing"
    )
    
    args = parser.parse_args()
    
    # Create organizer
    organizer = ArcheoFileOrganizer(shared_path=args.shared_path)
    
    # Organize files
    logger.info("\n" + "=" * 60)
    logger.info("Archaeological File Organizer")
    logger.info("=" * 60 + "\n")
    
    success_count, total_count = organizer.organize_all()
    
    logger.info("\n" + "=" * 60)
    logger.info(f"Organization complete: {success_count}/{total_count} successful")
    logger.info("=" * 60 + "\n")
    
    # Create index if requested
    if args.create_index:
        logger.info("Creating index files...")
        index_data = organizer.create_index()
        logger.info(f"\nIndexed {index_data['total_files']} pottery collections")
    
    # Print summary
    logger.info("\nOrganized files location:")
    logger.info(f"  {organizer.final_dir}")
    logger.info("\nFiles are renamed using main_label from JSON")
    logger.info("JSON files have updated image_path references")
    
    if success_count > 0:
        logger.info(f"\n✓ Successfully organized {success_count} pottery collection(s)")


if __name__ == "__main__":
    main()
