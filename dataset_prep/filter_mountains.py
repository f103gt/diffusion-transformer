"""
Script to filter mountain-related images and create a separate dataset.
This will:
1. Filter labels JSON to only mountain categories
2. Copy mountain images to a new directory
3. Save filtered labels to mountain-labels.json
"""

import json
import os
import shutil
from pathlib import Path

# Define mountain categories
MOUNTAIN_CATEGORIES = {
    'mountain', 'snowy mountain', 'glacier', 'volcano', 
    'canyon', 'cliff', 'rocky coast'
}

def filter_mountain_dataset(
    labels_json_path: str,
    source_dataset_dir: str,
    output_labels_path: str = "mountain-labels.json",
    output_dataset_dir: str = "dataset_mountains"
):
    """
    Filter dataset to only include mountain-related images.
    
    Args:
        labels_json_path: Path to original dataset_labels.json
        source_dataset_dir: Path to directory containing all images
        output_labels_path: Path for filtered labels JSON file
        output_dataset_dir: Directory to copy mountain images to
    """
    
    print("="*70)
    print("MOUNTAIN DATASET FILTER")
    print("="*70)
    
    # Load original labels
    print(f"\nLoading labels from: {labels_json_path}")
    with open(labels_json_path, 'r') as f:
        all_labels = json.load(f)
    
    print(f"✓ Total images in dataset: {len(all_labels)}")
    print(f"✓ Total unique categories: {len(set(all_labels.values()))}")
    
    # Filter to mountain categories only
    print(f"\nFiltering to mountain categories: {sorted(MOUNTAIN_CATEGORIES)}")
    mountain_labels = {
        img_name: label 
        for img_name, label in all_labels.items() 
        if label in MOUNTAIN_CATEGORIES
    }
    
    unique_mountain_classes = sorted(set(mountain_labels.values()))
    print(f"Filtered to {len(mountain_labels)} images")
    print(f"Mountain classes found: {unique_mountain_classes}")
    
    # Print distribution
    print(f"\nDistribution by category:")
    for category in unique_mountain_classes:
        count = sum(1 for label in mountain_labels.values() if label == category)
        print(f"   {category:20s}: {count:5d} images")
    
    # Save filtered labels
    print(f"\nSaving filtered labels to: {output_labels_path}")
    with open(output_labels_path, 'w') as f:
        json.dump(mountain_labels, f, indent=2)
    print(f"Saved {len(mountain_labels)} labels")
    
    # Create output directory
    output_dir_path = Path(output_dataset_dir)
    if output_dir_path.exists():
        print(f"\n  Output directory already exists: {output_dataset_dir}")
        response = input("   Delete and recreate? (y/N): ").strip().lower()
        if response == 'y':
            print(f"   Deleting existing directory...")
            shutil.rmtree(output_dir_path)
        else:
            print(f"   Skipping directory creation and file copying")
            print(f"\n✓ Done! Filtered labels saved to {output_labels_path}")
            return
    
    print(f"\n Creating output directory: {output_dataset_dir}")
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Copy images
    print(f"\n📸 Copying {len(mountain_labels)} images...")
    source_dir = Path(source_dataset_dir)
    copied_count = 0
    missing_count = 0
    
    for img_name in mountain_labels.keys():
        source_path = source_dir / img_name
        dest_path = output_dir_path / img_name
        
        if source_path.exists():
            shutil.copy2(source_path, dest_path)
            copied_count += 1
            
            # Progress indicator
            if copied_count % 1000 == 0:
                print(f"   Copied {copied_count}/{len(mountain_labels)} images...")
        else:
            missing_count += 1
            print(f"    Missing: {img_name}")
    
    print(f"\n✓ Successfully copied {copied_count} images")
    if missing_count > 0:
        print(f"  {missing_count} images were not found in source directory")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Original dataset size:     {len(all_labels)} images")
    print(f"Mountain dataset size:     {len(mountain_labels)} images")
    print(f"Filtered categories:       {len(unique_mountain_classes)} classes")
    print(f"Images copied:             {copied_count}")
    print(f"Labels JSON saved to:      {output_labels_path}")
    print(f"Images directory:          {output_dataset_dir}")
    print(f"{'='*70}")
    print(f"\n Done! Mountain dataset created successfully!")


def main():
    """Main function to run the filtering script."""
    
    # Paths - adjust these if needed
    LABELS_JSON = r"D:\ai\diploma\dataset_labels.json"
    SOURCE_DATASET = r"D:\ai\diploma\dataset"
    OUTPUT_LABELS = r"D:\ai\diploma\mountain-labels.json"
    OUTPUT_DATASET = r"D:\ai\diploma\dataset_mountains"
    
    print("\n  Mountain Dataset Extractor\n")
    print(f"Source labels:  {LABELS_JSON}")
    print(f"Source images:  {SOURCE_DATASET}")
    print(f"Output labels:  {OUTPUT_LABELS}")
    print(f"Output images:  {OUTPUT_DATASET}")
    print()
    
    # Check if source files exist
    if not os.path.exists(LABELS_JSON):
        print(f" Error: Labels file not found: {LABELS_JSON}")
        return
    
    if not os.path.exists(SOURCE_DATASET):
        print(f" Error: Source dataset directory not found: {SOURCE_DATASET}")
        return
    
    # Run the filter
    filter_mountain_dataset(
        labels_json_path=LABELS_JSON,
        source_dataset_dir=SOURCE_DATASET,
        output_labels_path=OUTPUT_LABELS,
        output_dataset_dir=OUTPUT_DATASET
    )


if __name__ == "__main__":
    main()
