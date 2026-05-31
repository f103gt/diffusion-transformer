"""
Helper script to validate and manage JSON label files.
"""

import json
import os
import argparse
from pathlib import Path


def validate_json(json_path, image_dir=None):
    """
    Validate the JSON label file.
    """
    with open(json_path, 'r') as f:
        labels = json.load(f)
    
    print(f'    Validation Report for {json_path}')
    print(f'   Total labels: {len(labels)}')
    
    unique_classes = sorted(set(labels.values()))
    print(f'   Unique classes: {len(unique_classes)}')
    print(f'   Classes: {unique_classes}')
    
    # Count per class
    class_counts = {}
    for class_name in labels.values():
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print(f'\n   Class distribution:')
    for class_name in sorted(class_counts.keys()):
        print(f'      {class_name}: {class_counts[class_name]} images')
    
    # Check if files exist
    if image_dir:
        image_path = Path(image_dir)
        missing = []
        for filename in labels.keys():
            # Check in main directory and subdirectories
            found = list(image_path.glob(f'**/{filename}'))
            if not found:
                missing.append(filename)
        
        if missing:
            print(f'\n    Warning: {len(missing)} files not found in {image_dir}')
            print(f'      First 5 missing: {missing[:5]}')
        else:
            print(f'\n    All {len(labels)} image files found in {image_dir}')


def merge_jsons(json_files, output_json):
    """
    Merge multiple JSON label files into one.
    Later files override earlier ones if there are conflicts.
    """
    merged = {}
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            labels = json.load(f)
            merged.update(labels)
    
    with open(output_json, 'w') as f:
        json.dump(merged, f, indent=2)
    
    print(f'Merged {len(json_files)} JSON files')
    print(f'Total labels: {len(merged)}')
    print(f'Saved to: {output_json}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate and manage JSON label files')
    parser.add_argument('--mode', choices=['validate', 'merge'], required=True,
                        help='Mode: validate (check JSON format and stats), merge (combine multiple JSONs)')
    parser.add_argument('--input', type=str, help='Input JSON file path for validation')
    parser.add_argument('--output', type=str, default='data/image_labels.json',
                        help='Output JSON file path for merge')
    parser.add_argument('--image-dir', type=str, help='Image directory for validation (optional)')
    parser.add_argument('--merge-files', nargs='+', help='JSON files to merge')
    
    args = parser.parse_args()
    
    if args.mode == 'validate':
        if not args.input:
            print('Error: --input required for validate mode')
            exit(1)
        validate_json(args.input, args.image_dir)
        
        print('\nNext steps:')
        print(f'   1. Update config/landscapeshq.yaml with: label_json_path: \'{args.input}\'')
        print(f'   2. Run training: python tools/train_vae_dit.py --config config/landscapeshq.yaml')
        
    elif args.mode == 'merge':
        if not args.merge_files:
            print('Error: --merge-files required for merge mode')
            exit(1)
        merge_jsons(args.merge_files, args.output)
        
        print('\nNext steps:')
        print(f'   1. Review the merged labels in: {args.output}')
        print(f'   2. Update config/landscapeshq.yaml with: label_json_path: \'{args.output}\'')
        print(f'   3. Run training: python tools/train_vae_dit.py --config config/landscapeshq.yaml')
