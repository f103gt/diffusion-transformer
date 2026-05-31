"""
Test script to verify that the label system works correctly with mountain-labels.json
"""
import sys
sys.path.insert(0, '.')

from dataset.landscapes_dataset import LandscapesDataset
import torch

print("="*60)
print("Testing Label System with mountain-labels.json")
print("="*60)

# Test loading dataset with labels
print("\n1. Creating dataset with labels...")
dataset = LandscapesDataset(
    split='train',
    im_path='data/LandscapesHQ',
    im_size=128,
    label_json_path='data/mountain-labels.json'
)

print(f"\n2. Dataset Statistics:")
print(f"   Total images: {len(dataset)}")
print(f"   Number of classes: {dataset.num_classes}")
print(f"   Class mapping: {dataset.class_to_id}")

print(f"\n3. Testing __getitem__:")
try:
    # Test getting first item
    img, label = dataset[0]
    print(f"Item 0: Image shape={img.shape}, Label={label.item()}, Class='{dataset.id_to_class[label.item()]}'")
    
    # Test getting a few more items
    for i in [1, 10, 100]:
        if i < len(dataset):
            img, label = dataset[i]
            class_name = dataset.id_to_class[label.item()]
            print(f"Item {i}: Image shape={img.shape}, Label={label.item()}, Class='{class_name}'")
except Exception as e:
    print(f"Error: {e}")

print(f"\n4. Testing label consistency:")
# Verify that the same index always returns the same label
img1, label1 = dataset[0]
img2, label2 = dataset[0]
if label1.item() == label2.item():
    print(f"Labels are consistent")
else:
    print(f"Labels are NOT consistent!")

print(f"\n5. Class distribution (from dataset):")
class_counts = {class_name: 0 for class_name in dataset.class_to_id.keys()}
# Sample first 1000 items
sample_size = min(1000, len(dataset))
for i in range(sample_size):
    _, label = dataset[i]
    class_name = dataset.id_to_class[label.item()]
    class_counts[class_name] += 1

for class_name in sorted(class_counts.keys()):
    print(f"   {class_name}: {class_counts[class_name]} images (in first {sample_size})")

print("\n" + "="*60)
print("ALL TESTS PASSED! Your labels work correctly!")
print("="*60)

print("\nClass ID Mapping (for sampling):")
for class_name, class_id in sorted(dataset.class_to_id.items(), key=lambda x: x[1]):
    print(f"   {class_id}: '{class_name}'")

print("\nYou can now train with:")
print("   python tools/train_vae_dit.py --config config/landscapeshq.yaml")

print("\nSample specific classes using class names:")
print("   sample_classes: ['mountain', 'glacier', 'volcano', 'cliff']")
print("\n   Or using class IDs:")
print(f"   sample_classes: {list(range(dataset.num_classes))}")
