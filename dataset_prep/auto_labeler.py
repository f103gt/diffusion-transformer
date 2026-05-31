import torch
import os
import glob
import json
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# --- CONFIG ---
DATASET_PATH = r"./dataset"
BATCH_SIZE = 128 # How many images to label at once
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0  # Number of CPU threads loading images in background


CANDIDATE_LABELS = [
    # --- Natural Landscapes ---
    "mountain", "snowy mountain", "volcano", "canyon", "desert", "sand dunes",
    "forest", "jungle", "woods", "field", "grassland", "meadow", "wetland", "swamp", "marsh",
    # --- Water & Coast ---
    "ocean", "beach", "rocky coast", "cliff", "lake", "river", "waterfall", "frozen lake", "glacier",
    # --- Urban ---
    "cityscape", "skyline", "urban park", "village", "town", "castle", "ruins", "bridge", 
    "road", "highway", "farm", "lighthouse", "architecture",
    # --- Atmospheric ---
    "sunset", "sunrise", "starry night", "stormy sky", "foggy landscape", "misty forest",
    # --- Trash Class ---
    "noise or low quality image" 
]

class ImageDataset(Dataset):
    def __init__(self,folder_path):
        self.image_paths = glob.glob(os.path.join(folder_path, "*.jpg"))
        self.image_paths += glob.glob(os.path.join(folder_path, "*.png"))
        print(f"Found {len(self.image_paths)} images.")

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        path = self.image_paths[index]
        try:
            # we return the open PIL image.
            # processing happens in the main loop.
            image = Image.open(path).convert("RGB")
            return image, path
        except Exception as e:
            # return None if error, we filter it out later
            print(f"Error loading {path}: {e}")
            return None, path
        
def collate_fn(batch):
    # filter out failed images (None)
    batch = [(img,path) for img,path in batch if img if not None]
    if len(batch) == 0:
        return [], []
    images, paths = zip(*batch)
    return list(images), list(paths)


def main():
    print(f"Loading CLIP to auto-label data on {DEVICE}...")
    # load CLIP model standard version
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)

    # ---- STEP 1: PRE-COMPUTE TEXT FEATURES (ONCE) -----
    print("Pre-computing text features...")
    # # Find images
    # image_paths = glob.glob(os.path.join(DATASET_PATH, "*.jpg"))
    # image_paths += glob.glob(os.path.join(DATASET_PATH, "*.png"))
    # print(f"Found {len(image_paths)} images.")
    text_inputs = processor(
        text=CANDIDATE_LABELS,
        return_tensors="pt",
        padding=True
    ).to(DEVICE)

    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        # normalize features (crucial for CLIP)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    # --- STEP 2: SETUP DATALOAD ---
    dataset = ImageDataset(DATASET_PATH)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    results = {}
    print("Starting auto-labeling...")

    # --- STEP 3: MAIN LOOP ---
    # we use Mixed Precision (autocast) for speed
    with torch.no_grad():
        for batch_images, batch_paths in tqdm(dataloader):
            if not batch_paths:
                continue

            # process images
            image_inputs = processor(
                images=batch_images,
                return_tensors="pt"
            ).to(DEVICE)

            with torch.amp.autocast('cuda'): # Faster on GPU
                image_features = model.get_image_features(**image_inputs)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                # Calculate similarly (Dot product)
                # (Batch, 512) @ (Labels, 512).T -> (Batch, Labels)
                similarity = image_features @ text_features.T

                # get top match
                probs = similarity.softmax(dim=1)
                top_probs, top_indices = probs.max(dim=1)

            for path, index in zip(batch_paths, top_indices):
                filename = os.path.basename(path)
                results[filename] = CANDIDATE_LABELS[index.item()]

    for i in tqdm(range(0, len(image_paths), BATCH_SIZE)):
        batch_paths = image_paths[i : i + BATCH_SIZE]
        images = []
        valid_paths = []

        # load images
        for path in batch_paths:
            try:
                images.append(Image.open(path).convert("RGB"))
                valid_paths.append(path)
            except Exception as e:
                print(f"Skipping {path}: {e}")
                continue

        if not images:
            continue

        # prepare inputs
        inputs = processor(
            text=CANDIDATE_LABELS,
            images=images,
            return_tensors="pt",
            padding=True
        ).to(DEVICE)

        # forward pass(get probabilities)
        with torch.no_grad():
            outputs = model(**inputs)
            # logits_per_image: (Batch_Size, Num_Labels)
            probs = outputs.logits_per_image.softmax(dim=1)
            
            # get the index of the highest probability
            top_probs, top_indices = probs.max(dim=1)

        # save results
        for path, index in zip(valid_paths, top_indices):
            label = CANDIDATE_LABELS[index.item()]

            # save just the filename, not the full path (cleaner)
            filename = os.path.basename(path)
            results[filename] = label

    with open("dataset_labels.json", "w") as f:
        json.dump(results, f, indent=4)

        print("Done! Saved labels to dataset_labels.json")


if __name__ == "__main__":
    main()


    