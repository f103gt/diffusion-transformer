import glob
import os
import cv2
import json
import torch
import torchvision
import numpy as np
from PIL import Image
from utils.diffusion_utils import load_latents
from tqdm import tqdm
from torch.utils.data.dataset import Dataset


class LandscapesDataset(Dataset):
    """
    A PyTorch Dataset for loading images (or pre-computed latents) and their class labels.

    This dataset supports two primary modes of operation for training generative models:
    1. Raw Image Mode: Loads images from disk, applies center cropping, resizes them, 
       and normalizes the pixel values to the [-1, 1] range required by diffusion models.
    2. Latent Mode: Loads pre-computed latent representations (e.g., from a VAE). 
       This drastically speeds up training for Latent Diffusion Models (LDMs) since 
       the encoder does not need to run during every training step.
    """
    def __init__(self, split, im_path, im_size=256, im_channels=3, im_ext='jpg',
                 use_latents=False, latent_path=None, label_json_path=None):
        """
        Initialize the dataset, load file paths, and build label mappings.

        :param split: String. Indicates the dataset split (e.g., 'train', 'val').
        :param im_path: String. Path to the directory containing the images.
        :param im_size: Int. Target height and width for the images.
        :param im_channels: Int. Target number of image channels (e.g., 3 for RGB).
        :param im_ext: String. Primary image extension format (though glob searches multiple).
        :param use_latents: Bool. If True, the dataset will return pre-encoded latents 
            instead of raw images.
        :param latent_path: String. Path to the saved latent representations.
        :param label_json_path: String. Path to the JSON file containing the filename-to-class mapping.
        """
        self.split = split
        self.im_size = im_size
        self.im_channels = im_channels
        self.im_ext = im_ext
        self.im_path = im_path
        self.latent_maps = None
        self.use_latents = False
        self.label_json_path = label_json_path
        
        # Dictionary to store filename -> class_label mapping
        self.label_map = {}
        # Dictionary to map text labels to numeric IDs
        self.class_to_id = {}
        self.id_to_class = {}
        self.num_classes = 0

        self.images = self.load_images(im_path)
        
        # Load class labels from JSON file
        if label_json_path and os.path.exists(label_json_path):
            self.load_labels(label_json_path)
        else:
            print('Warning: No label JSON file provided or file not found.')
            print('Using default class 0 for all images.')

        # Whether to load images or to load latents
        if use_latents and latent_path is not None:
            latent_maps = load_latents(latent_path)
            if len(latent_maps) == len(self.images):
                self.use_latents = True
                self.latent_maps = latent_maps
                print('Found {} latents'.format(len(self.latent_maps)))
            else:
                print('Latents not found')

    def load_images(self, im_path):
        """
        Scans the target directory and collects paths for all supported image files.

        :param im_path: String. Directory to scan.
        :return: List of strings. Absolute or relative paths to the discovered images.
        """
        assert os.path.exists(im_path), "images path {} does not exist".format(im_path)
        ims = []
        fnames = glob.glob(os.path.join(im_path, '*.{}'.format('png')))
        fnames += glob.glob(os.path.join(im_path, '*.{}'.format('jpg')))
        fnames += glob.glob(os.path.join(im_path, '*.{}'.format('jpeg')))

        for fname in tqdm(fnames):
            ims.append(fname)

        print('Found {} images'.format(len(ims)))
        return ims
    
    def load_labels(self, label_json_path):
        """
        Parses the JSON file mapping filenames to their string class labels, 
        and generates a continuous integer ID for each unique class.

        Expected JSON format: {"image_001.jpg": "mountain", "image_002.png": "forest", ...}

        :param label_json_path: String. Path to the label configuration file.
        """
        with open(label_json_path, 'r') as f:
            label_data = json.load(f)
        
        # Build class_to_id mapping
        unique_classes = sorted(set(label_data.values()))
        self.class_to_id = {class_name: idx for idx, class_name in enumerate(unique_classes)}
        self.id_to_class = {idx: class_name for class_name, idx in self.class_to_id.items()}
        self.num_classes = len(unique_classes)
        
        # Convert text labels to numeric IDs
        for filename, class_name in label_data.items():
            self.label_map[filename] = self.class_to_id[class_name]
        
        print(f'Loaded labels for {len(self.label_map)} images')
        print(f'Found {self.num_classes} unique classes: {list(self.class_to_id.keys())}')
    
    def get_class_label(self, image_path):
        """
        Retrieves the integer class ID associated with a specific image file.

        :param image_path: String. Path to the image file.
        :return: Int. The numeric class ID. Defaults to 0 if the filename is not in the map.
        """
        filename = os.path.basename(image_path)
        
        if filename in self.label_map:
            return self.label_map[filename]
        else:
            # Default to class 0 if not found
            print(f'Warning: No label found for {filename}, using class 0')
            return 0

    def __len__(self):
        """
        Returns the total number of items in the dataset.
        """
        return len(self.images)

    def __getitem__(self, index):
        """
        Retrieves a single data sample and its corresponding class label.

        If `use_latents` is True, returns the pre-computed tensor directly.
        If False, loads the image, resizes/crops it, and scales pixel values to [-1, 1].

        :param index: Int. The index of the item to fetch.
        :return: A tuple `(data, label)` where:
            - `data` is a Float tensor. Shape is `(im_channels, im_size, im_size)` if raw, 
              or `(latent_channels, latent_h, latent_w)` if latents.
            - `label` is a Long tensor of shape `()` containing the integer class ID.
        """
        image_path = self.images[index]
        class_label = self.get_class_label(image_path)
        
        if self.use_latents:
            latent = self.latent_maps[image_path]
            return latent, torch.tensor(class_label, dtype=torch.long)

        else:
            im = Image.open(image_path)
            im_tensor = torchvision.transforms.Compose([
                torchvision.transforms.Resize(self.im_size),
                torchvision.transforms.CenterCrop(self.im_size),
                torchvision.transforms.ToTensor(),
            ])(im)
            im.close()

            # Convert input to -1 to 1 range.
            im_tensor = (2 * im_tensor) - 1

            return im_tensor, torch.tensor(class_label, dtype=torch.long)
