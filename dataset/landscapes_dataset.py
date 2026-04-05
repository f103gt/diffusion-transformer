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
    r"""
    LandscapesHQ dataset will by default centre crop and resize the images.
    This can be replaced by any other dataset. As long as all the images
    are under one directory.
    """

    def __init__(self, split, im_path, im_size=256, im_channels=3, im_ext='jpg',
                 use_latents=False, latent_path=None, label_json_path=None):
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
        r"""
        Gets all images from the path specified
        and stacks them all up
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
        Load class labels from JSON file.
        Expected format: {"filename.jpg": "class_name", ...}
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
        Get the class label for an image.
        Returns numeric class ID.
        """
        filename = os.path.basename(image_path)
        
        if filename in self.label_map:
            return self.label_map[filename]
        else:
            # Default to class 0 if not found
            print(f'Warning: No label found for {filename}, using class 0')
            return 0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
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
