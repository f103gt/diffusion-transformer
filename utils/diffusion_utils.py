import pickle
import glob
import os
import torch


def load_latents(latent_path):
    r"""
    Load pre-computed latent tensors from a directory of pickle files.

    Each pickle file is expected to contain a dictionary mapping image filenames
    to (latent_tensor, ...) tuples. The first element of each tuple is extracted.

    :param latent_path: Path to the directory containing ``*.pkl`` latent files.
    :return: Dict mapping image filename to its latent tensor.
    """
    latent_maps = {}
    for fname in glob.glob(os.path.join(latent_path, '*.pkl')):
        s = pickle.load(open(fname, 'rb'))
        for k, v in s.items():
            latent_maps[k] = v[0]
    return latent_maps
