import torch
import torch.nn as nn
from einops import rearrange
from model.helpers import get_patch_position_embedding


class PatchEmbedding(nn.Module):
    r"""
    Layer to take in the input image and do the following:
        1.  Transform grid of image patches into a sequence of patches.
            Number of patches are decided based on image height,width and
            patch height, width.
        2. Add positional embedding to the above sequence
    """

    def __init__(self,
                 image_height,
                 image_width,
                 im_channels,
                 patch_height,
                 patch_width,
                 hidden_size):
        """
        Initialize the Patch Embedding layer.

        :param image_height: Int. Total height of the input image.
        :param image_width: Int. Total width of the input image.
        :param im_channels: Int. Number of color channels in the input image 
            (e.g., 3 for RGB, or varying for latent inputs).
        :param patch_height: Int. Height of each individual patch. `image_height` 
            must be divisible by this value.
        :param patch_width: Int. Width of each individual patch. `image_width` 
            must be divisible by this value.
        :param hidden_size: Int. The target dimensionality of the projected patch 
            tokens (the Transformer's expected embedding dimension).
        """
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.im_channels = im_channels

        self.hidden_size = hidden_size

        self.patch_height = patch_height
        self.patch_width = patch_width

        # Input dimension for Patch Embedding FC Layer
        patch_dim = self.im_channels * self.patch_height * self.patch_width
        self.patch_embed = nn.Sequential(
            nn.Linear(patch_dim, self.hidden_size)
        )

        # DiT Layer Initialization
        nn.init.xavier_uniform_(self.patch_embed[0].weight)
        nn.init.constant_(self.patch_embed[0].bias, 0)

    def forward(self, x):
        """
        Forward pass to patchify the image and compute final sequence embeddings.

        Steps involved:
        1. Restructure the 2D grid of patches into a flattened sequence of patches.
        2. Linearly project the raw patch pixels into the `hidden_size` dimension.
        3. Generate and add a 2D sinusoidal position embedding to the sequence.

        :param x: Float tensor of shape ``(B, C, H, W)``, representing a batch of images.
        :return: Float tensor of shape ``(B, num_patches, hidden_size)`` containing 
            the sequence of embedded tokens, where ``num_patches = (H/ph) * (W/pw)``.
        """
        grid_size_h = self.image_height // self.patch_height
        grid_size_w = self.image_width // self.patch_width

        # B, C, H, W -> B, (Patches along height * Patches along width), Patch Dimension
        # Number of tokens = Patches along height * Patches along width
        out = rearrange(x, 'b c (nh ph) (nw pw) -> b (nh nw) (ph pw c)',
                        ph=self.patch_height,
                        pw=self.patch_width)

        # BxNumber of tokens x Patch Dimension -> B x Number of tokens x Transformer Dimension
        out = self.patch_embed(out)

        # Add 2d sinusoidal position embeddings
        pos_embed = get_patch_position_embedding(pos_emb_dim=self.hidden_size,
                                                 grid_size=(grid_size_h, grid_size_w),
                                                 device=x.device)
        out += pos_embed
        return out

