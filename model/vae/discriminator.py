import torch
import torch.nn as nn


class Discriminator(nn.Module):
    r"""
    PatchGAN Discriminator.
    Rather than taking IMG_CHANNELSxIMG_HxIMG_W all the way to
    1 scalar value , we instead predict grid of values.
    Where each grid is prediction of how likely
    the discriminator thinks that the image patch corresponding
    to the grid cell is real
    """
    
    def __init__(self, im_channels=3,
                 conv_channels=None,
                 kernels=None,
                 strides=None,
                 paddings=None):
        """
        Initializes the PatchGAN Discriminator layers.

        Args:
            im_channels (int): Number of channels in the input image. Default: 3.
            conv_channels (list of int, optional): Channel dimensions for the intermediate 
                convolutional layers. Default: [64, 128, 256].
            kernels (list of int, optional): Kernel sizes for each conv layer. Default: [4, 4, 4, 4].
            strides (list of int, optional): Strides for each conv layer. Default: [2, 2, 2, 1].
            paddings (list of int, optional): Padding applied in each conv layer. Default: [1, 1, 1, 1].

        Architecture Logic:
            The model dynamically builds a series of Convolutional blocks based on the lists provided.
            - Block structure: Conv2d -> BatchNorm2d -> LeakyReLU
            - The first layer does NOT use BatchNorm and uses a Bias.
            - The last layer maps to exactly 1 channel (the scalar output per patch) 
              and does NOT use BatchNorm or an Activation function.
              
        Implemented Formulas:
            1. Spatial Size Calculation (per layer):
               H_out = floor((H_in + 2 * padding - kernel_size) / stride + 1)
               
            2. LeakyReLU Activation:
               f(x) = x if x >= 0 else 0.2 * x
        """
        super().__init__()
        if conv_channels is None:
            conv_channels = [64, 128, 256]
        if kernels is None:
            kernels = [4, 4, 4, 4]
        if strides is None:
            strides = [2, 2, 2, 1]
        if paddings is None:
            paddings = [1, 1, 1, 1]
        self.im_channels = im_channels
        activation = nn.LeakyReLU(0.2)
        channel_dims = [self.im_channels] + conv_channels + [1]
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channel_dims[i], channel_dims[i + 1],
                          kernel_size=kernels[i],
                          stride=strides[i],
                          padding=paddings[i],
                          bias=False if i != 0 else True),
                nn.BatchNorm2d(channel_dims[i + 1]) if i != len(channel_dims) - 2 and i != 0 else nn.Identity(),
                activation if i != len(channel_dims) - 2 else nn.Identity()
            )
            for i in range(len(channel_dims) - 1)
        ])
    
    def forward(self, x):
        """
        Forward pass of the PatchGAN discriminator.

        Args:
            x (torch.Tensor): A batch of images. 
                Expected shape: (Batch_Size, im_channels, Height, Width)

        Returns:
            torch.Tensor: A batch of 2D grids of probabilities (logits).
                Expected shape: (Batch_Size, 1, Grid_Height, Grid_Width)
                
                Example: If input is (2, 3, 256, 256) and defaults are used:
                - Layer 1 (stride 2): (2, 64, 128, 128)
                - Layer 2 (stride 2): (2, 128, 64, 64)
                - Layer 3 (stride 2): (2, 256, 32, 32)
                - Layer 4 (stride 1): (2, 1, 31, 31) -> This is the final PatchGAN grid.
        """
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


if __name__ == '__main__':
    x = torch.randn((2,3, 256, 256))
    prob = Discriminator(im_channels=3)(x)
    print(prob.shape)
