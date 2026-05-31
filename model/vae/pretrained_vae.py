"""
HuggingFace Stable Diffusion VAE Wrapper

This module provides a wrapper around the pre-trained Stable Diffusion VAE from
HuggingFace's diffusers library. It automatically downloads the official weights
and provides fine-tuning capabilities for your specific datasets.

Benefits of using the pre-trained VAE:
- No need to train from scratch (~3-5 hours on a GPU)
- Stable, well-tested architecture proven on billions of images
- Can be fine-tuned with much less data
- Latent space is standardized and proven to work with diffusion models
"""

import torch
import torch.nn as nn
from diffusers import AutoencoderKL


class HuggingFaceVAEWrapper(nn.Module):
    """
    Wrapper around HuggingFace's pre-trained Stable Diffusion VAE (AutoencoderKL).
    
    This wrapper:
    1. Automatically downloads pre-trained weights on first use
    2. Translates inputs/outputs to match the training loop requirements
    3. Provides scaling/unscaling of latents (standard practice for SD VAEs)
    4. Supports both training (fine-tuning) and inference modes
    
    The VAE compresses 128×128 RGB images into 32×32×4 latent vectors,
    achieving ~16× compression with minimal quality loss.
    """
    
    def __init__(self, pretrained_model_name_or_path="stabilityai/sd-vae-ft-mse", device='cuda'):
        """
        Initialize the HuggingFace VAE wrapper.
        
        Args:
            pretrained_model_name_or_path (str): HuggingFace model identifier
                - "stabilityai/sd-vae-ft-mse": Official Stable Diffusion VAE (recommended)
                - "stabilityai/sd-vae-ft-ema": EMA version (slightly better quality)
                - Custom path to local model
            device (str): Device to load the model on ('cuda', 'cpu', 'mps')
        
        Note:
            First initialization will download ~330MB of weights. Subsequent 
            runs use the cached weights (~/.cache/huggingface/hub/)
        """
        super().__init__()
        
        print(f"Loading pre-trained VAE from '{pretrained_model_name_or_path}'...")
        print("(First download may take a minute. Weights will be cached.)")
        
        # Load the pre-trained VAE
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch.float32
        ).to(device)
        
        # Freeze encoder/decoder by default (only fine-tune if needed)
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
        
        print(f"VAE loaded successfully!")
        print(f"Input shape: (B, 3, 128, 128) - RGB images")
        print(f"Latent shape: (B, 4, 32, 32) - Compressed representation")
        print(f"Compression ratio: 16×")

    def forward(self, x):
        """
        Full encode-decode pass for training.
        
        This method encodes the image to latent space, samples from the 
        distribution, and decodes back to image space. Used for computing
        reconstruction and perceptual losses during training.
        
        Args:
            x (torch.Tensor): Input images of shape (B, 3, 128, 128)
                             Assumed to be in range [-1, 1]
        
        Returns:
            reconstruction (torch.Tensor): Reconstructed images (B, 3, 128, 128)
            encoder_output (torch.Tensor): Concatenated [mean, logvar] (B, 8, 32, 32)
                                          Used for KL divergence loss computation
        """
        is_finetuning = next(self.vae.parameters()).requires_grad

        if is_finetuning:
            posterior = self.vae.encode(x).latent_dist
            z = posterior.sample()
            reconstruction = self.vae.decode(z).sample
        else:
            with torch.no_grad():
                posterior = self.vae.encode(x).latent_dist
                z = posterior.sample()
                reconstruction = self.vae.decode(z).sample

        mean = posterior.mean
        logvar = torch.log(posterior.var)
        encoder_output = torch.cat([mean, logvar], dim=1)  # (B, 8, 32, 32)

        return reconstruction, encoder_output

    def encode(self, x):
        """
        Encode images to latent space (used for pre-computing latents for DiT training).
        
        This is the standard encoding pipeline for extracting latent representations
        that will be used as input to your DiT model.
        
        Args:
            x (torch.Tensor): Input images of shape (B, 3, 128, 128)
        
        Returns:
            z_scaled (torch.Tensor): Scaled latents (B, 4, 32, 32)
            encoder_output (torch.Tensor): [mean, logvar] for KL loss (B, 8, 32, 32)
        
        Note:
            The latents are scaled by `self.vae.config.scaling_factor` (default: 0.18215)
            This scaling is standard for Stable Diffusion and improves diffusion training.
        """
        with torch.no_grad():
            posterior = self.vae.encode(x).latent_dist
            z = posterior.sample()
            
            # Scale latents (standard practice for Stable Diffusion VAEs)
            # This helps the diffusion model work with better-conditioned values
            z_scaled = z * self.vae.config.scaling_factor
            
            # Prepare encoder output for loss computation
            mean = posterior.mean
            logvar = torch.log(posterior.var)
            encoder_output = torch.cat([mean, logvar], dim=1)
        
        return z_scaled, encoder_output

    def decode(self, z_scaled):
        """
        Decode latent vectors back to image space (used for generation/reconstruction).
        
        Args:
            z_scaled (torch.Tensor): Scaled latents from encode() (B, 4, 32, 32)
        
        Returns:
            images (torch.Tensor): Reconstructed images (B, 3, 128, 128) in range [-1, 1]
        """
        with torch.no_grad():
            # Unscale the latents before decoding
            z = z_scaled / self.vae.config.scaling_factor
            images = self.vae.decode(z).sample
        
        return images

    def enable_finetuning(self):
        """
        Enable gradient computation for fine-tuning the VAE.
        
        By default, the VAE is frozen. Call this method if you want to
        fine-tune the pre-trained VAE on your specific dataset.
        
        Note: Fine-tuning typically only requires a few epochs and should
              use a low learning rate (1e-6 to 1e-5).
        """
        self.vae.train()
        for param in self.vae.parameters():
            param.requires_grad = True
        print("VAE fine-tuning enabled. Gradients will be computed.")

    def disable_finetuning(self):
        """
        Disable gradient computation and freeze the VAE.
        
        Use this when you want to keep the pre-trained weights fixed
        and only train the discriminator or other components.
        """
        for param in self.vae.parameters():
            param.requires_grad = False
        self.vae.eval()
        print("VAE frozen. Gradients disabled.")

    @property
    def config(self):
        """Access the VAE configuration (includes scaling_factor and other settings)."""
        return self.vae.config

    @property
    def scaling_factor(self):
        """Get the latent scaling factor (typically 0.18215 for SD VAEs)."""
        return self.vae.config.scaling_factor
