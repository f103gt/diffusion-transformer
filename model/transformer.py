import torch
import torch.nn as nn
from model.patch_embed import PatchEmbedding
from model.transformer_layer import TransformerLayer
from model.helpers import get_time_embedding
from einops import rearrange

class LabelEmbedder(nn.Module):
    """
    Embeds discrete class labels into continuous vector representations.
    
    This module is crucial for conditional generation. It maps integer class labels 
    into high-dimensional vectors that the Transformer can understand. Additionally, 
    it natively supports Classifier-Free Guidance (CFG) by randomly dropping a 
    percentage of the labels during training and replacing them with a learned 
    "unconditional" or "null" token.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        """
        Initialize the Label Embedder.

        :param num_classes: Int. Total number of valid classes in the dataset.
        :param hidden_size: Int. The target dimensionality of the label embeddings 
            (must match the transformer's embedding dimension).
        :param dropout_prob: Float (0.0 to 1.0). The probability of dropping a label 
            during training to enable Classifier-Free Guidance.
        """
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Randomly replaces class labels with the unconditional "null" token.

        During training, the model must learn to generate images both conditionally 
        (with a specific label) and unconditionally (without a label) to allow 
        CFG to extrapolate between the two states during inference.

        :param labels: Integer tensor of shape ``(B,)`` containing batch labels.
        :param force_drop_ids: Optional boolean/integer tensor of shape ``(B,)``. 
            If provided, explicitly forces these specific indices to be dropped.
        :return: Integer tensor of shape ``(B,)`` where dropped elements are 
            replaced by the `self.num_classes` index.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        """
        Forward pass to generate the conditional label embeddings.

        :param labels: Integer tensor of shape ``(B,)`` containing the class indices.
        :param train: Boolean. Indicates if the model is in training mode. Label 
            dropout is only applied when `train == True` or `force_drop_ids` is passed.
        :param force_drop_ids: Optional tensor of shape ``(B,)`` to force unconditional 
            embeddings for specific items in the batch.
        :return: Float tensor of shape ``(B, hidden_size)`` containing the label embeddings.
        """
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

class DIT(nn.Module):
    """
    Diffusion Transformer (DiT) base model.

    Unlike standard diffusion models (like DDPMs) that use U-Net architectures 
    based on convolutional layers, DiT operates entirely using a Transformer 
    architecture. It treats the noisy input images (or latents) as a sequence 
    of patches.
    """
    def __init__(self, im_size, im_channels, config):
        """
        Initialize the complete Diffusion Transformer network.

        :param im_size: Int. Spatial height/width of the input image or latent tensor.
        :param im_channels: Int. Number of channels in the input tensor.
        :param config: Dictionary containing architecture hyperparameters:
            - 'num_layers': Number of Transformer blocks.
            - 'hidden_size': Dimensionality of the model (C).
            - 'patch_size': Size of the non-overlapping patches to extract.
            - 'timestep_emb_dim': Dimension of the initial sinusoidal time embedding.
            - 'num_classes': Number of class conditions.
            - 'class_dropout_prob': Probability for classifier-free guidance dropout.
        """
        super().__init__()

        num_layers = config['num_layers']
        self.image_height = im_size
        self.image_width = im_size
        self.im_channels = im_channels
        self.hidden_size = config['hidden_size']
        self.patch_height = config['patch_size']
        self.patch_width = config['patch_size']

        self.timestep_emb_dim = config['timestep_emb_dim']
        self.num_classes = config['num_classes']
        self.class_dropout_prob = config['class_dropout_prob']

        # Number of patches along height and width
        self.nh = self.image_height // self.patch_height
        self.nw = self.image_width // self.patch_width

        # Patch Embedding Block
        self.patch_embed_layer = PatchEmbedding(image_height=self.image_height,
                                                image_width=self.image_width,
                                                im_channels=self.im_channels,
                                                patch_height=self.patch_height,
                                                patch_width=self.patch_width,
                                                hidden_size=self.hidden_size)
        
        self.y_embedder = LabelEmbedder(self.num_classes, self.hidden_size, self.class_dropout_prob)

        # Initial projection from sinusoidal time embedding
        self.t_proj = nn.Sequential(
            nn.Linear(self.timestep_emb_dim, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )

        # All Transformer Layers
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(num_layers)
        ])

        # Final normalization for unpatchify block
        self.norm = nn.LayerNorm(self.hidden_size, elementwise_affine=False, eps=1E-6)

        # Scale and Shift parameters for the norm
        self.adaptive_norm_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, 2 * self.hidden_size, bias=True)
        )

        # Final Linear Layer
        self.proj_out = nn.Linear(self.hidden_size,
                                  self.patch_height * self.patch_width * self.im_channels)


        # DiT Layer Initialization
        nn.init.normal_(self.t_proj[0].weight, std=0.02)
        nn.init.normal_(self.t_proj[2].weight, std=0.02)

        nn.init.constant_(self.adaptive_norm_layer[-1].weight, 0)
        nn.init.constant_(self.adaptive_norm_layer[-1].bias, 0)

        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

    def forward(self, x, t, y):
        """
        Main Forward Pass of the Diffusion Transformer.

        What it is designed for:
        The forward pass acts as the core "denoising" or "score-prediction" step in 
        the diffusion process. Given an image `x` that has been corrupted by a 
        specific level of noise `t`, and a class condition `y`, the network attempts 
        to predict the exact noise that was added (or the clean image, depending on 
        the specific loss formulation). This allows the reverse diffusion process to 
        iteratively subtract this predicted noise to generate a new image from scratch.

        :param x: Float tensor of shape ``(B, C, H, W)``. The noisy input image/latents.
        :param t: Integer/Float tensor of shape ``(B,)``. The diffusion timestep or 
            noise level corresponding to `x`.
        :param y: Integer tensor of shape ``(B,)``. The class label indices.
        :return: Float tensor of shape ``(B, C, H, W)``. The un-patchified, full-resolution 
            prediction (typically the predicted noise).
        """
        # Patchify
        out = self.patch_embed_layer(x)

        # Compute Timestep representation
        # t_emb -> (Batch, timestep_emb_dim)
        t_emb = get_time_embedding(torch.as_tensor(t).long(), self.timestep_emb_dim)
        # (Batch, timestep_emb_dim) -> (Batch, hidden_size)
        t_emb = self.t_proj(t_emb)

        # Embed class labels
        y_emb = self.y_embedder(y, self.training)    # (N, D)
        t_emb = t_emb + y_emb

        # Go through the transformer layers
        for layer in self.layers:
            out = layer(out, t_emb)

        # Shift and scale predictions for output normalization
        pre_mlp_shift, pre_mlp_scale = self.adaptive_norm_layer(t_emb).chunk(2, dim=1)
        out = (self.norm(out) * (1 + pre_mlp_scale.unsqueeze(1)) +
               pre_mlp_shift.unsqueeze(1))

        # Unpatchify
        # (B,patches,hidden_size) -> (B,patches,channels * patch_width * patch_height)
        out = self.proj_out(out)
        out = rearrange(out, 'b (nh nw) (ph pw c) -> b c (nh ph) (nw pw)',
                        ph=self.patch_height,
                        pw=self.patch_width,
                        nw=self.nw,
                        nh=self.nh)
        return out
