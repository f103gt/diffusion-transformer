import torch.nn as nn


class MidBlock(nn.Module):
    r"""
    Bottleneck building block placed at the deepest point between the encoder
    and decoder paths of a U-Net or VAE architecture.

    Why it exists:
    At the absolute bottom of the network, spatial dimensions (Height and Width) 
    are at their most compressed, meaning each "pixel" in the feature map contains 
    highly semantic, global information. Because the spatial resolution is tiny 
    here, it is computationally feasible to use Attention mechanisms. The MidBlock's 
    job is to process this dense representation, allowing distant parts of the image 
    to communicate with each other globally (via self-attention) and to incorporate 
    external conditioning like text embeddings (via cross-attention) before the 
    feature map is handed off to the decoder for upsampling.

    Processing order:
        1. Initial ResNet-style residual convolution.
        For each of num_layers iterations:
            2. Self-attention: Global spatial mixing of features.
            3. Cross-attention (optional): Conditioned on an external context tensor.
            4. ResNet-style residual convolution: Local feature refinement.
    """
    def __init__(self, in_channels, out_channels, t_emb_dim, num_heads, num_layers,
                 norm_channels, cross_attn=None, context_dim=None):
        r"""
        Initializes the MidBlock components.

        :param in_channels: Int. Number of channels in the incoming feature map.
        :param out_channels: Int. Number of channels for the output and internal 
            feature maps.
        :param t_emb_dim: Int or None. Dimensionality of the timestep embedding. 
            Used to condition the ResNet blocks on the current diffusion noise level.
        :param num_heads: Int. Number of attention heads for the Multi-Head 
            Attention layers.
        :param num_layers: Int. Number of times to repeat the core sequence of 
            (Self-Attn -> Cross-Attn -> ResNet).
        :param norm_channels: Int. Number of groups to use for Group Normalization 
            (standard for diffusion models as it is batch-size independent).
        :param cross_attn: Bool. If True, enables Cross-Attention layers to condition 
            the features on external context (e.g., text prompts).
        :param context_dim: Int or None. Dimensionality of the external context tensor. 
            Must be provided if `cross_attn` is True.
        """
        super().__init__()
        self.num_layers = num_layers
        self.t_emb_dim = t_emb_dim
        self.context_dim = context_dim
        self.cross_attn = cross_attn
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels,
                              kernel_size=3, stride=1, padding=1),
                )
                for i in range(num_layers + 1)
            ]
        )

        if self.t_emb_dim is not None:
            self.t_emb_layers = nn.ModuleList([
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(t_emb_dim, out_channels)
                )
                for _ in range(num_layers + 1)
            ])
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers + 1)
            ]
        )

        self.attention_norms = nn.ModuleList(
            [nn.GroupNorm(norm_channels, out_channels)
             for _ in range(num_layers)]
        )

        self.attentions = nn.ModuleList(
            [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
             for _ in range(num_layers)]
        )
        if self.cross_attn:
            assert context_dim is not None, "Context Dimension must be passed for cross attention"
            self.cross_attention_norms = nn.ModuleList(
                [nn.GroupNorm(norm_channels, out_channels)
                 for _ in range(num_layers)]
            )
            self.cross_attentions = nn.ModuleList(
                [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                 for _ in range(num_layers)]
            )
            self.context_proj = nn.ModuleList(
                [nn.Linear(context_dim, out_channels)
                 for _ in range(num_layers)]
            )
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers + 1)
            ]
        )

    def forward(self, x, t_emb=None, context=None):
        r"""
        Forward pass of the MidBlock.

        What it is designed for:
        This function takes the heavily downsampled spatial features from the encoder, 
        injects time-step conditioning (to tell the network what the current noise level is), 
        allows spatial elements to cross-pollinate information (self-attention), injects 
        external conditions (cross-attention), and outputs a highly refined feature map 
        ready to be decoded.

        :param x: Float tensor of shape ``(B, in_channels, H, W)``. The incoming feature map 
            from the final layer of the downsampling (encoder) path.
        :param t_emb: Float tensor of shape ``(B, t_emb_dim)`` or None. The time-step 
            embedding representing the current stage of diffusion.
        :param context: Float tensor of shape ``(B, seq_len, context_dim)`` or None. The 
            external conditioning context (e.g., text prompt embeddings).
        :return: Float tensor of shape ``(B, out_channels, H, W)``. The processed feature 
            map ready for the upsampling (decoder) path.
        """
        out = x

        # First residual convolution block
        resnet_input = out
        out = self.resnet_conv_first[0](out)
        if self.t_emb_dim is not None:
            out = out + self.t_emb_layers[0](t_emb)[:, :, None, None]
        out = self.resnet_conv_second[0](out)
        out = out + self.residual_input_conv[0](resnet_input)

        for i in range(self.num_layers):
            # Self-attention block
            batch_size, channels, h, w = out.shape
            in_attn = out.reshape(batch_size, channels, h * w)
            in_attn = self.attention_norms[i](in_attn)
            in_attn = in_attn.transpose(1, 2)
            out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
            out = out + out_attn

            if self.cross_attn:
                assert context is not None, "context cannot be None if cross attention layers are used"
                batch_size, channels, h, w = out.shape
                in_attn = out.reshape(batch_size, channels, h * w)
                in_attn = self.cross_attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                assert context.shape[0] == x.shape[0] and context.shape[-1] == self.context_dim
                context_proj = self.context_proj[i](context)
                out_attn, _ = self.cross_attentions[i](in_attn, context_proj, context_proj)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
                out = out + out_attn

            # Residual convolution block
            resnet_input = out
            out = self.resnet_conv_first[i + 1](out)
            if self.t_emb_dim is not None:
                out = out + self.t_emb_layers[i + 1](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i + 1](out)
            out = out + self.residual_input_conv[i + 1](resnet_input)

        return out
