import torch.nn as nn


class DownBlock(nn.Module):
    r"""
    Encoder building block combining residual convolutions with optional
    self-attention and cross-attention, followed by an optional spatial
    downsampling step.

    Processing order per layer:
        1. ResNet-style residual convolution (with optional time-embedding injection).
        2. Self-attention (optional).
        3. Cross-attention conditioned on an external context tensor (optional).
    After all layers: strided convolution downsampling (optional).
    """

    def __init__(self, in_channels, out_channels, t_emb_dim,
                 down_sample, num_heads, num_layers, attn, norm_channels,
                 cross_attn=False, context_dim=None):
        """
        Initialize the DownBlock.

        Args:
            in_channels (int): Number of input feature channels.
            out_channels (int): Number of output feature channels.
            t_emb_dim (int or None): Dimension of the timestep embedding. If None, time conditioning is skipped.
            down_sample (bool): If True, applies a strided convolution at the end to halve spatial dimensions.
            num_heads (int): Number of attention heads for both self and cross-attention.
            num_layers (int): Number of repeated ResNet (+ Attention) sub-blocks within this DownBlock.
            attn (bool): If True, applies Self-Attention after each ResNet block.
            norm_channels (int): Number of groups to use for Group Normalization.
            cross_attn (bool): If True, applies Cross-Attention after Self-Attention.
            context_dim (int or None): Dimension of the external context sequence (e.g., CLIP text embeddings). 
                                       Required if cross_attn is True.

        Attributes:
            resnet_conv_first (nn.ModuleList): First half of the ResNet block (Norm -> SiLU -> Conv).
            t_emb_layers (nn.ModuleList): Projects timestep embeddings to match out_channels.
            resnet_conv_second (nn.ModuleList): Second half of the ResNet block (Norm -> SiLU -> Conv).
            residual_input_conv (nn.ModuleList): 1x1 Conv to match channel dimensions for the residual connection.
            attentions, attention_norms (nn.ModuleList): Self-Attention layers and their pre-norms.
            cross_attentions, cross_attention_norms, context_proj (nn.ModuleList): Cross-Attention layers, norms, 
                                                                                   and context projection matrices.
            down_sample_conv (nn.Module): Strided convolution to halve spatial dimensions if down_sample is True.
        """
        super().__init__()
        self.num_layers = num_layers
        self.down_sample = down_sample
        self.attn = attn
        self.context_dim = context_dim
        self.cross_attn = cross_attn
        self.t_emb_dim = t_emb_dim
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels,
                              kernel_size=3, stride=1, padding=1),
                )
                for i in range(num_layers)
            ]
        )
        if self.t_emb_dim is not None:
            self.t_emb_layers = nn.ModuleList([
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(self.t_emb_dim, out_channels)
                )
                for _ in range(num_layers)
            ])
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels,
                              kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers)
            ]
        )

        if self.attn:
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
                for i in range(num_layers)
            ]
        )
        self.down_sample_conv = nn.Conv2d(out_channels, out_channels,
                                          4, 2, 1) if self.down_sample else nn.Identity()

    def forward(self, x, t_emb=None, context=None):
        """
        Forward pass for the DownBlock.

        Process:
        --------
        For each of the `num_layers`:
        1. RESNET BLOCK:
           - Pass input through GroupNorm, SiLU, and a 3x3 Conv.
           - Add the timestep embedding (broadcasted spatially).
           - Pass through a second GroupNorm, SiLU, and 3x3 Conv.
           - Add the original input (projected via 1x1 Conv if channels changed) as a residual connection.

        2. SELF-ATTENTION (Optional):
           - Flatten the spatial dimensions (H, W) into a 1D sequence of length N = H * W.
           - Apply GroupNorm.
           - Compute Multi-Head Self-Attention where Query, Key, and Value are all the image sequence.
           - Add to the feature map as a residual connection.

        3. CROSS-ATTENTION (Optional):
           - Flatten the spatial dimensions again.
           - Apply GroupNorm.
           - Project the external conditioning `context` to match the channel dimension.
           - Compute Multi-Head Cross-Attention where Query = Image Sequence, Key = Value = Context.
           - Add to the feature map as a residual connection.

        4. DOWNSAMPLE (Optional):
           - Apply a strided Convolution to halve the Height and Width.

        Shape transformations:
        ----------------------
        B = Batch Size, C = Channels, H = Height, W = Width, L = Context Length
        
        Input: 
            x: (B, C_in, H, W)
            t_emb: (B, t_emb_dim)
            context: (B, L, context_dim)
            
        Inside Loop:
            ResNet Output: (B, C_out, H, W)
            Flattened for Attn: (B, H*W, C_out)
            After Attn: (B, C_out, H, W)
            
        Output:
            If down_sample == True: (B, C_out, H/2, W/2)
            If down_sample == False: (B, C_out, H, W)
        """
        out = x
        for i in range(self.num_layers):
            # Residual convolution block
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            if self.t_emb_dim is not None:
                out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)

            if self.attn:
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

        out = self.down_sample_conv(out)
        return out
