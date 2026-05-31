import torch
import torch.nn as nn
from einops import rearrange


class Attention(nn.Module):
    r"""
    Attention Module for DiT.
    This is same as VIT code and does not have any changes
    from it.
    """
    def __init__(self, config):
        """
        Initialize the Attention module.
        
        This module implements the scaled dot-product attention mechanism used in 
        Transformer models. It projects the input into Query (Q), Key (K), and Value (V)
        representations, computes attention weights, and projects the output back to 
        the original hidden dimension.
        
        Args:
            config (dict): Configuration dictionary containing:
                - 'num_heads' (int): Number of attention heads
                - 'hidden_size' (int): Dimension of input and output embeddings
                - 'head_dim' (int): Dimension of each attention head
        
        Attributes:
            n_heads (int): Number of parallel attention heads
            hidden_size (int): Original embedding dimension
            head_dim (int): Dimension per head (usually hidden_size // num_heads)
            att_dim (int): Total attention dimension = num_heads * head_dim
            qkv_proj (nn.Linear): Linear layer projecting from hidden_size to 3*att_dim
                                 Used to compute Q, K, V projections simultaneously
            output_proj (nn.Sequential): Linear layer projecting from att_dim back to hidden_size
        
        Shape:
            - Input to this module: (batch_size, num_patches, hidden_size)
            - Output from this module: (batch_size, num_patches, hidden_size)
        """
        super().__init__()
        self.n_heads = config['num_heads']
        # The main dimension of the entire Transformer (e.g., 768 in DiT-B)
        self.hidden_size = config['hidden_size']
        # The dimension each individual head will operate on (e.g., 768 // 12 = 64)
        self.head_dim = config['head_dim']

        # The total dimension used for attention (usually equals hidden_size, but defined explicitly)
        self.att_dim = self.n_heads * self.head_dim

        # QKV projection for the input
        # Instead of 3 separate linear layers for Query, Key, and Value, we use 1 massive layer.
        # It takes the input (hidden_size) and projects it to 3 times the attention dimension.
        # We will slice this into Q, K, and V during the forward pass.
        self.qkv_proj = nn.Linear(self.hidden_size, 3 * self.att_dim, bias=True)

        # A linear layer to project the concatenated attention heads back to the original hidden size.
        self.output_proj = nn.Sequential(
            nn.Linear(self.att_dim, self.hidden_size)
        )

        ############################
        # DiT Layer Initialization #
        ############################
        # Initialize the QKV weights using Xavier Uniform to keep the variance of activations 
        # stable across layers, preventing exploding/vanishing gradients early in training.
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        # Start with zero bias so the linear layer doesn't introduce immediate shift.
        nn.init.constant_(self.qkv_proj.bias, 0)
        # Same Xavier initialization for the final output projection.
        nn.init.xavier_uniform_(self.output_proj[0].weight)
        nn.init.constant_(self.output_proj[0].bias, 0)

    def forward(self, x):
        """
        Apply scaled dot-product attention to the input.
        
        This method implements the multi-head scaled dot-product attention mechanism:
        
            Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
        
        Where:
            - Q (Query): Computed from input via linear projection
            - K (Key): Computed from input via linear projection
            - V (Value): Computed from input via linear projection
            - d_k: Dimension of each head (head_dim)
            - sqrt(d_k): Scaling factor to stabilize gradients
        
        The computation is parallelized across multiple attention heads, each operating
        on a subset of the embedding dimension, which allows the model to attend to 
        different representation subspaces simultaneously.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_patches, hidden_size)
                            Typically represents patch embeddings from an image or sequence tokens
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_patches, hidden_size)
                         Contains the weighted combination of values based on attention scores
        
        Process:
        --------
        1. PROJECT TO Q, K, V:
           - Input x: (B, N, hidden_size)
           - After qkv_proj: (B, N, 3*att_dim)
           - Split into Q, K, V each of shape: (B, N, att_dim)
           
        2. RESHAPE FOR MULTI-HEAD ATTENTION:
           - Q, K, V: (B, N, num_heads*head_dim)
           - After rearrange: (B, num_heads, N, head_dim)
           - Where B=batch_size, N=num_patches, num_heads=number of attention heads
           
        3. COMPUTE ATTENTION WEIGHTS (scaled dot-product):
           - Scores = Q @ K^T: (B, num_heads, N, N)
           - Scale by 1/sqrt(head_dim) for numerical stability
           - Apply softmax across last dimension to get weights in [0, 1]
           - Formula: att = softmax(Q @ K^T / sqrt(d_k))
           
        4. APPLY ATTENTION TO VALUES:
           - Weighted values: att @ V: (B, num_heads, N, head_dim)
           - Combines value information according to attention weights
           
        5. RESHAPE AND PROJECT OUTPUT:
           - Concatenate heads: (B, num_heads, N, head_dim) -> (B, N, att_dim)
           - Project to original dimension: (B, N, att_dim) -> (B, N, hidden_size)
        
        Shape transformations:
        ----------------------
        (B, N, hidden_size) 
            ↓ [qkv_proj]
        (B, N, 3*att_dim) 
            ↓ [split]
        3 × (B, N, att_dim)
            ↓ [rearrange]
        3 × (B, num_heads, N, head_dim)
            ↓ [attention computation]
        (B, num_heads, N, head_dim)
            ↓ [rearrange back]
        (B, N, att_dim)
            ↓ [output_proj]
        (B, N, hidden_size)
        """
        #  Converting to Attention Dimension
        ######################################################
        # Batch Size x Number of Patches x Dimension
        B, N = x.shape[:2]
        # Projecting to 3*att_dim and then splitting to get q, k v(each of att_dim)
        # qkv -> Batch Size x Number of Patches x (3* Attention Dimension)
        # q(as well as k and v) -> Batch Size x Number of Patches x Attention Dimension
        q, k, v = self.qkv_proj(x).split(self.att_dim, dim=-1)
        # Batch Size x Number of Patches x Attention Dimension
        # -> Batch Size x Number of Patches x (Heads * Head Dimension)
        # -> Batch Size x Number of Patches x (Heads * Head Dimension)
        # -> Batch Size x Heads x Number of Patches x Head Dimension
        # -> B x H x N x Head Dimension
        q = rearrange(q, 'b n (n_h h_dim) -> b n_h n h_dim',
                      n_h=self.n_heads, h_dim=self.head_dim)
        k = rearrange(k, 'b n (n_h h_dim) -> b n_h n h_dim',
                      n_h=self.n_heads, h_dim=self.head_dim)
        v = rearrange(v, 'b n (n_h h_dim) -> b n_h n h_dim',
                      n_h=self.n_heads, h_dim=self.head_dim)
        #########################################################

        # Compute Attention Weights
        #########################################################
        # B x H x N x Head Dimension @ B x H x Head Dimension x N
        # -> B x H x N x N
        att = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** (-0.5))
        att = torch.nn.functional.softmax(att, dim=-1)
        #########################################################

        # Weighted Value Computation
        #########################################################
        #  B x H x N x N @ B x H x N x Head Dimension
        # -> B x H x N x Head Dimension
        out = torch.matmul(att, v)
        #########################################################

        # Converting to Transformer Dimension
        #########################################################
        # B x N x (Heads * Head Dimension) -> B x N x (Attention Dimension)
        out = rearrange(out, 'b n_h n h_dim -> b n (n_h h_dim)')
        #  B x N x Dimension
        out = self.output_proj(out)
        ##########################################################

        return out
