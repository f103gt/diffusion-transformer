import torch
import torch.nn as nn
from model.patch_embed import PatchEmbedding
from model.transformer_layer import TransformerLayer
from model.helpers import get_time_embedding
from einops import rearrange

class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. 
    Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

class DIT(nn.Module):
    def __init__(self, im_size, im_channels, config):
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

        ############################
        # DiT Layer Initialization #
        ############################
        nn.init.normal_(self.t_proj[0].weight, std=0.02)
        nn.init.normal_(self.t_proj[2].weight, std=0.02)

        nn.init.constant_(self.adaptive_norm_layer[-1].weight, 0)
        nn.init.constant_(self.adaptive_norm_layer[-1].bias, 0)

        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

    def forward(self, x, t, y):
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
