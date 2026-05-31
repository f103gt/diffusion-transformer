"""
Standalone utility functions shared across model components.

Functions here have no dependencies on specific model classes and are
kept separate so they can be reused without importing the classes that
happen to use them.
"""
import torch


def get_time_embedding(time_steps, temb_dim):
    r"""
    Build a sinusoidal time-step embedding vector for each timestep in the batch.

    Converts integer timestep indices into dense float vectors using sine and cosine
    functions at geometrically-spaced frequencies, following the positional encoding
    formulation from "Attention Is All You Need".

    :param time_steps: 1-D integer tensor of shape (B,) containing timestep indices.
    :param temb_dim: Dimensionality of the output embedding. Must be even.
    :return: Float tensor of shape (B, temb_dim).
    """
    assert temb_dim % 2 == 0, "time embedding dimension must be divisible by 2"

    # frequency = 10000^(2i / temb_dim) for i in [0, temb_dim // 2)
    freq = 10000 ** (
        torch.arange(
            start=0,
            end=temb_dim // 2,
            dtype=torch.float32,
            device=time_steps.device,
        )
        / (temb_dim // 2)
    )

    # (B,) -> (B, temb_dim // 2), each row divided element-wise by the frequencies
    t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / freq
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb


def get_patch_position_embedding(pos_emb_dim, grid_size, device):
    r"""
    Build a 2-D sinusoidal position embedding for a grid of image patches.

    Produces one embedding vector per patch by concatenating independent
    sinusoidal encodings of the patch's row index and column index.

    :param pos_emb_dim: Total embedding dimension. Must be divisible by 4 so that
        each of the two spatial axes gets an equal-length ``pos_emb_dim // 2``
        sub-embedding.
    :param grid_size: ``(grid_height, grid_width)`` — number of patches along
        each spatial axis.
    :param device: Torch device on which to create the tensors.
    :return: Float tensor of shape ``(grid_height * grid_width, pos_emb_dim)``.
    """
    assert pos_emb_dim % 4 == 0, "Position embedding dimension must be divisible by 4"
    grid_height, grid_width = grid_size

    row_indices = torch.arange(grid_height, dtype=torch.float32, device=device)
    col_indices = torch.arange(grid_width, dtype=torch.float32, device=device)
    grid = torch.meshgrid(row_indices, col_indices, indexing='ij')
    grid = torch.stack(grid, dim=0)

    # Flatten spatial dims: (grid_height * grid_width,)
    row_positions = grid[0].reshape(-1)
    col_positions = grid[1].reshape(-1)

    # frequency = 10000^(2i / (pos_emb_dim // 4)) for i in [0, pos_emb_dim // 4)
    freq = 10000 ** (
        torch.arange(
            start=0,
            end=pos_emb_dim // 4,
            dtype=torch.float32,
            device=device,
        )
        / (pos_emb_dim // 4)
    )

    row_emb = row_positions[:, None].repeat(1, pos_emb_dim // 4) / freq
    row_emb = torch.cat([torch.sin(row_emb), torch.cos(row_emb)], dim=-1)
    # row_emb -> (num_patches, pos_emb_dim // 2)

    col_emb = col_positions[:, None].repeat(1, pos_emb_dim // 4) / freq
    col_emb = torch.cat([torch.sin(col_emb), torch.cos(col_emb)], dim=-1)

    # pos_emb -> (num_patches, pos_emb_dim)
    return torch.cat([row_emb, col_emb], dim=-1)


def spatial_average(feature_map, keepdim=True):
    r"""
    Average a 4-D feature map over its spatial (height and width) dimensions.

    :param feature_map: Float tensor of shape (B, C, H, W).
    :param keepdim: If ``True`` (default) the output shape is (B, C, 1, 1),
        otherwise (B, C).
    :return: Spatially averaged tensor.
    """
    return feature_map.mean([2, 3], keepdim=keepdim)
