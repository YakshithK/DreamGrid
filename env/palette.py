import torch

PALETTE_RGB = torch.tensor(
    [
        [235, 235, 225],
        [45, 45, 50],
        [220, 55, 55],
        [50, 180, 90],
        [45, 105, 230],
    ],
    dtype=torch.float32,
) / 255.0

def rgb_to_palette_indices(image):
    """
    image shape: [B, 3, H, W], values in [0, 1]
    returns: [B, H, W], values in class IDs
    """

    palette = PALETTE_RGB.to(image.device)

    image_hw = image.permute(0, 2, 3, 1)  # [B, H, W, 3]
    diff = image_hw[:, :, :, None, :] - palette[None, None, None, :, :]
    dist = (diff ** 2).sum(dim=-1)  # [B, H, W, 5]

    return dist.argmin(dim=-1)

def palette_indices_to_rgb(indices):
    """
    indices shapes: [B, H, W], values in class IDs
    returns: [B, 3, H, W], values in [0, 1]
    """
    palette = PALETTE_RGB.to(indices.device)
    return palette[indices].permute(0, 3, 1, 2)