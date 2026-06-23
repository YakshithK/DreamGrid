import torch

from env.constants import GRID_SIZE, TILE_SIZE

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

def image_to_tile_classes(image):
    """ 
    convert rendered RGB images into 10x10 visual tile classes
    
    image: [B, 3, 80, 80], values in [0, 1]
    returns: [B, 10, 10], values in class IDs
    
    """

    b, c, h, w = image.shape

    tiles = image.view(
        b,
        c, 
        GRID_SIZE,
        TILE_SIZE,
        GRID_SIZE,
        TILE_SIZE
    )

    tiles = tiles.permute(0, 2, 4, 1, 3, 5)
    tile_rgb = tiles.mean(dim=(-1, -2))

    palette = PALETTE_RGB.to(image.device)
    diff = tile_rgb[:, :, :, None, :] - palette[None, None, None, :, :]
    dist = (diff ** 2).sum(dim=-1)

    return dist.argmin(dim=-1)

def tile_classes_to_image(tile_classes):
    """
    convert 10x10 visual tile classes into rendered RGB images
    
    tile_classes: [B, 10, 10], values in class IDs
    returns: [B, 3, 80, 80], values in [0, 1]
    
    """

    palette = PALETTE_RGB.to(tile_classes.device)

    tile_rgb = palette[tile_classes]
    image = tile_rgb.repeat_interleave(TILE_SIZE, dim=-2).repeat_interleave(TILE_SIZE, dim=-1)

    return image.permute(0, 3, 1, 2)