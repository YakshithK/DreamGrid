import torch
import torch.nn.functional as F

from env.tile_palette import image_to_tile_classes, tile_classes_to_image


def copy_logits_from_tiles(tile_classes, num_classes=5, strength=8.0, agent_strength=0.0):
    """
    tile_classes: [B, 10, 10]
    returns: [B, 5, 10, 10]

    creates logits saying "predict the current tile unless the model has a reason to change it"
    """

    onehot = F.one_hot(tile_classes, num_classes=num_classes).float()
    onehot = onehot.permute(0, 3, 1, 2)

    strengths = torch.full_like(tile_classes, strength, dtype=torch.float32)
    strengths = strengths.masked_fill(tile_classes == 4, agent_strength)

    return onehot * strengths[:, None, :, :]


def build_copy_residual_tile_logits(outputs, current_tiles):
    copy_logits = copy_logits_from_tiles(current_tiles)
    return copy_logits + outputs["tile_delta_logits"]

def build_copy_residual_tile_logits_from_image(outputs, current_image):
    current_tiles = image_to_tile_classes(current_image)
    copy_logits = copy_logits_from_tiles(current_tiles)
    return copy_logits + outputs["tile_delta_logits"]


def decode_latent_prediction_to_tiles(outputs, current_image):
    current_tiles = image_to_tile_classes(current_image)
    logits = build_copy_residual_tile_logits(outputs, current_tiles)
    return logits.argmax(dim=1)


def decode_tiles_to_images(tile_classes):
    return tile_classes_to_image(tile_classes)