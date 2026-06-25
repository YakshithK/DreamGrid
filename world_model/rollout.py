import torch

from env.tile_palette import image_to_tile_classes, tile_classes_to_image
from world_model.decoder import build_copy_residual_tile_logits


def rollout_latent_model(autoencoder, dynamics, start_image, actions):
    """
    start_image: [B, 3, 80, 80]
    actions: [B, H]

    returns dict with:
        "pred_tiles": [B, H, 3, 80, 80]
        "pred_images": [B, H, 10, 10]
        "rewards": [B, H]
        "done_probs": [B, H]
        "collision_probs": [B, H]
    """

    z = autoencoder.encode(start_image)
    current_tiles = image_to_tile_classes(start_image)

    pred_tiles_list = []
    pred_images_list = []
    reward_list = []
    done_list = []
    collision_list = []

    horizon = actions.shape[1]

    for t in range(horizon):
        action_t = actions[:, t]

        outputs = dynamics(z, action_t)

        pred_logits = build_copy_residual_tile_logits(outputs, current_tiles)
        pred_tiles = pred_logits.argmax(dim=1)

        pred_images = tile_classes_to_image(pred_tiles)

        pred_tiles_list.append(pred_tiles)
        pred_images_list.append(pred_images)
        reward_list.append(outputs["reward"])
        done_list.append(torch.sigmoid(outputs["done_logit"]))
        collision_list.append(torch.sigmoid(outputs["collision_logit"]))

        z = outputs["next_z"]
        current_tiles = pred_tiles

    return {
        "pred_tiles": torch.stack(pred_tiles_list, dim=1),
        "pred_images": torch.stack(pred_images_list, dim=1),
        "rewards": torch.stack(reward_list, dim=1),
        "done_probs": torch.stack(done_list, dim=1),
        "collision_probs": torch.stack(collision_list, dim=1),
    }