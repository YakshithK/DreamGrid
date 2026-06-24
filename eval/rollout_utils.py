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


def rollout_model(autoencoder, dynamics, start_image, actions):
    """
    start_image: [B, 3, 80, 80]
    actions: [B, H]

    returns dict with:
        "predicted_images": [B, H, 3, 80, 80]
        "predicted_tiles": [B, H, 10, 10]
        "predicted_rewards": [B, H]
        "predicted_dones": [B, H]
        "predicted_collisions": [B, H]
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