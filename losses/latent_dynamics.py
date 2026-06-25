import torch
import torch.nn.functional as F

from env.tile_palette import image_to_tile_classes
from world_model.decoder import build_copy_residual_tile_logits_from_image

def compute_latent_dynamics_loss(outputs, current_image, target_image, target_z, reward, done, collision):
    latent_loss = F.mse_loss(outputs["next_z"], target_z)

    current_tiles = image_to_tile_classes(current_image)
    target_tiles = image_to_tile_classes(target_image)

    pred_tile_logits = build_copy_residual_tile_logits_from_image(outputs, current_image)

    per_tile_ce = F.cross_entropy(pred_tile_logits, target_tiles, reduction="none")

    class_weights = torch.tensor(
        [1.0, 5.0, 15.0, 15.0, 25.0],
        device=current_image.device,
    )

    tile_weights = class_weights[target_tiles]

    static_important = (current_tiles == target_tiles) & (target_tiles != 0)
    changed = current_tiles != target_tiles

    agent_changed = changed & ((current_tiles == 4) | (target_tiles == 4))

    tile_weights += static_important.float() * 30.0
    tile_weights += changed.float() * 20.0
    tile_weights += agent_changed.float() * 40.0

    tile_loss = (per_tile_ce * tile_weights).sum() / tile_weights.sum()

    static_residual_penalty = (
        outputs["tile_delta_logits"]
        .permute(0, 2, 3, 1)[static_important]
        .pow(2)
        .mean()
    )

    if torch.isnan(static_residual_penalty):
        static_residual_penalty = torch.tensor(0.0, device=target_image.device)

    reward_loss = F.smooth_l1_loss(outputs["reward"], reward)

    done_loss = F.binary_cross_entropy_with_logits(
        outputs["done_logit"],
        done
    )

    collision_loss = F.binary_cross_entropy_with_logits(
        outputs["collision_logit"],
        collision
    )

    delta_reg = outputs["delta_z"].pow(2).mean()

    total = (
        0.10 * latent_loss
        + 4.00 * tile_loss
        + 0.25 * reward_loss
        + 0.25 * done_loss
        + 0.50 * collision_loss
        + 0.01 * delta_reg
        + 0.05 * static_residual_penalty
    )

    return total, {
        "latent_loss": latent_loss.item(),
        "tile_loss": tile_loss.item(),
        "reward_loss": reward_loss.item(),
        "done_loss": done_loss.item(),
        "collision_loss": collision_loss.item(),
        "delta_reg": delta_reg.item(),
        "static_residual_penalty": static_residual_penalty.item(),
    }
