import torch


def rollout_vq_model(vqvae, dynamics, start_codes, actions):
    """
    start_codes: [B, 10, 10]
    actions: [B, H]

    returns:
        pred_codes: [B, H, 10, 10]
        pred_tiles: [B, H, 10, 10]
        rewards: [B, H]
        done_probs: [B, H]
        collision_probs: [B, H]
    """

    current_codes = start_codes

    pred_codes_list = []
    pred_tiles_list = []
    rewards_list = []
    done_list = []
    collision_list = []

    horizon = actions.shape[1]

    for t in range(horizon):
        action_t = actions[:, t]

        outputs = dynamics(current_codes, action_t)

        next_codes = outputs["next_code_logits"].argmax(dim=1)

        _, tile_logits = vqvae.decode_codes(next_codes)
        pred_tiles = tile_logits.argmax(dim=1)

        pred_codes_list.append(next_codes)
        pred_tiles_list.append(pred_tiles)
        rewards_list.append(outputs["reward"])
        done_list.append(torch.sigmoid(outputs["done_logit"]))
        collision_list.append(torch.sigmoid(outputs["collision_logit"]))

        current_codes = next_codes

    return {
        "pred_codes": torch.stack(pred_codes_list, dim=1),
        "pred_tiles": torch.stack(pred_tiles_list, dim=1),
        "rewards": torch.stack(rewards_list, dim=1),
        "done_probs": torch.stack(done_list, dim=1),
        "collision_probs": torch.stack(collision_list, dim=1)
    }