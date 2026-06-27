import torch


def rollout_spatial_model(model, start_tiles, actions):
    """
    model: SpatialDynamicsModel
    start_tiles: [B, 10, 10]
    actions: [B, H]

    returns:
        pred_tiles: [B, H, 10, 10]
        rewards: [B, H]
        done_probs: [B, H]
        collision_probs: [B, H]

    """
    current_tiles = start_tiles

    pred_tiles_list= []
    reward_list = []
    done_list = []
    collision_list = []

    horizon = actions.shape[1]

    for t in range(horizon):
        action_t = actions[:, t]

        outputs= model(current_tiles, action_t)

        next_tiles = outputs["next_tile_logits"].argmax(dim=1)

        pred_tiles_list.append(next_tiles)
        reward_list.append(outputs["reward"])
        done_list.append(torch.sigmoid(outputs["done_logit"]))
        collision_list.append(torch.sigmoid(outputs["collision_logit"]))

        current_tiles = next_tiles
    
    return {
        "pred_tiles": torch.stack(pred_tiles_list, dim=1),
        "rewards": torch.stack(reward_list, dim=1),
        "done_probs": torch.stack(done_list, dim=1),
        "collision_probs": torch.stack(collision_list, dim=1)
    }