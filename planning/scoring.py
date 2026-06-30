import torch

def discounted_scores(per_step_score, done_probs, gamma):
    batch_size, horizon = per_step_score.shape

    discounts = torch.tensor(
        [gamma ** t for t in range(horizon)],
        device=per_step_score.device,
        dtype=torch.float32
    )[None, :]

    previous_not_done = 1.0 - done_probs[:, :-1]
    continuation = torch.cat(
        [torch.ones(batch_size, 1, device=per_step_score.device), previous_not_done],
        dim=1,
    )
    continuation = torch.cumprod(continuation, dim=1)

    return (discounts * continuation * per_step_score).sum(dim=1)


def invalid_agent_penalty(pred_tiles):
    agent_counts = (pred_tiles == 4).sum(dim=(2, 3)).float()
    return (agent_counts != 1).float()


def goal_progress_score(pred_tiles, start_tiles):
    batch_size, horizon, height, width = pred_tiles.shape
    device = pred_tiles.device

    start_flat = start_tiles.view(1, -1)
    goal_mask = start_flat == 3

    if not goal_mask.any():
        return torch.zeros(batch_size, horizon, device=device)

    goal_idx = goal_mask.float().argmax(dim=1)[0]
    goal_r = goal_idx // width
    goal_c = goal_idx % width

    flat = pred_tiles.view(batch_size, horizon, height * width)
    agent_mask = flat == 4
    agent_exists = agent_mask.any(dim=2)
    agent_idx = agent_mask.float().argmax(dim=2)

    idxs = torch.arange(height * width, device=device)
    row = idxs // width
    col = idxs % width

    agent_r = row[agent_idx]
    agent_c = col[agent_idx]

    distance = (agent_r - goal_r).abs() + (agent_c - goal_c).abs()
    distance = distance.float()

    progress = -distance
    progress = torch.where(
        agent_exists,
        progress,
        torch.full_like(progress, -20.0),
    )

    reached_goal = agent_exists & (distance == 0)
    progress = progress + reached_goal.float() * 40.0

    return progress

def agent_on_static_tile_penalty(pred_tiles, start_tiles, tile_id):
    """
    pred_tiles: [N, H, 10, 10]
    start_tiles: [1, 10, 10]

    Returns [N, H], where 1 means the imagined agent is standing on
    a tile that was hazard/wall/whatever in the observed map.
    """
    batch_size, horizon, height, width = pred_tiles.shape

    flat_pred = pred_tiles.view(batch_size, horizon, height * width)
    agent_mask = flat_pred == 4
    agent_exists = agent_mask.any(dim=2)
    agent_idx = agent_mask.float().argmax(dim=2)

    static_flat = start_tiles.view(-1)
    static_bad = static_flat == tile_id

    bad_at_agent = static_bad[agent_idx]
    bad_at_agent = bad_at_agent & agent_exists

    return bad_at_agent.float()


def score_tile_rollout(
        rollout,
        start_tiles,
        gamma=0.95,
        collision_penalty=5.0,
        invalid_agent_penalty_weight=5.0,
        progress_weight=0.10,
        hazard_penalty_weight=20.0,
        wall_penalty_weight=5.0
):
    rewards = rollout["rewards"]
    done_probs = rollout["done_probs"]
    collision_probs = rollout["collision_probs"]
    pred_tiles = rollout["pred_tiles"]

    invalid_agent = invalid_agent_penalty(pred_tiles)
    progress = goal_progress_score(pred_tiles, start_tiles)
    hazard_penalty = agent_on_static_tile_penalty(pred_tiles, start_tiles, tile_id=2)
    wall_penalty = agent_on_static_tile_penalty(pred_tiles, start_tiles, tile_id=1)

    per_step_score = (
        rewards
        + progress_weight * progress
        - collision_penalty * collision_probs
        - invalid_agent_penalty_weight * invalid_agent
        - hazard_penalty_weight * hazard_penalty
        - wall_penalty_weight * wall_penalty
    )

    return discounted_scores(per_step_score, done_probs, gamma)