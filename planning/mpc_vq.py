import torch

from env.constants import NUM_ACTIONS
from world_model.vq_rollout import rollout_vq_model

class VQMPCPlanner:
    def __init__(
            self,
            vqvae,
            dynamics,
            device,
            horizon=8,
            num_candidates=1024,
            gamma=0.95,
            collision_penalty=5.0,
            invalid_agent_penalty=5.0,
            progress_weight=0.10
    ):
        self.vqvae = vqvae
        self.dynamics = dynamics
        self.device = device
        self.horizon = horizon
        self.num_candidates = num_candidates
        self.gamma = gamma
        self.collision_penalty = collision_penalty
        self.invalid_agent_penalty = invalid_agent_penalty
        self.progress_weight = progress_weight


    def plan(self, obs_image):
        self.vqvae.eval()
        self.dynamics.eval()

        obs = torch.as_tensor(obs_image, device=self.device, dtype=torch.float32)
        
        if obs.max() > 1.0:
            obs = obs / 255.0

        obs = obs.permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            encoded = self.vqvae(obs)
            start_codes = encoded["code_ids"]
            start_tiles = encoded["tile_logits"].argmax(dim=1)

            start_batch = start_codes.repeat(self.num_candidates, 1, 1)

            candidate_actions = torch.randint(
                low=0,
                high=NUM_ACTIONS - 1,
                size=(self.num_candidates, self.horizon),
                device=self.device
            )

            rollout = rollout_vq_model(self.vqvae, self.dynamics, start_batch, candidate_actions)

            scores = self.score_rollouts(rollout, start_tiles)

        best_idx = torch.argmax(scores).item()
        best_sequence = candidate_actions[best_idx]
        best_action = int(best_sequence[0].item())

        return best_action, {
            "best_score": float(scores[best_idx].item()),
            "mean_score": float(scores.mean().item()),
            "min_score": float(scores.min().item()),
            "max_score": float(scores.max().item()),
            "best_sequence": best_sequence.detach().cpu().tolist()
        }
    
    def score_rollouts(self, rollout, start_tiles):
        rewards = rollout["rewards"]
        done_probs = rollout["done_probs"]
        collision_probs = rollout["collision_probs"]
        pred_tiles = rollout["pred_tiles"]

        batch_size, horizon = rewards.shape

        discounts = torch.tensor(
            [self.gamma ** t for t in range(horizon)],
            device=rewards.device,
            dtype=torch.float32,
        )[None, :]

        previous_not_done = 1.0 - done_probs[:, :-1]
        continuation = torch.cat(
            [
                torch.ones(batch_size, 1, device=rewards.device),
                previous_not_done,
            ],
            dim=1,
        )
        continuation = torch.cumprod(continuation, dim=1)

        agent_counts = (pred_tiles == 4).sum(dim=(2, 3)).float()
        invalid_agent = (agent_counts != 1).float()

        progress = self.goal_progress_score(pred_tiles, start_tiles)
        hazard_penalty = self.agent_on_static_tile_penalty(pred_tiles, start_tiles, tile_id=2)
        wall_penalty = self.agent_on_static_tile_penalty(pred_tiles, start_tiles, tile_id=1)

        per_step_score = (
            rewards
            + self.progress_weight * progress
            - self.collision_penalty * collision_probs
            - self.invalid_agent_penalty * invalid_agent
            - 20.0 * hazard_penalty
            - 5.0 * wall_penalty
        )

        scores = (discounts * continuation * per_step_score).sum(dim=1)
        return scores
    
    def goal_progress_score(self, pred_tiles, start_tiles):
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
            torch.full_like(progress, -20.0)
        )

        reached_goal = agent_exists & (distance == 0)
        progress += reached_goal.float() * 40.0

        return progress
    
    def agent_on_static_tile_penalty(self, pred_tiles, start_tiles, tile_id):
        """
        pred_tiles: [N, H, 10, 10]
        start_tiles: [1, 10, 10]

        Returns [N, H], where 1 means the imagined agent is standing on
        a tile that was hazard/wall/whatever in the observed map.
        """
        batch_size, horizon, height, width = pred_tiles.shape
        device = pred_tiles.device

        flat_pred = pred_tiles.view(batch_size, horizon, height * width)
        agent_mask = flat_pred == 4
        agent_exists = agent_mask.any(dim=2)
        agent_idx = agent_mask.float().argmax(dim=2)

        static_flat = start_tiles.view(-1)
        static_bad = static_flat == tile_id

        bad_at_agent = static_bad[agent_idx]
        bad_at_agent = bad_at_agent & agent_exists

        return bad_at_agent.float()