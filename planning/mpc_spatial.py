import torch

from env.constants import NUM_ACTIONS
from env.tile_palette import image_to_tile_classes
from world_model.spatial_rollout import rollout_spatial_model
from planning.scoring import score_tile_rollout

class SpatialMPCPlanner:
    def __init__(
            self,
            model,
            device,
            horizon=5,
            num_candidates=512,
            gamma=0.95,
            collision_penalty=5.0,
            invalid_agent_penalty=5.0,
            progress_weight=0.10
    ):
        self.model = model
        self.device = device
        self.horizon = horizon
        self.num_candidates = num_candidates
        self.gamma = gamma
        self.collision_penalty = collision_penalty
        self.invalid_agent_penalty = invalid_agent_penalty
        self.progress_weight = progress_weight


    def plan(self, obs_image):
        self.model.eval()

        obs = torch.as_tensor(obs_image, device=self.device, dtype=torch.float32)
        if obs.max() > 1.0:
            obs = obs / 255.0

        obs = obs.permute(2, 0, 1).unsqueeze(0)
        start_tiles =image_to_tile_classes(obs)

        start_batch = start_tiles.repeat(self.num_candidates, 1, 1)

        candidate_actions = torch.randint(
            low=0,
            high=NUM_ACTIONS -1,
            size=(self.num_candidates, self.horizon),
            device=self.device
        )

        with torch.no_grad():
            rollout = rollout_spatial_model(self.model, start_batch, candidate_actions)

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
        return score_tile_rollout(
            rollout=rollout,
            start_tiles=start_tiles,
            gamma=self.gamma,
            collision_penalty=self.collision_penalty,
            invalid_agent_penalty_weight=self.invalid_agent_penalty,
            progress_weight=self.progress_weight,
        )