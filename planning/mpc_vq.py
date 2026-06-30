import torch

from env.constants import NUM_ACTIONS
from world_model.vq_rollout import rollout_vq_model
from planning.scoring import score_tile_rollout

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
        return score_tile_rollout(
            rollout=rollout,
            start_tiles=start_tiles,
            gamma=self.gamma,
            collision_penalty=self.collision_penalty,
            invalid_agent_penalty_weight=self.invalid_agent_penalty,
            progress_weight=self.progress_weight,
        )
