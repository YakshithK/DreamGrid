import torch

from env.constants import NUM_ACTIONS
from world_model.rollout import rollout_latent_model

class LatentMPCPlanner:
    def __init__(
            self,
            autoencoder,
            dynamics,
            device,
            horizon=2,
            num_candidates=128,
            gamma=0.95,
            collision_penalty=0.5,
            invalid_agent_penalty=0.25,
    ):
        self.autoencoder = autoencoder
        self.dynamics = dynamics
        self.device = device
        self.horizon = horizon
        self.num_candidates = num_candidates
        self.gamma = gamma
        self.collision_penalty = collision_penalty
        self.invalid_agent_penalty = invalid_agent_penalty

    def plan(self, obs_image):
        """
        obs_image: numpy array [80, 80, 3], uint8 or float image.

        returns:
        action: int, the best action to take
        info: dict with chosen sequence and score details
        """
        self.autoencoder.eval()
        self.dynamics.eval()

        obs = torch.as_tensor(obs_image, device=self.device, dtype=torch.float32)

        if obs.max() > 1.0:
            obs = obs / 255.0

        obs = obs.permute(2, 0, 1).unsqueeze(0)  # [1, 3, 80, 80]
        obs_batch = obs.repeat(self.num_candidates, 1, 1, 1)

        candidate_actions = torch.randint(
            low=0,
            high=NUM_ACTIONS,
            size=(self.num_candidates, self.horizon),
            device=self.device,
        )

        with torch.no_grad():
            rollout = rollout_latent_model(
                self.autoencoder,
                self.dynamics,
                obs_batch,
                candidate_actions
            )

            scores = self.score_rollouts(rollout)

        best_idx = torch.argmax(scores).item()
        best_sequence = candidate_actions[best_idx]
        best_action = int(best_sequence[0].item())

        return best_action, {
            "best_score": float(scores[best_idx].item()),
            "best_sequence": best_sequence.detach().cpu().tolist(),
            "mean_score": float(scores.mean().item()),
            "max_score": float(scores.max().item()),
            "min_score": float(scores.min().item()),
        }

    def score_rollouts(self, rollout):
        """
        Scores imagined futures.

        rollout["rewards"]: [N, H]
        rollout["done_probs"]: [N, H]
        rollout["collision_probs"]: [N, H]
        rollout["pred_tiles"]: [N, H, 10, 10]
        """

        rewards = rollout["rewards"]
        done_probs = rollout["done_probs"]
        collision_probs = rollout["collision_probs"]
        pred_tiles = rollout["pred_tiles"]

        batch_size, horizon = rewards.shape

        discounts = torch.tensor(
            [self.gamma ** t for t in range(horizon)],
            device=rewards.device,
            dtype=torch.float32
        )[None, :]

        previous_not_done = 1.0 - done_probs[:, :-1]
        continuation = torch.cat(
            [
                torch.ones(batch_size, 1, device=rewards.device),
                previous_not_done,
            ],
            dim=1
        )

        continuation = torch.cumprod(continuation, dim=1)

        agent_counts = (pred_tiles == 4).sum(dim=(2, 3)).float()
        invalid_agent = (agent_counts != 1).float()

        per_step_score = (
            rewards
            - self.collision_penalty * collision_probs
            - self.invalid_agent_penalty * invalid_agent
        )

        scores = (discounts * continuation * per_step_score).sum(dim=1)
        return scores