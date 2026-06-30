import torch

from env.constants import NUM_ACTIONS
from world_model.rollouts.latent import rollout_latent_model

class LatentMPCPlanner:
    def __init__(
            self,
            autoencoder,
            dynamics,
            device,
            horizon=2,
            num_candidates=128,
            gamma=0.95,
            collision_penalty=3.0,
            invalid_agent_penalty=2.0,
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
            high=NUM_ACTIONS - 1,
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

        per_step_score = (
            rewards
            - self.collision_penalty * collision_probs
            - self.invalid_agent_penalty * invalid_agent
        )

        return (discounts * continuation * per_step_score).sum(dim=1)
        
    def visual_goal_progress_score(self, pred_tiles):
        """
        pred_tiles: [N, H, 10, 10]

        Returns:
        progress_score: [N, H]

        This gives the planner dense feedback from imagined visual states:
        smaller agent-goal distance is better.

        This does nto teach the world model the rules.
        The world model still predicts future tiles.
        The planner is just scoring the imagined futures.
        """
        batch_size, horizon, height, width = pred_tiles.shape
        device = pred_tiles.device

        flat = pred_tiles.view(batch_size, horizon, height * width)

        agent_mask = flat == 4
        goal_mask = flat == 3

        idxs = torch.arange(height * width, device=device)

        row = idxs // width
        col = idxs % width

        agent_exists = agent_mask.any(dim=2)
        goal_exists = goal_mask.any(dim=2)

        agent_idx = agent_mask.float().argmax(dim=2)
        goal_idx = goal_mask.float().argmax(dim=2)

        agent_r = row[agent_idx]
        agent_c = col[agent_idx]

        goal_r = row[goal_idx]
        goal_c = col[goal_idx]

        distance = (agent_r - goal_r).abs() + (agent_c - goal_c).abs()
        distance = distance.float()

        valid = agent_exists & goal_exists

        progress_score = -0.15 * distance
        progress_score = torch.where(
            valid,
            progress_score,
            torch.full_like(progress_score, -5.0)
        )

        reached_goal = valid & (distance == 0)
        progress_score += reached_goal.float() * 10.0

        return progress_score
