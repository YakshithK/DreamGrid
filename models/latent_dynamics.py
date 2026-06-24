import torch
import torch.nn as nn
import torch.nn.functional as F

from env.constants import GRID_SIZE, NUM_ACTIONS


class LatentDynamicsModel(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=256, num_tile_classes=5):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_tile_classes = num_tile_classes

        self.trunk = nn.Sequential(
            nn.Linear(latent_dim + NUM_ACTIONS, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.delta_head = nn.Linear(hidden_dim, latent_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)
        self.done_head = nn.Linear(hidden_dim, 1)
        self.collision_head = nn.Linear(hidden_dim, 1)

        self.tile_delta_head = nn.Linear(
            hidden_dim,
            num_tile_classes * GRID_SIZE * GRID_SIZE
        )

    def forward(self, z, action):
        action_onehot = F.one_hot(action, NUM_ACTIONS).float()
        x = torch.cat([z, action_onehot], dim=1)

        h = self.trunk(x)

        delta_z = self.delta_head(h)
        next_z = z + delta_z

        reward = self.reward_head(h).squeeze(1)
        done_logit = self.done_head(h).squeeze(1)
        collision_logit = self.collision_head(h).squeeze(1)

        tile_delta_logits = self.tile_delta_head(h)
        tile_delta_logits = tile_delta_logits.view(
            z.shape[0],
            self.num_tile_classes,
            GRID_SIZE,
            GRID_SIZE
        )

        return {
            "next_z": next_z,
            "delta_z": delta_z,
            "reward": reward,
            "done_logit": done_logit,
            "collision_logit": collision_logit,
            "tile_delta_logits": tile_delta_logits
        }