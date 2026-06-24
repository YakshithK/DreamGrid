import torch
import torch.nn as nn
import torch.nn.functional as F

from env.constants import NUM_ACTIONS

class LatentDynamicsModel(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=256):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.trunk = nn.Sequential(
            nn.Linear(latent_dim + NUM_ACTIONS, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )

        self.delta_head = nn.Linear(hidden_dim, latent_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)
        self.done_head = nn.Linear(hidden_dim, 1)
        self.collision_head = nn.Linear(hidden_dim, 1)

    def forward(self, z, action):
        action_onehot = F.one_hot(action, num_classes=NUM_ACTIONS).float()
        x = torch.cat([z, action_onehot], dim=1)

        h = self.trunk(x)

        delta_z = self.delta_head(h)
        next_z = z + delta_z

        reward = self.reward_head(h).squeeze(1)
        done_logit = self.done_head(h).squeeze(1)
        collision_logit = self.collision_head(h).squeeze(1)

        return {
            "next_z": next_z,
            "delta_z": delta_z,
            "reward": reward,
            "done_logit": done_logit,
            "collision_logit": collision_logit
        }