import torch
import torch.nn as nn
import torch.nn.functional as F

from env.constants import GRID_SIZE, NUM_ACTIONS


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return F.relu(x + self.net(x))
    

class TileDynamicsModel(nn.Module):
    def __init__(self, hidden_dim=256, num_tile_classes=5):
        super().__init__()
        self.num_tile_classes = num_tile_classes

        in_channels = num_tile_classes + NUM_ACTIONS

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
        )

        self.tile_head = nn.Conv2d(hidden_dim, num_tile_classes, kernel_size=3, padding=1)

        self.global_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim * GRID_SIZE * GRID_SIZE, 256),
            nn.ReLU(),
        )

        self.reward_head = nn.Linear(256, 1)
        self.done_head = nn.Linear(256, 1)
        self.collision_head = nn.Linear(256, 1)

    def forward(self, current_tiles, action):
        tile_onehot = F.one_hot(current_tiles, num_classes=self.num_tile_classes).float()
        tile_onehot = tile_onehot.permute(0, 3, 1, 2)

        action_onehot = F.one_hot(action, NUM_ACTIONS).float()
        action_planes = action_onehot[:, :, None, None].expand(
            -1,
            -1,
            GRID_SIZE,
            GRID_SIZE
        )

        x = torch.cat([tile_onehot, action_planes], dim=1)

        h = self.encoder(x)

        tile_logits = self.tile_head(h)

        g = self.global_head(h)

        reward = self.reward_head(g).squeeze(1)
        done_logit = self.done_head(g).squeeze(1)
        collision_logit = self.collision_head(g).squeeze(1)

        return {
            "tile_logits": tile_logits,
            "reward": reward,
            "done_logit": done_logit,
            "collision_logit": collision_logit
        }