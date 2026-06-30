import torch
import torch.nn as nn
import torch.nn.functional as F

from env.constants import NUM_ACTIONS, GRID_SIZE
from models.blocks import ResidualBlock
    
class SpatialDynamicsModel(nn.Module):
    def __init__(self, num_tile_classes=5, hidden_dim=128, num_blocks=4):
        super().__init__()

        self.num_tile_classes = num_tile_classes

        input_channels = num_tile_classes + NUM_ACTIONS

        self.input_proj = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(num_blocks)]
        )

        self.tile_head = nn.Conv2d(
            hidden_dim,
            num_tile_classes,
            kernel_size=1
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.reward_head = nn.Linear(hidden_dim, 1)
        self.done_head = nn.Linear(hidden_dim, 1)
        self.collision_head = nn.Linear(hidden_dim, 1)

    def forward(self, current_tiles, action):
        """
        current_tiles: [B, 10, 10], integer tile ids
        action: [B], integer action ids

        returns:
            next_tile_logits: [B, 5, 10, 10]
            reward: [B]
            done_logit: [B]
            collision_logit: [B]
        """

        batch_size = current_tiles.size(0)

        tile_onehot = F.one_hot(
            current_tiles,
            num_classes=self.num_tile_classes
        ).float()

        tile_onehot = tile_onehot.permute(0, 3, 1, 2)

        action_onehot = F.one_hot(
            action,
            num_classes=NUM_ACTIONS
        ).float()

        action_planes = action_onehot[:, :, None, None].expand(
            batch_size,
            NUM_ACTIONS,
            GRID_SIZE,
            GRID_SIZE
        )

        x = torch.cat([tile_onehot, action_planes], dim=1)

        h = self.input_proj(x)
        h = self.blocks(h)

        next_tile_logits = self.tile_head(h)

        pooled = self.pool(h).flatten(1)

        reward = self.reward_head(pooled).squeeze(1)
        done_logit = self.done_head(pooled).squeeze(1)
        collision_logit = self.collision_head(pooled).squeeze(1)

        return {
            "next_tile_logits": next_tile_logits,
            "reward": reward,
            "done_logit": done_logit,
            "collision_logit": collision_logit
        }