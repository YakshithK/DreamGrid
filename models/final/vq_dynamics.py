import torch
import torch.nn as nn
import torch.nn.functional as F

from env.constants import GRID_SIZE, NUM_ACTIONS
from models.blocks import ResidualBlock
    

class VQDynamics(nn.Module):
    def __init__(self, num_codes=128, hidden_dim=128, num_blocks=4):
        super().__init__()

        self.num_codes = num_codes

        self.code_embedding = nn.Embedding(num_codes, hidden_dim)

        input_channels = hidden_dim + NUM_ACTIONS

        self.input_proj = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(num_blocks)]
        )

        self.code_head = nn.Conv2d(hidden_dim, num_codes, kernel_size=1)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.reward_head = nn.Linear(hidden_dim, 1)
        self.done_head = nn.Linear(hidden_dim, 1)
        self.collision_head = nn.Linear(hidden_dim, 1)


    def forward(self, code_ids, actions):
        """
        code_ids: [B, 10, 10]
        action: [B]

        returns:
            next_code_logits: [B, num_codes, 10, 10]
            reward: [B]
            done: [B]
            collision_logit: [B]
        """
        batch_size = code_ids.shape[0]

        code_features = self.code_embedding(code_ids)
        code_features = code_features.permute(0, 3, 1, 2).contiguous()

        action_onehot = F.one_hot(actions, NUM_ACTIONS).float()
        action_planes = action_onehot[:, :, None, None].expand(
            batch_size,
            NUM_ACTIONS,
            GRID_SIZE,
            GRID_SIZE
        )

        x = torch.cat([code_features, action_planes], dim=1)

        h = self.input_proj(x)
        h = self.blocks(h)

        next_code_logits = self.code_head(h)

        pooled = self.pool(h).flatten(1)

        reward = self.reward_head(pooled).squeeze(1)
        done_logit = self.done_head(pooled).squeeze(1)
        collision_logit = self.collision_head(pooled).squeeze(1)

        return {
            "next_code_logits": next_code_logits,
            "reward": reward,
            "done_logit": done_logit,
            "collision_logit": collision_logit
        }