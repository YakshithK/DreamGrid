import torch
import torch.nn as nn
import torch.nn.functional as F

from env.constants import NUM_ACTIONS

class PixelTransitionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3 + NUM_ACTIONS, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, image, action):
        batch_size, _, height, width = image.shape

        action_onehot = F.one_hot(action, NUM_ACTIONS).float()
        action_planes = action_onehot[:, :, None, None].expand(
            batch_size, NUM_ACTIONS, height, width
        )

        x = torch.cat([image, action_planes], dim=1)
        z = self.encoder(x)
        pred = self.decoder(z)

        return pred