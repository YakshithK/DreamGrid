import torch.nn as nn

class TileAutoencoder(nn.Module):
    def __init__(self, latent_dim=128, num_classes=5):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),    # 80 -> 40
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),   # 40 -> 20
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 20 -> 10
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # 10 -> 5
            nn.ReLU(),
        )

        self.encoder_fc = nn.Linear(256 * 5 * 5, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes * 10 * 10),
        )

    def encode(self, image):
        x = self.encoder_cnn(image)
        x = x.flatten(start_dim=1)
        z = self.encoder_fc(x)
        return z

    def decode_tile_logits(self, z):
        logits = self.decoder(z)
        logits = logits.view(z.shape[0], self.num_classes, 10, 10)
        return logits

    def forward(self, image):
        z = self.encode(image)
        logits = self.decode_tile_logits(z)
        return logits, z