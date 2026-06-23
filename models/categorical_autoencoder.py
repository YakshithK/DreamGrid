import torch
import torch.nn as nn

class CategoricalAutoencoder(nn.Module):
    def __init__(self, latent_dim=64, num_classes=5):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # Encoder
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        self.encoder_fc = nn.Linear(256 * 5 * 5, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, 256 * 5 * 5)

        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, num_classes, kernel_size=4, stride=2, padding=1),
        )

    def encode(self, image):
        x = self.encoder_cnn(image)
        x = x.flatten(start_dim=1)
        z = self.encoder_fc(x)
        return z
    
    def decode_logits(self, z):
        x = self.decoder_fc(z)
        x = x.view(z.shape[0], 256, 5, 5)
        logits = self.decoder_cnn(x)
        return logits

    def forward(self, image):
        z = self.encode(image)
        logits = self.decode_logits(z)
        return logits, z