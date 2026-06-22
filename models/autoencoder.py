import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim

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
        self.decoder_fc = nn.Linear(latent_dim, 256 * 5 
        * 5)

        # Decoder
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, image):
        x = self.encoder_cnn(image)
        x = x.flatten(start_dim=1)
        z = self.encoder_fc(x)
        return z
    
    def decode(self, z):
        x = self.decoder_fc(z)
        x = x.view(z.shape[0], 256, 5, 5)
        recon = self.decoder_cnn(x)
        return recon
    
    def forward(self, image):
        z = self.encode(image)
        recon = self.decode(z)
        return recon, z