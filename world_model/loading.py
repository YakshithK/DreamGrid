import torch

from models.latent_dynamics import LatentDynamicsModel
from models.tile_autoencoder import TileAutoencoder

def load_tile_autoencoder(checkpoint_path, latent_dim, device):
    autoencoder = TileAutoencoder(latent_dim=latent_dim).to(device)
    autoencoder.load_state_dict(torch.load(checkpoint_path, map_location=device))
    autoencoder.eval()

    for param in autoencoder.parameters():
        param.requires_grad = False

    return autoencoder

def load_latent_dynamics(checkpoint_path, latent_dim, device):
    dynamics = LatentDynamicsModel(latent_dim=latent_dim).to(device)
    dynamics.load_state_dict(torch.load(checkpoint_path, map_location=device))
    dynamics.eval()

    return dynamics