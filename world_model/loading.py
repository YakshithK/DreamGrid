import torch

from models.baselines.latent_dynamics import LatentDynamicsModel
from models.baselines.spatial_dynamics import SpatialDynamicsModel
from models.baselines.tile_autoencoder import TileAutoencoder
from models.final.vq_dynamics import VQDynamics
from models.final.vqvae import VQVAE


def freeze_eval(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model
    

def load_tile_autoencoder(checkpoint_path, latent_dim, device):
    autoencoder = TileAutoencoder(latent_dim=latent_dim).to(device)
    autoencoder.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return freeze_eval(autoencoder)


def load_latent_dynamics(checkpoint_path, latent_dim, device):
    model = LatentDynamicsModel(latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    return model


def load_spatial_dynamics(checkpoint_path, device):
    model = SpatialDynamicsModel().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    return model


def load_vqvae(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = VQVAE(
        num_codes=checkpoint["num_codes"],
        code_dim=checkpoint["code_dim"],
        hidden_dim=checkpoint["hidden_dim"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    return freeze_eval(model)


def load_vqvae_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = VQVAE(
        num_codes=checkpoint["num_codes"],
        code_dim=checkpoint["code_dim"],
        hidden_dim=checkpoint["hidden_dim"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model = freeze_eval(model)

    return model, checkpoint


def load_vq_dynamics(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = VQDynamics(
        num_codes=checkpoint["num_codes"],
        hidden_dim=checkpoint["hidden_dim"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
