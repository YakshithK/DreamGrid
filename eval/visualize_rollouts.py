import argparse
import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from eval.rollout_dataset import RolloutDataset
from eval.rollout_utils import rollout_model
from models.latent_dynamics import LatentDynamicsModel
from models.tile_autoencoder import TileAutoencoder


def image_for_plot(tensor):
    return tensor.permute(1, 2, 0).cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/test_transitions.npz")
    parser.add_argument("--autoencoder_checkpoint", default="checkpoints/tile_autoencoder_latent128.pt")
    parser.add_argument("--dynamics_checkpoint", default="checkpoints/latent_dynamics_latent128.pt")
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--num_examples", type=int, default=4)
    parser.add_argument("--out_path", default="outputs/rollouts_h5.png")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = RolloutDataset(args.data_path, horizon=args.horizon)
    loader = DataLoader(dataset, batch_size=args.num_examples, shuffle=True)

    batch = next(iter(loader))

    start_image = batch["start_image"].to(device)
    actions = batch["actions"].to(device)
    true_images = batch["true_images"].to(device)

    autoencoder = TileAutoencoder(latent_dim=args.latent_dim).to(device)
    autoencoder.load_state_dict(torch.load(args.autoencoder_checkpoint, map_location=device))
    autoencoder.eval()

    dynamics = LatentDynamicsModel(latent_dim=args.latent_dim).to(device)
    dynamics.load_state_dict(torch.load(args.dynamics_checkpoint, map_location=device))
    dynamics.eval()

    with torch.no_grad():
        rollout = rollout_model(autoencoder, dynamics, start_image, actions)

    pred_images = rollout["predicted_images"]

    rows = args.num_examples * 2
    cols = args.horizon + 1

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))

    for ex in range(args.num_examples):
        real_row = ex * 2
        pred_row = ex * 2 + 1

        axes[real_row, 0].imshow(image_for_plot(start_image[ex]))
        axes[real_row, 0].set_title("Start")
        axes[pred_row, 0].imshow(image_for_plot(start_image[ex]))
        axes[pred_row, 0].set_title("Start")

        for t in range(args.horizon):
            axes[real_row, t + 1].imshow(image_for_plot(true_images[ex, t]))
            axes[real_row, t + 1].set_title(f"True t={t + 1}")

            axes[pred_row, t + 1].imshow(image_for_plot(pred_images[ex, t]))
            axes[pred_row, t + 1].set_title(f"Pred t={t + 1}")
        
        for c in range(cols):
            axes[real_row, c].axis("off")
            axes[pred_row, c].axis("off")

    plt.tight_layout()
    plt.savefig(args.out_path)
    print(f"Saved rollout visualization to {args.out_path}")


if __name__ == "__main__":
    main()