import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from env.constants import ACTION_NAMES
from env.tile_palette import tile_classes_to_image
from world_model.decoder import build_copy_residual_tile_logits_from_image
from world_model.loading import load_latent_dynamics, load_tile_autoencoder


def decode_to_rgb(autoencoder, z):
    logits = autoencoder.decode_tile_logits(z)
    tiles = logits.argmax(dim=1)
    rgb = tile_classes_to_image(tiles)
    return rgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/test_transitions.npz")
    parser.add_argument("--autoencoder_checkpoint", default="checkpoints/tile_autoencoder_latent128.pt")
    parser.add_argument("--dynamics_checkpoint", default="checkpoints/latent_dynamics_latent128.pt")
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--out_path", default="outputs/latent_predictions.png")
    parser.add_argument("--num_examples", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = np.load(args.data_path)
    current_images = data["current_images"]
    next_images = data["next_images"]
    actions = data["actions"]
    rewards = data["rewards"]
    dones = data["dones"]
    collisions = data["collisions"]

    autoencoder = load_tile_autoencoder(args.autoencoder_checkpoint, args.latent_dim, device)

    dynamics = load_latent_dynamics(args.dynamics_checkpoint, args.latent_dim, device)

    indices = np.linspace(0, len(actions) - 1, args.num_examples, dtype=int)

    fig, axes = plt.subplots(args.num_examples, 3, figsize=(9, args.num_examples * 2.6))

    with torch.no_grad():
        for row, idx in enumerate(indices):
            current = current_images[idx].astype(np.float32) / 255.0
            true_next = next_images[idx].astype(np.float32) / 255.0
            action = int(actions[idx])

            current_tensor = torch.from_numpy(current).permute(2, 0, 1)[None].to(device)
            action_tensor = torch.tensor([action], dtype=torch.long, device=device)

            z = autoencoder.encode(current_tensor)
            outputs = dynamics(z, action_tensor)

            pred_logits = build_copy_residual_tile_logits_from_image(outputs, current_tensor)
            pred_tiles = pred_logits.argmax(dim=1)
            pred_rgb = tile_classes_to_image(pred_tiles)

            pred = pred_rgb[0].permute(1, 2, 0).cpu().numpy()

            pred_reward = outputs["reward"].item()
            pred_done= torch.sigmoid(outputs["done_logit"]).item()
            pred_collision = torch.sigmoid(outputs["collision_logit"]).item()

            axes[row, 0].imshow(current)
            axes[row, 0].set_title(f"Current\nAction: {ACTION_NAMES[action]}")

            axes[row, 1].imshow(true_next)
            axes[row, 1].set_title(
                f"True next\nr={rewards[idx]:.2f}, done={bool(dones[idx])}, col={bool(collisions[idx])}"
            )

            axes[row, 2].imshow(pred)
            axes[row, 2].set_title(
                f"Pred next\nr={pred_reward:.2f}, done={pred_done:.2f}, col={pred_collision:.2f}"
            )

            for col in range(3):
                axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(args.out_path, dpi=160)
    print(f"Saved latent prediction visualization to {args.out_path}")


if __name__ == "__main__":
    main()