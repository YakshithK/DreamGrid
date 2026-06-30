import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from env.tile_palette import image_to_tile_classes
from datasets_utils.transitions import ImageActionTransitionDataset
from world_model.decoder import build_copy_residual_tile_logits_from_image
from world_model.loading import load_latent_dynamics, load_tile_autoencoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data/transitions/test_transitions.npz')
    parser.add_argument('--autoencoder_checkpoint', default='checkpoints/baselines/tile_autoencoder_latent128.pt')
    parser.add_argument('--dynamics_checkpoint', default='checkpoints/baselines/latent_dynamics_latent128.pt')
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loader = DataLoader(
        ImageActionTransitionDataset(args.data_path),
        batch_size=args.batch_size,
        shuffle=False
    )

    autoencoder = load_tile_autoencoder(args.autoencoder_checkpoint, args.latent_dim, device)

    dynamics = load_latent_dynamics(args.dynamics_checkpoint, args.latent_dim, device)

    total =0

    tile_correct = 0
    tile_total =0

    important_correct = 0
    important_total = 0

    reward_abs_error = 0.0
    done_correct = 0
    collision_correct = 0

    static_important_correct = 0
    static_important_total = 0
    changed_correct = 0
    changed_total = 0

    with torch.no_grad():
        for current, action, nxt, reward, done, collision in loader:
            current = current.to(device)
            action = action.to(device)
            nxt = nxt.to(device)
            reward = reward.to(device)

            done = done.to(device)
            collision = collision.to(device)

            z = autoencoder.encode(current)
            outputs = dynamics(z, action)

            pred_logits = build_copy_residual_tile_logits_from_image(outputs, current)
            pred_tiles = pred_logits.argmax(dim=1)
            true_tiles = image_to_tile_classes(nxt)

            current_tiles = image_to_tile_classes(current)

            static_important = (current_tiles == true_tiles) & (true_tiles != 0)
            changed = current_tiles != true_tiles

            static_important_correct += ((pred_tiles == true_tiles) & static_important).sum().item()
            static_important_total += static_important.sum().item()

            changed_correct += ((pred_tiles == true_tiles) & changed).sum().item()
            changed_total += changed.sum().item()

            tile_correct += (pred_tiles == true_tiles).sum().item()
            tile_total += true_tiles.numel()

            important = true_tiles != 0
            important_correct += ((pred_tiles == true_tiles) & important).sum().item()
            important_total += important.sum().item()

            reward_abs_error += (outputs["reward"] - reward).abs().sum().item()

            done_pred = torch.sigmoid(outputs["done_logit"]) > 0.5
            collision_pred = torch.sigmoid(outputs["collision_logit"]) > 0.5

            done_correct += (done_pred.float() == done).sum().item()
            collision_correct += (collision_pred.float() == collision).sum().item()

            total += current.shape[0]

    print(f"tile accuracy: {tile_correct / tile_total:.4f}")
    print(f"important tile accuracy: {important_correct / important_total:.4f}")
    print(f"reward MAE: {reward_abs_error / total:.4f}")
    print(f"done accuracy: {done_correct / total:.4f}")
    print(f"collision accuracy: {collision_correct / total:.4f}")
    print(f"static important tile accuracy: {static_important_correct / max(static_important_total, 1):.4f}")
    print(f"changed tile accuracy: {changed_correct / max(changed_total, 1):.4f}")

if __name__ == "__main__":
    main()
