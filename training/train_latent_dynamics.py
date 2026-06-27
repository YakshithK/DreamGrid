import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from env.tile_palette import image_to_tile_classes
from losses.latent_dynamics import compute_latent_dynamics_loss
from models.latent_dynamics import LatentDynamicsModel
from models.tile_autoencoder import TileAutoencoder
from datasets_utils.transitions import ImageActionTransitionDataset
from world_model.decoder import copy_logits_from_tiles, build_copy_residual_tile_logits_from_image


def encode_images(autoencoder, images):
    with torch.no_grad():
        z = autoencoder.encode(images)
    return z

def evaluate(autoencoder, dynamics, loader, device):
    autoencoder.eval()
    dynamics.eval()

    total_loss = 0.0
    total_items = 0

    done_correct = 0
    collision_correct = 0
    total_binary = 0

    reward_abs_error = 0.0

    with torch.no_grad():
        for current, action, nxt, reward, done, collision in loader:
            current = current.to(device)
            action = action.to(device)
            nxt = nxt.to(device)
            reward = reward.to(device)
            done = done.to(device)
            collision = collision.to(device)

            z = autoencoder.encode(current)
            target_z = autoencoder.encode(nxt)

            outputs = dynamics(z, action)
            loss, _ = compute_latent_dynamics_loss(
                outputs,
                current,
                nxt,
                target_z,
                reward,
                done,
                collision
            )

            done_pred = torch.sigmoid(outputs["done_logit"]) > 0.5
            collision_pred = torch.sigmoid(outputs["collision_logit"]) > 0.5

            done_correct += (done_pred.float() == done).sum().item()
            collision_correct += (collision_pred.float() == collision).sum().item()

            total_binary += done.shape[0]

            reward_abs_error += (outputs["reward"] - reward).abs().sum().item()

            total_loss += loss.item() * current.shape[0]
            total_items += current.shape[0]

    return {
        "loss": total_loss / total_items,
        "done_acc": done_correct / total_binary,
        "collision_acc": collision_correct / total_binary,
        "reward_mae": reward_abs_error / total_items
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="data/train_transitions.npz")
    parser.add_argument("--val_path", default="data/val_transitions.npz")
    parser.add_argument("--autoencoder_checkpoint", default="checkpoints/tile_autoencoder_latent128.pt")
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    autoencoder = TileAutoencoder(latent_dim=args.latent_dim).to(device)
    autoencoder.load_state_dict(torch.load(args.autoencoder_checkpoint, map_location=device))
    autoencoder.eval()

    for param in autoencoder.parameters():
        param.requires_grad = False

    dynamics = LatentDynamicsModel(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(dynamics.parameters(), lr=args.lr)

    train_loader = DataLoader(
        ImageActionTransitionDataset(args.train_path),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    val_loader = DataLoader(
        ImageActionTransitionDataset(args.val_path),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        dynamics.train()

        total_loss = 0.0
        total_items = 0
        loss_parts = {
            "latent_loss": 0.0,
            "tile_loss": 0.0,
            "reward_loss": 0.0,
            "done_loss": 0.0,
            "collision_loss": 0.0,
            "delta_reg": 0.0,
            "static_residual_penalty": 0.0,
        }

        for current, action, nxt, reward, done, collision in tqdm(train_loader, desc=f"Epoch {epoch}"):
            current = current.to(device)
            action = action.to(device)
            nxt = nxt.to(device)
            reward = reward.to(device)
            done = done.to(device)
            collision = collision.to(device)

            z = encode_images(autoencoder, current)
            target_z = encode_images(autoencoder, nxt)

            outputs = dynamics(z, action)

            loss, parts = compute_latent_dynamics_loss(
                outputs,
                current,
                nxt,
                target_z,
                reward,
                done,
                collision
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = current.shape[0]
            total_loss += loss.item() * batch_size
            total_items += batch_size

            for key in loss_parts:
                loss_parts[key] += parts[key] * batch_size

        train_loss = total_loss / total_items
        train_parts = {key: val / total_items for key, val in loss_parts.items()}

        val_metrics = evaluate(autoencoder, dynamics, val_loader, device)

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.6f}, "
            f"latent={train_parts['latent_loss']:.6f}, "
            f"tile={train_parts['tile_loss']:.6f}, "
            f"reward={train_parts['reward_loss']:.6f}, "
            f"done={train_parts['done_loss']:.6f}, "
            f"collision={train_parts['collision_loss']:.6f}, "
            f"delta={train_parts['delta_reg']:.6f}, "
            f"static_resid={train_parts['static_residual_penalty']:.6f}, "
            f"val_loss={val_metrics['loss']:.6f}, "
            f"done_acc={val_metrics['done_acc']:.4f}, "
            f"collision_acc={val_metrics['collision_acc']:.4f}, "
            f"reward_mae={val_metrics['reward_mae']:.4f}"
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            path = os.path.join(
                args.checkpoint_dir,
                f"latent_dynamics_latent{args.latent_dim}.pt"
            )
            torch.save(dynamics.state_dict(), path)
            print(f"Saved best model to {path}")


if __name__ == "__main__":
    main()