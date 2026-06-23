import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from env.tile_palette import image_to_tile_classes
from models.tile_autoencoder import TileAutoencoder


class ImageDataset(Dataset):
    def __init__(self, path):
        data = np.load(path)
        self.current_images = data["current_images"]
        self.next_images = data["next_images"]
        self.length = len(self.current_images) * 2

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        real_idx = idx // 2
        image = self.next_images[real_idx] if idx % 2 else self.current_images[real_idx]
        image = image.astype(np.float32) / 255.0
        return torch.from_numpy(image).permute(2, 0, 1)


def loss_fn(logits, image):
    target = image_to_tile_classes(image)

    class_weights = torch.tensor(
        [1.0, 4.0, 10.0, 10.0, 25.0],
        device=image.device,
    )

    return F.cross_entropy(logits, target, weight=class_weights)


def evaluate(model, loader, device):
    model.eval()

    total_loss = 0.0
    total_items = 0
    correct_tiles = 0
    total_tiles = 0
    important_correct = 0
    important_total = 0

    with torch.no_grad():
        for image in loader:
            image = image.to(device)

            logits, _ = model(image)
            loss = loss_fn(logits, image)

            target = image_to_tile_classes(image)
            pred = logits.argmax(dim=1)

            correct_tiles += (pred == target).sum().item()
            total_tiles += target.numel()

            important = target != 0
            important_correct += ((pred == target) & important).sum().item()
            important_total += important.sum().item()

            total_loss += loss.item() * image.shape[0]
            total_items += image.shape[0]

    tile_acc = correct_tiles / total_tiles
    important_acc = important_correct / max(important_total, 1)

    return total_loss / total_items, tile_acc, important_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="data/train_transitions.npz")
    parser.add_argument("--val_path", default="data/val_transitions.npz")
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

    train_loader = DataLoader(
        ImageDataset(args.train_path),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = DataLoader(
        ImageDataset(args.val_path),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = TileAutoencoder(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total_items = 0

        for image in tqdm(train_loader, desc=f"Epoch {epoch}"):
            image = image.to(device)

            logits, _ = model(image)
            loss = loss_fn(logits, image)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * image.shape[0]
            total_items += image.shape[0]

        train_loss = total_loss / total_items
        val_loss, tile_acc, important_acc = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"tile_acc={tile_acc:.4f}, "
            f"important_tile_acc={important_acc:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            path = os.path.join(
                args.checkpoint_dir,
                f"tile_autoencoder_latent{args.latent_dim}.pt",
            )
            torch.save(model.state_dict(), path)
            print(f"Saved best model to {path}")


if __name__ == "__main__":
    main()