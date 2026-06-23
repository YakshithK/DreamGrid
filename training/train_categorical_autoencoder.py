import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from env.palette import rgb_to_palette_indices
from models.categorical_autoencoder import CategoricalAutoencoder

class ImageDataset(Dataset):
    def __init__(self, path):
        data = np.load(path)
        self.current_images = data['current_images']
        self.next_images = data['next_images']
        self.length = len(self.current_images) * 2

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        real_idx = idx // 2
        use_next = idx % 2 == 1

        if use_next:
            image = self.next_images[real_idx]
        else:
            image = self.current_images[real_idx]

        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        return image

def reconstruction_loss(logits, target_image):
    target_classes = rgb_to_palette_indices(target_image)

    class_weights = torch.tensor(
        [1.0, 4.0, 8.0, 8.0, 20.0],
        device=target_image.device,
    )

    return F.cross_entropy(logits, target_classes, weight=class_weights)

def eval(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_items = 0
    correct_pixels = 0
    total_pixels = 0

    with torch.no_grad():
        for image in loader:
            image = image.to(device)

            logits, _ = model(image)
            loss = reconstruction_loss(logits, image)

            target_classes = rgb_to_palette_indices(image)
            pred_classes = logits.argmax(dim=1)

            correct_pixels += (pred_classes == target_classes).sum().item()
            total_pixels += target_classes.numel()

            total_loss += loss.item() * image.shape[0]
            total_items += image.shape[0]

    return total_loss / total_items, correct_pixels / total_pixels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default='data/train_transitions.npz')
    parser.add_argument('--val_path', default='data/val_transitions.npz')
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/')
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_dataset = ImageDataset(args.train_path)
    val_dataset = ImageDataset(args.val_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = CategoricalAutoencoder(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0

        for image in tqdm(train_loader, desc=f"Epoch {epoch}"):
            image = image.to(device)

            logits, _ = model(image)
            loss = reconstruction_loss(logits, image)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * image.shape[0]
            total_items += image.shape[0]

        train_loss = total_loss / total_items
        val_loss, val_pixel_acc = eval(model, val_loader, device)

        print(
            f"Epoch {epoch}: "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Pixel Acc: {val_pixel_acc:.4f}"
        )

        if val_loss < best_val:
            best_val  = val_loss
            path = os.path.join(
                args.checkpoint_dir,
                f"categorical_autoencoder_latent{args.latent_dim}.pt"
            )
            torch.save(model.state_dict(), path)
            print(f"Saved best model to {path}")

if __name__ == '__main__':
    main()