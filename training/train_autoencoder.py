import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.autoencoder import ConvAutoencoder
from datasets.images import ImageDataset
    
def weighted_reconstruction_loss(recon, target):
    agent_blue = torch.tensor(
        [45 / 255.0, 105 / 255.0, 230 / 255.0],
        device=target.device,
    )[None, :, None, None]

    hazard_red = torch.tensor(
        [220 / 255.0, 55 / 255.0, 55 / 255.0],
        device=target.device,
    )[None, :, None, None]

    goal_green = torch.tensor(
        [50 / 255.0, 180 / 255.0, 90 / 255.0],
        device=target.device,
    )[None, :, None, None]

    agent_mask = (target - agent_blue).abs().mean(dim=1, keepdim=True) < 0.03
    hazard_mask = (target - hazard_red).abs().mean(dim=1, keepdim=True) < 0.03
    goal_mask = (target - goal_green).abs().mean(dim=1, keepdim=True) < 0.03

    
    weights = torch.ones_like(target[:, :1])
    weights += agent_mask.float() * 30.0
    weights += hazard_mask.float() * 15.0
    weights += goal_mask.float() * 10.0

    pixel_l1 = (recon - target).abs().mean(dim=1, keepdim=True)
    return (pixel_l1 * weights).sum() / weights.sum()

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_items = 0

    with torch.no_grad():
        for image in loader:
            image = image.to(device)
            recon, _ = model(image)
            loss = weighted_reconstruction_loss(recon, image)

            total_loss += loss.item() * image.shape[0]
            total_items += image.shape[0]

    return total_loss / total_items

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default='data/train_transitions.npz')
    parser.add_argument('--val_path', default='data/val_transitions.npz')
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--checkpoint_dir', default='checkpoints')
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_dataset = ImageDataset(args.train_path)
    val_dataset = ImageDataset(args.val_path)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = ConvAutoencoder(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")

    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        total_items = 0

        for image in tqdm(train_loader, desc=f"Epoch {epoch}"):
            image = image.to(device)

            recon, _ = model(image)
            loss = weighted_reconstruction_loss(recon, image)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * image.shape[0]
            total_items += image.shape[0]

        train_loss = total_loss / total_items
        val_loss = evaluate(model, val_loader, device)

        print(f"Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            checkpoint_path = os.path.join(args.checkpoint_dir, f"autoencoder_latent{args.latent_dim}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")

if __name__ == "__main__":
    main()
