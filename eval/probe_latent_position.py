import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from env.constants import GRID_SIZE
from models.autoencoder import ConvAutoencoder
from models.categorical_autoencoder import CategoricalAutoencoder

class PositionProbeDataset(Dataset):
    def __init__(self, path):
        data = np.load(path)
        self.images = data["next_images"]
        self.positions = data["next_positions"]

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)

        pos = self.positions[idx]
        row = torch.tensor(pos[0], dtype=torch.long)
        col = torch.tensor(pos[1], dtype=torch.long)

        return image, row, col
    
class LatentPositionProbe(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.row_head = nn.Linear(128, GRID_SIZE)
        self.col_head = nn.Linear(128, GRID_SIZE)

    def forward(self, z):
        h = self.net(z)
        row_logits = self.row_head(h)
        col_logits = self.col_head(h)
        return row_logits, col_logits

def evaluate(autoencoder, probe, loader, device):
    autoencoder.eval()
    probe.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for image, row, col in loader:
            image = image.to(device)
            row = row.to(device)
            col = col.to(device)

            z = autoencoder.encode(image)
            row_logits, col_logits = probe(z)

            row_pred = row_logits.argmax(dim=1)
            col_pred = col_logits.argmax(dim=1)

            correct += ((row_pred == row) & (col_pred == col)).sum().item()
            total += image.shape[0]

    return correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='data/train_transitions.npz')
    parser.add_argument('--test_path', type=str, default='data/test_transitions.npz')
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--checkpoint', type=str, default='checkpoints/categorical_autoencoder_latent64.pt')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # autoencoder = ConvAutoencoder(latent_dim=args.latent_dim).to(device)
    
    autoencoder = CategoricalAutoencoder(latent_dim=args.latent_dim).to(device)
    autoencoder.load_state_dict(torch.load(args.checkpoint, map_location=device))
    autoencoder.eval()

    for param in autoencoder.parameters():
        param.requires_grad = False

    probe = LatentPositionProbe(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)

    train_loader = DataLoader(PositionProbeDataset(args.train_path), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(PositionProbeDataset(args.test_path), batch_size=args.batch_size, shuffle=False)

    for epoch in range(1, args.epochs + 1):
        probe.train()

        for image, row, col in tqdm(train_loader, desc=f"Probe epoch {epoch}"):
            image = image.to(device)
            row = row.to(device)
            col = col.to(device)

            with torch.no_grad():
                z = autoencoder.encode(image)

            row_logits, col_logits = probe(z)

            loss = nn.functional.cross_entropy(row_logits, row)
            loss += nn.functional.cross_entropy(col_logits, col)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc = evaluate(autoencoder, probe, test_loader, device)
        print(f"Epoch {epoch}: Test Accuracy: {acc:.4f}")

if __name__ == '__main__':
    main()
