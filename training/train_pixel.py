import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.pixel_model import PixelTransitionModel
from datasets.transitions import PixelTransitionDataset

def weighted_transition_loss(pred, current, target):
    agent_blue = torch.tensor(
        [45 / 255.0, 105 / 255.0, 230 / 255.0],
        device=target.device,
    )[None, :, None, None]

    target_agent = (target - agent_blue).abs().sum(dim=1, keepdim=True) < 0.03
    current_agent = (current - agent_blue).abs().sum(dim=1, keepdim=True) < 0.03
    changed = (target - current).abs().mean(dim=1, keepdim=True) > 0.01

    weights = torch.ones_like(target[:, :1])
    weights += target_agent.float() * 30.0
    weights += current_agent.float() * 15.0
    weights += changed.float() * 10.0

    pixel_l1 = (pred - target).abs().mean(dim=1, keepdim=True)

    return (pixel_l1 * weights).sum() / weights.sum()

    
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_items = 0

    with torch.no_grad():
        for current, action, nxt in loader:
            current = current.to(device)
            action = action.to(device)
            nxt = nxt.to(device)

            pred = model(current, action)
            loss = weighted_transition_loss(pred, current, nxt)

            total_loss += loss.item() * current.shape[0]
            total_items += current.shape[0]

    return total_loss / total_items

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="data/train_transitions.npz")
    parser.add_argument("--val_path", default="data/val_transitions.npz")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = PixelTransitionDataset(args.train_path)
    val_dataset = PixelTransitionDataset(args.val_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )

    model = PixelTransitionModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_items = 0

        for current, action, nxt in tqdm(train_loader, desc=f"Epoch {epoch}"):
            current = current.to(device)
            action = action.to(device)
            nxt = nxt.to(device)

            pred = model(current, action)
            loss = weighted_transition_loss(pred, current, nxt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * current.shape[0]
            train_items += current.shape[0]

        avg_train = train_loss / train_items
        avg_val = evaluate(model, val_loader, device)

        print(f"Epoch {epoch}: train_l1={avg_train:.6f}, val_l1={avg_val:.6f}")

        if avg_val < best_val:
            best_val = avg_val
            checkpoint_path = os.path.join(args.checkpoint_dir, f"pixel_model.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")

if __name__ == "__main__":
    main()