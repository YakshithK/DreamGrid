import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from env.tile_palette import image_to_tile_classes
from models.tile_autoencoder import TileAutoencoder
from models.tile_dynamics import TileDynamicsModel

class TileTransitionDataset(Dataset):
    def __init__(self, path):
        data = np.load(path)
        self.current_images = data["current_images"]
        self.next_images = data["next_images"]
        self.actions = data["actions"]
        self.rewards = data["rewards"]
        self.dones = data["dones"]
        self.collisions = data["collisions"]

    def __len__(self):
        return len(self.actions)
    
    def __getitem__(self, idx):
        current = self.current_images[idx].astype(np.float32) / 255.0
        nxt = self.next_images[idx].astype(np.float32) / 255.0

        current = torch.from_numpy(current).permute(2, 0, 1)
        nxt = torch.from_numpy(nxt).permute(2, 0, 1)

        return {
            "current": current,
            "next": nxt,
            "action": torch.tensor(self.actions[idx], dtype=torch.long),
            "reward": torch.tensor(self.rewards[idx], dtype=torch.float32),
            "done": torch.tensor(self.dones[idx], dtype=torch.float32),
            "collision": torch.tensor(self.collisions[idx], dtype=torch.float32),
        }
    
def compute_loss(outputs, target_tiles, reward, done, collision):
    class_weights = torch.tensor(
        [1.0, 5.0, 15.0, 15.0, 30.0],
        device=target_tiles.device,
    )

    tile_loss = F.cross_entropy(
        outputs["tile_logits"],
        target_tiles,
        weight=class_weights,
    )

    reward_loss = F.smooth_l1_loss(outputs["reward"], reward)

    done_loss = F.binary_cross_entropy_with_logits(
        outputs["done_logit"],
        done,
    )

    collision_loss = F.binary_cross_entropy_with_logits(
        outputs["collision_logit"],
        collision,
    )

    total = (
        4.0 * tile_loss
        + 0.5 * reward_loss
        + 0.5 * done_loss
        + 0.75 * collision_loss
    )

    return total, {
        "tile_loss": tile_loss.item(),
        "reward_loss": reward_loss.item(),
        "done_loss": done_loss.item(),
        "collision_loss": collision_loss.item(),
    }

def eval(model, loader, device):
    model.eval()

    tile_correct = 0
    tile_total = 0
    important_correct = 0
    important_total = 0
    changed_correct = 0
    changed_total = 0
    single_agent = 0
    total_examples = 0
    reward_abs_error = 0.0
    done_correct = 0
    collision_correct = 0

    with torch.no_grad():
        for batch in loader:
            current = batch["current"].to(device)
            nxt = batch["next"].to(device)
            action = batch["action"].to(device)
            reward = batch["reward"].to(device)
            done = batch["done"].to(device)
            collision = batch["collision"].to(device)

            current_tiles = image_to_tile_classes(current)
            target_tiles = image_to_tile_classes(nxt)

            outputs = model(current_tiles, action)
            pred_tiles = outputs["tile_logits"].argmax(dim=1)

            tile_correct += (pred_tiles == target_tiles).sum().item()
            tile_total += target_tiles.numel()

            important = target_tiles != 0
            important_correct += ((pred_tiles == target_tiles) & important).sum().item()
            important_total += important.sum().item()

            changed = current_tiles != target_tiles
            changed_correct += ((pred_tiles == target_tiles) & changed).sum().item()
            changed_total += changed.sum().item()

            pred_agent_count = (pred_tiles == 4).sum(dim=(1, 2))
            single_agent += (pred_agent_count == 1).sum().item()

            reward_abs_error += torch.abs(outputs["reward"] - reward).sum().item()

            done_pred = torch.sigmoid(outputs["done_logit"]) > 0.5
            collision_pred = torch.sigmoid(outputs["collision_logit"]) > 0.5

            done_correct += (done_pred.float() == done).sum().item()
            collision_correct += (collision_pred.float() == collision).sum().item()

            total_examples += current.shape[0]

    return {
        "tile_acc": tile_correct / tile_total,
        "important_acc": important_correct / max(important_total, 1),
        "changed_acc": changed_correct / max(changed_total, 1),
        "single_agent": single_agent / total_examples,
        "reward_mae": reward_abs_error / total_examples,
        "done_acc": done_correct / total_examples,
        "collision_acc": collision_correct / total_examples,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="data/train_transitions.npz")
    parser.add_argument("--val_path", default="data/val_transitions.npz")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader = DataLoader(
        TileTransitionDataset(args.train_path),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        TileTransitionDataset(args.val_path),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )

    model = TileDynamicsModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_score = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total_items = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            current = batch["current"].to(device)
            nxt = batch["next"].to(device)
            action = batch["action"].to(device)
            reward = batch["reward"].to(device)
            done = batch["done"].to(device)
            collision = batch["collision"].to(device)

            current_tiles = image_to_tile_classes(current)
            target_tiles = image_to_tile_classes(nxt)

            outputs = model(current_tiles, action)
            loss, _ = compute_loss(outputs, target_tiles, reward, done, collision)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * current.shape[0]
            total_items += current.shape[0]

        metrics = eval(model, val_loader, device)

        print(
            f"Epoch {epoch}: "
            f"train_loss={total_loss / total_items:.4f}, "
            f"tile_acc={metrics['tile_acc']:.4f}, "
            f"important_acc={metrics['important_acc']:.4f}, "
            f"changed_acc={metrics['changed_acc']:.4f}, "
            f"single_agent={metrics['single_agent']:.4f}, "
            f"reward_mae={metrics['reward_mae']:.4f}, "
            f"done_acc={metrics['done_acc']:.4f}, "
            f"collision_acc={metrics['collision_acc']:.4f}"
        )

        score = (metrics["changed_acc"]
        + metrics["single_agent"]
        + metrics["important_acc"])

        if score > best_score:
            best_score = score
            checkpoint_path = os.path.join(args.checkpoint_dir, "tile_dynamics_best.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")

if __name__ == "__main__":
    main()