import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets_utils.transitions import ImageActionTransitionDataset
from env.tile_palette import image_to_tile_classes
from models.spatial_dynamics import SpatialDynamicsModel


def agent_position_targets(next_tiles):
    """
    next_tiles: [B, 10, 10]

    return:
    target index in [0, 99] for the agent position
    """

    batch_size = next_tiles.shape[0]
    flat = next_tiles.view(batch_size, -1)
    return (flat == 4).float().argmax(dim=1)

def compute_loss(outputs, current_tiles, next_tiles, reward, done, collision):
    logits = outputs["next_tile_logits"]

    per_tile_ce = F.cross_entropy(logits, next_tiles, reduction="none")

    class_weights = torch.tensor(
        [1.0, 5.0, 15.0, 15.0, 30.0],
        device=next_tiles.device,
    )

    weights = class_weights[next_tiles]

    changed = current_tiles != next_tiles
    agent_related = (current_tiles == 4) | (next_tiles == 4)

    weights += changed.float() * 20.0
    weights += agent_related.float() * 50.0

    tile_loss = (per_tile_ce * weights).sum() / weights.sum()

    agent_logits = logits[:, 4].flatten(1)

    agent_target = agent_position_targets(next_tiles)
    agent_loss = F.cross_entropy(agent_logits, agent_target)

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
        2.0 * tile_loss
        + 3.0 * agent_loss
        + 0.25 * reward_loss
        + 0.5 * done_loss
        + 1.0 * collision_loss
    )

    return total, {
        "tile_loss": tile_loss.item(),
        "agent_loss": agent_loss.item(),
        "reward_loss": reward_loss.item(),
        "done_loss": done_loss.item(),
        "collision_loss": collision_loss.item()
    }

def evaluate(model, loader, device):
    model.eval()

    total = 0
    tile_correct = 0
    tile_total = 0
    agent_correct = 0
    single_agent = 0
    collision_correct = 0

    reward_abs_error = 0.0

    with torch.no_grad():
        for current, action, nxt, reward, done, collision in loader:
            current = current.to(device)
            nxt = nxt.to(device)
            action = action.to(device)
            reward = reward.to(device)
            collision = collision.to(device)

            current_tiles = image_to_tile_classes(current)
            next_tiles = image_to_tile_classes(nxt)

            outputs = model(current_tiles, action)

            pred_tiles = outputs["next_tile_logits"].argmax(dim=1)

            tile_correct += (pred_tiles == next_tiles).sum().item()

            tile_total += next_tiles.numel()

            true_agent = agent_position_targets(next_tiles)
            pred_agent_logits = outputs["next_tile_logits"][:, 4].flatten(1)
            pred_agent = pred_agent_logits.argmax(dim=1)

            agent_correct += (pred_agent == true_agent).sum().item()

            agent_counts = (pred_tiles == 4).sum(dim=(1, 2))
            single_agent += (agent_counts == 1).sum().item()

            pred_collision = torch.sigmoid(outputs["collision_logit"]) > 0.5
            collision_correct += (pred_collision == collision).sum().item()

            reward_abs_error += (outputs["reward"] - reward).abs().sum().item()

            total += current.shape[0]

    return {
        "tile_acc": tile_correct / tile_total,
        "agent_pos_acc": agent_correct / total,
        "single_agent_rate": single_agent / total,
        "collision_acc": collision_correct / total,
        "reward_mae": reward_abs_error / total
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="data/train_transitions.npz")
    parser.add_argument("--val_path", default="data/val_transitions.npz")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    model = SpatialDynamicsModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    train_loader = DataLoader(
        ImageActionTransitionDataset(args.train_path),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = DataLoader(
        ImageActionTransitionDataset(args.val_path),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    best_agent_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total_items = 0

        for current, action, nxt, reward, done, collision in tqdm(train_loader, desc=f"Epoch {epoch}"):
            current = current.to(device)
            nxt = nxt.to(device)
            action = action.to(device)
            reward = reward.to(device)
            done = done.to(device)
            collision = collision.to(device)

            current_tiles = image_to_tile_classes(current)
            next_tiles = image_to_tile_classes(nxt)

            outputs = model(current_tiles, action)
            loss, parts = compute_loss(outputs, current_tiles, next_tiles, reward, done, collision)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * current.shape[0]
            total_items += current.shape[0]

        metrics = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch}: "
            f"train_loss={total_loss / total_items:.6f}, "
            f"tile_acc={metrics['tile_acc']:.4f}, "
            f"agent_pos_acc={metrics['agent_pos_acc']:.4f}, "
            f"single_agent={metrics['single_agent_rate']:.4f}, "
            f"collision_acc={metrics['collision_acc']:.4f}, "
            f"reward_mae={metrics['reward_mae']:.4f}"
        )

        if metrics["agent_pos_acc"] > best_agent_acc:
            best_agent_acc = metrics["agent_pos_acc"]
            path = os.path.join(args.checkpoint_dir, "spatial_dynamics.pt")
            torch.save(model.state_dict(), path)
            print(f"Saved best model to {path}")


if __name__ == "__main__":
    main()