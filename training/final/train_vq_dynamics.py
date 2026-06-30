import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets_utils.transitions import ImageActionTransitionDataset
from env.tile_palette import image_to_tile_classes
from models.final.vq_dynamics import VQDynamics
from world_model.loading import load_vqvae_checkpoint
from eval.metrics import agent_position_targets


def compute_loss(outputs, next_codes, decoded_tile_logits, next_tiles, reward, done, collision):
    code_loss = F.cross_entropy(outputs["next_code_logits"], next_codes)

    pred_agent_logits = decoded_tile_logits[:, 4].flatten(1)
    agent_targets = agent_position_targets(next_tiles)
    agent_loss = F.cross_entropy(pred_agent_logits, agent_targets)

    tile_loss = F.cross_entropy(decoded_tile_logits, next_tiles)

    reward_loss = F.smooth_l1_loss(outputs["reward"], reward)

    done_loss = F.binary_cross_entropy_with_logits(outputs["done_logit"], done)

    collision_loss = F.binary_cross_entropy_with_logits(outputs["collision_logit"], collision)

    total = (
        2.0 * code_loss
        + 2.0 * tile_loss
        + 3.0 * agent_loss
        + 0.25 * reward_loss
        + 0.5 * done_loss
        + 1.0 * collision_loss
    )

    return total, {
        "code_loss": code_loss.item(),
        "tile_loss": tile_loss.item(),
        "agent_loss": agent_loss.item(),
        "reward_loss": reward_loss.item(),
        "done_loss": done_loss.item(),
        "collision_loss": collision_loss.item()
    }

def evaluate(vqvae, dynamics, loader, device):
    vqvae.eval()
    dynamics.eval()

    total = 0
    code_correct = 0
    code_total = 0

    tile_correct = 0
    tile_total = 0

    agent_correct = 0
    single_agent = 0
    collision_correct = 0
    reward_abs_error = 0.0

    with torch.no_grad():
        for current, action, nxt, reward, done, collision in tqdm(loader):
            current = current.to(device)
            nxt = nxt.to(device)
            action = action.to(device)
            reward = reward.to(device)
            done = done.to(device)
            collision = collision.to(device)

            current_codes = vqvae.encode(current)
            next_codes = vqvae.encode(nxt)
            next_tiles = image_to_tile_classes(nxt)

            outputs = dynamics(current_codes, action)

            pred_codes = outputs["next_code_logits"].argmax(dim=1)
            decoded_tile_logits = vqvae.decode_code_tiles(pred_codes)

            pred_tiles = decoded_tile_logits.argmax(dim=1)

            code_correct += (pred_codes == next_codes).sum().item()
            code_total += next_codes.numel()

            tile_correct += (pred_tiles == next_tiles).sum().item()
            tile_total += next_tiles.numel()

            true_agent = agent_position_targets(next_tiles)
            pred_agent = decoded_tile_logits[:, 4].flatten(1).argmax(dim=1)
            agent_correct += (pred_agent == true_agent).sum().item()

            agent_counts = (pred_tiles == 4).sum(dim=(1, 2))
            single_agent += (agent_counts == 1).sum().item()

            pred_collision = torch.sigmoid(outputs["collision_logit"]) > 0.5
            collision_correct += (pred_collision.float() == collision).sum().item()

            reward_abs_error += (outputs["reward"] - reward).abs().sum().item()

            total += current.shape[0]

    return {
        "code_acc": code_correct / code_total,
        "tile_acc": tile_correct / tile_total,
        "agent_pos_acc": agent_correct / total,
        "single_agent_rate": single_agent / total,
        "collision_acc": collision_correct / total,
        "reward_mae": reward_abs_error / total
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="data/transitions/train_transitions.npz")
    parser.add_argument("--val_path", default="data/transitions/val_transitions.npz")
    parser.add_argument("--vqvae_checkpoint", default="checkpoints/final/vqvae.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--checkpoint_dir", default="checkpoints/final")
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vqvae, vqvae_checkpoint = load_vqvae_checkpoint(args.vqvae_checkpoint, device)
    num_codes = vqvae_checkpoint["num_codes"]

    dynamics = VQDynamics(
        num_codes=num_codes,
        hidden_dim=args.hidden_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        dynamics.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    train_loader = DataLoader(
        ImageActionTransitionDataset(args.train_path),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        ImageActionTransitionDataset(args.val_path),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    best_score = 0.0

    for epoch in range(1, args.epochs + 1):
        dynamics.train()

        total_loss = 0.0
        total_items = 0

        for current, action, nxt, reward, done, collision in tqdm(train_loader, desc=f"Epoch {epoch}"):
            current = current.to(device)
            nxt = nxt.to(device)
            action = action.to(device)
            reward = reward.to(device)
            done = done.to(device)
            collision = collision.to(device)

            with torch.no_grad():
                current_codes = vqvae.encode(current)
                next_codes = vqvae.encode(nxt)
                next_tiles = image_to_tile_classes(nxt)

            outputs = dynamics(current_codes, action)
            pred_codes = outputs["next_code_logits"].argmax(dim=1)

            decoded_tile_logits = vqvae.decode_code_tiles(pred_codes)

            loss, parts = compute_loss(
                outputs,
                next_codes,
                decoded_tile_logits,
                next_tiles,
                reward,
                done,
                collision
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * current.shape[0]
            total_items += current.shape[0]

        metrics = evaluate(vqvae, dynamics, val_loader, device)

        print(
            f"Epoch {epoch}: "
            f"train_loss={total_loss / total_items:.6f}, "
            f"code_acc={metrics['code_acc']:.4f}, "
            f"tile_acc={metrics['tile_acc']:.4f}, "
            f"agent_pos_acc={metrics['agent_pos_acc']:.4f}, "
            f"single_agent={metrics['single_agent_rate']:.4f}, "
            f"collision_acc={metrics['collision_acc']:.4f}, "
            f"reward_mae={metrics['reward_mae']:.4f}"
        )

        score = metrics["agent_pos_acc"] + metrics["collision_acc"]

        if score > best_score:
            best_score = score
            path = os.path.join(args.checkpoint_dir, "vq_dynamics.pt")
            torch.save(
                {
                    "model_state_dict": dynamics.state_dict(),
                    "num_codes": num_codes,
                    "hidden_dim": args.hidden_dim
                },
                path
            )
            print(f"Saved best model to {path} with score {best_score:.4f}")

if __name__ == "__main__":
    main()
