import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets_utils.images import ImageDataset
from env.tile_palette import image_to_tile_classes
from models.vqvae import VQVAE


def compute_loss(outputs, images, target_tiles):
    rgb_logits = outputs["rgb_logits"]
    tile_logits = outputs["tile_logits"]
    vq_loss = outputs["vq_loss"]

    recon_rgb = torch.sigmoid(rgb_logits)

    rgb_loss = F.mse_loss(recon_rgb, images)

    per_tile_ce = F.cross_entropy(
        tile_logits,
        target_tiles,
        reduction="none"
    )

    class_weights = torch.tensor(
        [1.0, 5.0, 15.0, 15.0, 30.0],
        device=images.device
    )

    tile_weights = class_weights[target_tiles]
    tile_loss = (per_tile_ce * tile_weights).sum() / tile_weights.sum()

    total = (
        1.0 * rgb_loss
        + 5.0 * tile_loss
        + 1.0 * vq_loss
    )

    return total, {
        "rgb_loss": rgb_loss.item(),
        "tile_loss": tile_loss.item(),
        "vq_loss": vq_loss.item()
    }



def evaluate(model, loader, device):
    model.eval()

    tile_correct = 0
    tile_total = 0

    important_correct = 0
    important_total = 0

    single_agent = 0
    total_images = 0

    rgb_abs_error = 0.0
    perplexity_total = 0.0
    batches = 0

    with torch.no_grad():
        for images in loader:
            images = images.to(device)
            target_tiles = image_to_tile_classes(images)

            outputs = model(images)

            recon_rgb = torch.sigmoid(outputs["rgb_logits"])
            pred_tiles = outputs["tile_logits"].argmax(dim=1)

            tile_correct += (pred_tiles == target_tiles).sum().item()
            tile_total += target_tiles.numel()

            important = target_tiles != 0
            important_correct += ((pred_tiles == target_tiles) & important).sum().item()
            important_total += important.sum().item()

            agent_counts = (pred_tiles == 4).sum(dim=(1, 2))
            single_agent += (agent_counts == 1).sum().item()

            rgb_abs_error += (recon_rgb - images).abs().sum().item()
            total_images += images.shape[0]

    rgb_mae = rgb_abs_error / (total_images * 3 * 80 * 80)

    return {
        "tile_acc": tile_correct / tile_total,
        "important_acc": important_correct / max(important_total, 1),
        "single_agent_rate": single_agent / total_images,
        "rgb_mae": rgb_mae,
        "perplexity": perplexity_total / max(batches, 1)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="data/train_transitions.npz")
    parser.add_argument("--val_path", default="data/val_transitions.npz")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_codes", type=int, default=128)
    parser.add_argument("--code_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = VQVAE(
        num_codes=args.num_codes,
        code_dim=args.code_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    train_loader = DataLoader(
        ImageDataset(args.train_path),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        ImageDataset(args.val_path),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    best_score = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total_items = 0

        parts_total = {
            "rgb_loss": 0.0,
            "tile_loss": 0.0,
            "vq_loss": 0.0,
        }

        for images in tqdm(train_loader, desc=f"Epoch {epoch}"):
            images = images.to(device)
            target_tiles = image_to_tile_classes(images)

            outputs = model(images)
            loss, parts = compute_loss(outputs, images, target_tiles)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = images.shape[0]
            total_loss += loss.item() * batch_size
            total_items += batch_size

            for key in parts_total:
                parts_total[key] += parts[key] * batch_size

        metrics = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch}: "
            f"train_loss={total_loss / total_items:.6f}, "
            f"rgb={parts_total['rgb_loss'] / total_items:.6f}, "
            f"tile={parts_total['tile_loss'] / total_items:.6f}, "
            f"vq={parts_total['vq_loss'] / total_items:.6f}, "
            f"val_tile_acc={metrics['tile_acc']:.4f}, "
            f"val_important_acc={metrics['important_acc']:.4f}, "
            f"single_agent={metrics['single_agent_rate']:.4f}, "
            f"rgb_mae={metrics['rgb_mae']:.4f}, "
            f"perplexity={metrics['perplexity']:.2f}"
        )

        score = metrics["important_acc"] + metrics["single_agent_rate"]

        if score > best_score:
            best_score = score
            path = os.path.join(args.checkpoint_dir, "vqvae.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_codes": args.num_codes,
                    "code_dim": args.code_dim,
                    "hidden_dim": args.hidden_dim,
                },
                path,
            )
            print(f"Saved best model to {path}")


if __name__ == "__main__":
    main()