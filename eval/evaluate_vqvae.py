import argparse
from collections import Counter

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets_utils.images import ImageDataset
from env.tile_palette import image_to_tile_classes
from models.vqvae import VQVAE


def load_vqvae(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = VQVAE(
        num_codes=checkpoint["num_codes"],
        code_dim=checkpoint["code_dim"],
        hidden_dim=checkpoint["hidden_dim"]
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/test_transitions.npz")
    parser.add_argument("--checkpoint", default="checkpoints/vqvae.pt")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_vqvae(args.checkpoint, device)

    loader = DataLoader(
        ImageDataset(args.data_path),
        batch_size=args.batch_size,
        shuffle=False,
    )

    tile_correct = 0
    tile_total = 0

    important_correct = 0
    important_total = 0

    single_agent = 0
    total_images = 0

    code_counter = Counter()

    with torch.no_grad():
        for images in tqdm(loader, desc="Evaluating VQ-VAE"):
            images = images.to(device)
            target_tiles = image_to_tile_classes(images)

            outputs = model(images)

            pred_tiles = outputs["tile_logits"].argmax(dim=1)
            code_ids = outputs["code_ids"]

            tile_correct += (pred_tiles == target_tiles).sum().item()
            tile_total += target_tiles.numel()

            important = target_tiles != 0
            important_correct += ((pred_tiles == target_tiles) & important).sum().item()
            important_total += important.sum().item()

            agent_counts = (pred_tiles == 4).sum(dim=(1, 2))
            single_agent += (agent_counts == 1).sum().item()
            total_images += images.shape[0]

            for code in code_ids.detach().cpu().view(-1).tolist():
                code_counter[int(code)] += 1

    print(f"tile accuracy: {tile_correct / tile_total:.4f}")
    print(f"important tile accuracy: {important_correct / max(important_total, 1):.4f}")
    print(f"single-agent rate: {single_agent / total_images:.4f}")
    print(f"codes used: {len(code_counter)}")
    print("top codes:")
    for code, count in code_counter.most_common(10):
        print(f"  code {code}: {count}")


if __name__ == "__main__":
    main()