import argparse
import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from datasets_utils.images import ImageDataset
from env.tile_palette import tile_classes_to_image
from world_model.loading import load_vqvae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/transitions/test_transitions.npz")
    parser.add_argument("--checkpoint", default="checkpoints/final/vqvae.pt")
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--out_path", default="outputs/vqvae_reconstructions.png")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = load_vqvae(args.checkpoint, device)

    loader = DataLoader(
        ImageDataset(args.data_path),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,              
    )

    images = next(iter(loader)).to(device)

    with torch.no_grad():
        outputs = model(images)
        recon_rgb = torch.sigmoid(outputs["rgb_logits"])
        pred_tiles = outputs["tile_logits"].argmax(dim=1)
        tile_recon = tile_classes_to_image(pred_tiles)

    images = images.cpu()
    recon_rgb = recon_rgb.cpu()
    tile_recon = tile_recon.cpu()

    n = images.shape[0]

    fig, axes = plt.subplots(3, n, figsize=(n * 1.8, 5.4))

    for i in range(n):
        axes[0, i].imshow(images[i].permute(1, 2, 0))
        axes[0, i].axis("off")
        axes[0, i].set_title("Input")

        axes[1, i].imshow(recon_rgb[i].permute(1, 2, 0).clamp(0, 1))
        axes[1, i].axis("off")
        axes[1, i].set_title("rgb", fontsize=8)

        axes[2, i].imshow(tile_recon[i].permute(1, 2, 0).clamp(0, 1))
        axes[2, i].axis("off")
        axes[2, i].set_title("tiles", fontsize=8)

    plt.tight_layout()
    plt.savefig(args.out_path, dpi=160)
    print(f"Saved reconstruction visualization to {args.out_path}")


if __name__ == "__main__":
    main()
