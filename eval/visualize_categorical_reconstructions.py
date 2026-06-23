import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from models.categorical_autoencoder import CategoricalAutoencoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/test_transitions.npz')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/categorical_autoencoder_latent64.pt')
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--out_path', default='outputs/categorical_reconstructions.png')
    parser.add_argument('--num_examples', type=int, default=8)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = np.load(args.data_path)
    images = data['next_images']

    model = CategoricalAutoencoder(latent_dim=args.latent_dim).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    indices = np.linspace(0, len(images) - 1, args.num_examples, dtype=int)

    fig, axes = plt.subplots(args.num_examples, 2, figsize=(6, args.num_examples * 2.5))

    with torch.no_grad():
        for row, idx in enumerate(indices):
            image = images[idx].astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image).permute(2, 0, 1)[None].to(device)

            logits, z = model(image_tensor)
            pred_classes = logits.argmax(dim=1)
            recon_rgb = model.palette_indices_to_rgb(pred_classes)
            recon = recon_rgb[0].permute(1, 2, 0).cpu().numpy()

            axes[row, 0].imshow(image)
            axes[row, 0].set_title('Original')

            axes[row, 1].imshow(recon)
            axes[row, 1].set_title(f'Categorical Reconstruction\nlatent={z.shape[-1]}')

            for col in range(2):
                axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(args.out_path, dpi=160)
    print(f"Saved reconstruction visualization to {args.out_path}")

if __name__ == '__main__':
    main()