import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from env.constants import ACTION_NAMES
from models.pixel_model import PixelTransitionModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/test_transitions.npz")
    parser.add_argument("--checkpoint", default="checkpoints/pixel_model.pt")
    parser.add_argument("--out_path", default="outputs/pixel_predictions.png")
    parser.add_argument("--num_examples", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = np.load(args.data_path)
    current_images = data["current_images"]
    actions = data["actions"]
    next_images = data["next_images"]

    model = PixelTransitionModel().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    indices = np.linspace(0, len(actions) - 1, args.num_examples, dtype=int)

    fig, axes = plt.subplots(args.num_examples, 3, figsize=(8, args.num_examples * 2.5))

    with torch.no_grad():
        for row, idx in enumerate(indices):
            current = current_images[idx].astype(np.float32) / 255.0
            true_next = next_images[idx].astype(np.float32) / 255.0
            action = int(actions[idx])

            current_tensor = torch.from_numpy(current).permute(2, 0, 1)[None].to(device)
            action_tensor = torch.tensor([action], dtype=torch.long, device=device)

            pred = model(current_tensor, action_tensor)
            pred = pred[0].permute(1, 2, 0).cpu().numpy()

            axes[row, 0].imshow(current)
            axes[row, 0].set_title(f"Current\nAction: {ACTION_NAMES[action]}")

            axes[row, 1].imshow(true_next)
            axes[row, 1].set_title("True Next")

            axes[row, 2].imshow(pred)
            axes[row, 2].set_title("Predicted Next")

            for col in range(3):
                axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(args.out_path, dpi=160)
    print(f"Saved visualization to {args.out_path}")

if __name__ == "__main__":
    main()