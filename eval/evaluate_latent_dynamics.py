import argparse

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from env.tile_palette import image_to_tile_classes
from models.latent_dynamics import LatentDynamicsModel
from models.tile_autoencoder import TileAutoencoder

class TransitionDataset(Dataset):
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

        action = torch.tensor(self.actions[idx], dtype=torch.long)
        reward = torch.tensor(self.rewards[idx], dtype=torch.float32)
        done = torch.tensor(self.dones[idx], dtype=torch.float32)
        collision = torch.tensor(self.collisions[idx], dtype=torch.float32)

        return current, action, nxt, reward, done, collision


def copy_logits_from_tiles(tile_classes, num_classes=5, strength=8.0, agent_strength=0.0):
    """
    tile_classes: [B, 10, 10]
    returns: [B, 5, 10, 10]

    creates logits saying "predict the current tile unless the model has a reason to change it"
    """

    onehot = F.one_hot(tile_classes, num_classes=num_classes).float()
    onehot = onehot.permute(0, 3, 1, 2)

    strengths = torch.full_like(tile_classes, strength, dtype=torch.float32)
    strengths = strengths.masked_fill(tile_classes == 4, agent_strength)

    return onehot * strengths[:, None, :, :]


def build_copy_residual_tile_logits(outputs, current_image):
    current_tiles = image_to_tile_classes(current_image)
    copy_logits = copy_logits_from_tiles(current_tiles)
    return copy_logits + outputs["tile_delta_logits"]



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data/test_transitions.npz')
    parser.add_argument('--autoencoder_checkpoint', default='checkpoints/tile_autoencoder_latent128.pt')
    parser.add_argument('--dynamics_checkpoint', default='checkpoints/latent_dynamics_latent128.pt')
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loader = DataLoader(
        TransitionDataset(args.data_path),
        batch_size=args.batch_size,
        shuffle=False
    )

    autoencoder = TileAutoencoder(latent_dim=args.latent_dim).to(device)
    autoencoder.load_state_dict(torch.load(args.autoencoder_checkpoint, map_location=device))
    autoencoder.eval()

    dynamics = LatentDynamicsModel(latent_dim=args.latent_dim).to(device)
    dynamics.load_state_dict(torch.load(args.dynamics_checkpoint, map_location=device))
    dynamics.eval()

    total =0

    tile_correct = 0
    tile_total =0

    important_correct = 0
    important_total = 0

    reward_abs_error = 0.0
    done_correct = 0
    collision_correct = 0

    static_important_correct = 0
    static_important_total = 0
    changed_correct = 0
    changed_total = 0

    with torch.no_grad():
        for current, action, nxt, reward, done, collision in loader:
            current = current.to(device)
            action = action.to(device)
            nxt = nxt.to(device)
            reward = reward.to(device)

            done = done.to(device)
            collision = collision.to(device)

            z = autoencoder.encode(current)
            outputs = dynamics(z, action)

            pred_logits = build_copy_residual_tile_logits(outputs, current)
            pred_tiles = pred_logits.argmax(dim=1)
            true_tiles = image_to_tile_classes(nxt)

            current_tiles = image_to_tile_classes(current)

            static_important = (current_tiles == true_tiles) & (true_tiles != 0)
            changed = current_tiles != true_tiles

            static_important_correct += ((pred_tiles == true_tiles) & static_important).sum().item()
            static_important_total += static_important.sum().item()

            changed_correct += ((pred_tiles == true_tiles) & changed).sum().item()
            changed_total += changed.sum().item()

            tile_correct += (pred_tiles == true_tiles).sum().item()
            tile_total += true_tiles.numel()

            important = true_tiles != 0
            important_correct += ((pred_tiles == true_tiles) & important).sum().item()
            important_total += important.sum().item()

            reward_abs_error += (outputs["reward"] - reward).abs().sum().item()

            done_pred = torch.sigmoid(outputs["done_logit"]) > 0.5
            collision_pred = torch.sigmoid(outputs["collision_logit"]) > 0.5

            done_correct += (done_pred.float() == done).sum().item()
            collision_correct += (collision_pred.float() == collision).sum().item()

            total += current.shape[0]

    print(f"tile accuracy: {tile_correct / tile_total:.4f}")
    print(f"important tile accuracy: {important_correct / important_total:.4f}")
    print(f"reward MAE: {reward_abs_error / total:.4f}")
    print(f"done accuracy: {done_correct / total:.4f}")
    print(f"collision accuracy: {collision_correct / total:.4f}")
    print(f"static important tile accuracy: {static_important_correct / max(static_important_total, 1):.4f}")
    print(f"changed tile accuracy: {changed_correct / max(changed_total, 1):.4f}")

if __name__ == "__main__":
    main()