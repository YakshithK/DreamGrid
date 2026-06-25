import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from env.tile_palette import image_to_tile_classes
from world_model.loading import load_latent_dynamics
from world_model.loading import load_tile_autoencoder
from world_model.rollout import rollout_latent_model
from datasets.rollouts import RolloutDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/test_transitions.npz")
    parser.add_argument("--autoencoder_checkpoint", default="checkpoints/tile_autoencoder_latent128.pt")
    parser.add_argument("--dynamics_checkpoint", default="checkpoints/latent_dynamics_latent128.pt")
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = RolloutDataset(args.data_path, horizon=args.horizon)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    autoencoder = load_tile_autoencoder(args.autoencoder_checkpoint, args.latent_dim, device)

    dynamics = load_latent_dynamics(args.dynamics_checkpoint, args.latent_dim, device)

    tile_correct = torch.zeros(args.horizon, device=device)
    tile_total = torch.zeros(args.horizon, device=device)

    important_correct = torch.zeros(args.horizon, device=device)
    important_total = torch.zeros(args.horizon, device=device)

    changed_correct = torch.zeros(args.horizon, device=device)
    changed_total = torch.zeros(args.horizon, device=device)

    single_agent = torch.zeros(args.horizon, device=device)
    total_examples = 0

    reward_abs_error = torch.zeros(args.horizon, device=device)
    done_correct = torch.zeros(args.horizon, device=device)
    collision_correct = torch.zeros(args.horizon, device=device)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating rollouts"):
            start_image = batch["start_image"].to(device)
            actions = batch["actions"].to(device)
            true_images = batch["true_images"].to(device)
            rewards = batch["rewards"].to(device)
            dones = batch["dones"].to(device)
            collisions = batch["collisions"].to(device)

            rollout = rollout_latent_model(autoencoder, dynamics, start_image, actions)

            pred_tiles = rollout["pred_tiles"]
            pred_rewards = rollout["rewards"]
            pred_done = rollout["done_probs"] > 0.5
            pred_collision = rollout["collision_probs"] > 0.5

            batch_size = start_image.shape[0]
            total_examples += batch_size

            prev_true_tiles = image_to_tile_classes(start_image)

            for t in range(args.horizon):
                true_tiles = image_to_tile_classes(true_images[:, t])
                pred_t = pred_tiles[:, t]

                tile_correct[t] += (pred_t == true_tiles).sum()
                tile_total[t] += true_tiles.numel()

                important = true_tiles != 0
                important_correct[t] += ((pred_t == true_tiles) & important).sum()
                important_total[t] += important.sum()

                changed = prev_true_tiles != true_tiles
                changed_correct[t] += ((pred_t == true_tiles) & changed).sum()
                changed_total[t] += changed.sum()

                pred_agent_count = (pred_t == 4).sum(dim=(1, 2))
                single_agent[t] += (pred_agent_count == 1).sum()

                reward_abs_error[t] += torch.abs(pred_rewards[:, t] - rewards[:, t]).sum()
                done_correct[t] += (pred_done[:, t].float() == dones[:, t]).sum()
                collision_correct[t] += (pred_collision[:, t].float() == collisions[:, t]).sum()

                prev_true_tiles = true_tiles

    print(f"Horizon: {args.horizon}")
    for t in range(args.horizon):
        tile_acc = tile_correct[t] / tile_total[t]
        important_acc = important_correct[t] / important_total[t].clamp_min(1)
        changed_acc = changed_correct[t] / changed_total[t].clamp_min(1)
        single_agent_rate = single_agent[t] / total_examples
        reward_mae = reward_abs_error[t] / total_examples
        done_acc = done_correct[t] / total_examples
        collision_acc = collision_correct[t] / total_examples

        print(
            f"t+{t+1}: "
            f"tile_acc={tile_acc.item():.4f}, "
            f"important_acc={important_acc.item():.4f}, "
            f"changed_acc={changed_acc.item():.4f}, "
            f"single_agent={single_agent_rate.item():.4f}, "
            f"reward_mae={reward_mae.item():.4f}, "
            f"done_acc={done_acc.item():.4f}, "
            f"collision_acc={collision_acc.item():.4f}"
        )


if __name__ == "__main__":
    main()