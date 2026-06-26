import argparse

import torch
from tqdm import tqdm

from env.constants import NUM_ACTIONS, ACTION_NAMES
from env.grid import RescueGridEnv
from env.tile_palette import image_to_tile_classes
from world_model.loading import load_tile_autoencoder, load_latent_dynamics
from world_model.rollout import rollout_latent_model


def find_single_agent(tile_classes):
    """
    tile_classes: [10, 10]

    Returns:
    (row, col) if exactly one agent exists, otherwise None
    """
    
    positions = (tile_classes == 4).nonzero(as_tuple=False)
    

    if positions.shape[0] != 1:
        return None
    
    return tuple(positions[0].tolist())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--autoencoder_checkpoint", default="checkpoints/tile_autoencoder_latent128.pt")
    parser.add_argument("--dynamics_checkpoint", default="checkpoints/latent_dynamics_latent128.pt")
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--states", type=int, default=500)
    parser.add_argument("--seed_offset", type=int, default=10000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    autoencoder = load_tile_autoencoder(args.autoencoder_checkpoint, device=device, latent_dim=args.latent_dim)

    dynamics = load_latent_dynamics(args.dynamics_checkpoint, device=device, latent_dim=args.latent_dim)

    env = RescueGridEnv()

    total = 0
    position_correct = 0
    single_agent = 0

    collision_total = 0
    collision_correct = 0

    per_action = {
        action: {
            "total": 0,
            "position_correct": 0,
            "single_agent": 0,
            "collision_correct": 0,
        }
        for action in range(NUM_ACTIONS)
    }

    for i in tqdm(range(args.states), desc="Evaluating dynamics action outcomes"):
        obs = env.reset(seed=args.seed_offset + i)

        for action in range(NUM_ACTIONS):
            probe_env = RescueGridEnv()
            probe_env.grid = env.grid.copy()
            probe_env.agent_pos = env.agent_pos
            probe_env.goal_pos = env.goal_pos
            probe_env.steps = env.steps
            probe_env.done = env.done

            true_next_obs, reward, done, info = probe_env.step(action)
            true_next_tiles = image_to_tile_classes(
                torch.as_tensor(true_next_obs, dtype=torch.float32)
                .permute(2, 0, 1)
                .unsqueeze(0)
                / 255.0
            )[0]

            obs_tensor = (
                torch.as_tensor(obs, device=device, dtype=torch.float32)
                .permute(2, 0, 1)
                .unsqueeze(0)
                / 255.0
            )

            actions = torch.tensor([[action]], device=device)

            with torch.no_grad():
                rollout = rollout_latent_model(
                    autoencoder,
                    dynamics,
                    obs_tensor,
                    actions
                )

            pred_tiles = rollout["pred_tiles"][0, 0]
            pred_collision_prob = rollout["collision_probs"][0, 0].item()
            pred_collision = pred_collision_prob > 0.5

            true_agent = find_single_agent(true_next_tiles)
            pred_agent = find_single_agent(pred_tiles)

            total += 1
            collision_total += 1

            per_action[action]["total"] += 1

            per_action[action]["total"] += 1

            if pred_agent is not None:
                single_agent += 1
                per_action[action]["single_agent"] += 1

            true_collision = bool(info["collision"])

            if pred_collision == true_collision:
                collision_correct += 1
                per_action[action]["collision_correct"] += 1

    print()
    print(f"Total action probes: {total}")
    print(f"Single-agent rate: {single_agent / total:.4f}")
    print(f"Next-agent-position accuracy: {position_correct / total:.4f}")
    print(f"Collision accuracy: {collision_correct / collision_total:.4f}")

    print()
    print("Per action:")
    for action, stats in per_action.items():
        count = stats["total"]

        print(
            f"{ACTION_NAMES[action]}: "
            f"single_agent={stats['single_agent'] / count:.4f}, "
            f"pos_acc={stats['position_correct'] / count:.4f}, "
            f"collision_acc={stats['collision_correct'] / count:.4f}"
        )


if __name__ == "__main__":
    main()