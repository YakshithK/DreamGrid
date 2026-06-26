import argparse
from collections import Counter

import torch
from tqdm import tqdm

from env.constants import ACTION_NAMES
from env.grid import RescueGridEnv
from env.pathfinding import shortest_path, action_between
from planning.mpc_latent import LatentMPCPlanner
from world_model.loading import load_latent_dynamics, load_tile_autoencoder

def oracle_action(env):
    path = shortest_path(env.grid, env.agent_pos, env.goal_pos)

    if path is None or len(path) < 2:
        return None
    
    return action_between(path[0], path[1])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--autoencoder_checkpoint", default="checkpoints/tile_autoencoder_latent128.pt")
    parser.add_argument("--dynamics_checkpoint", default="checkpoints/latent_dynamics_latent128.pt")
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--states", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--candidates", type=int, default=512)
    parser.add_argument("--seed_offset", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    autoencoder = load_tile_autoencoder(args.autoencoder_checkpoint, latent_dim=args.latent_dim, device=device)

    dynamics = load_latent_dynamics(args.dynamics_checkpoint, latent_dim=args.latent_dim, device=device)
    
    planner = LatentMPCPlanner(
        autoencoder= autoencoder,
        dynamics=dynamics,
        device=device,
        horizon=args.horizon,
        candidates=args.candidates
    )

    env = RescueGridEnv()

    total = 0
    matches = 0
    mpc_counts = Counter()
    oracle_counts = Counter()

    for i in tqdm(range(args.states), desc="Evaluating MPC first actions"):
        obs = env.reset(seed=args.seed_offset + i)

        target = oracle_action(env)

        if target is None:
            continue

        action, info =  planner.plan(obs)

        total += 1
        matches += int(action == target)

        mpc_counts[ACTION_NAMES[action]] += 1
        oracle_counts[ACTION_NAMES[target]] += 1

    print()
    print(f"States: {total}")
    print(f"Oracle action match: {matches / total:.4f}")
    print()
    print("MPC action distribution")
    for name, count in mpc_counts.items():
        print(f"   {name}: {count}")

    
    print()
    print("Oracle action distribution")
    for name, count in oracle_counts.items():
        print(f"   {name}: {count}")

if __name__ == "__main__":
    main()