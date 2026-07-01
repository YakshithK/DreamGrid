import argparse
from collections import Counter

import torch
from tqdm import tqdm

from env.constants import ACTION_NAMES
from env.grid import RescueGridEnv
from env.pathfinding import shortest_path, action_between
from planning.mpc_vq import VQMPCPlanner
from world_model.loading import load_vq_dynamics, load_vqvae

def oracle_action(env):
    path = shortest_path(env.grid, env.agent_pos, env.goal_pos)

    if path is None or len(path) < 2:
        return None
    
    return action_between(path[0], path[1])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vqvae_checkpoint", default="checkpoints/final/vqvae.pt")
    parser.add_argument("--dynamics_checkpoint", default="checkpoints/final/vq_dynamics.pt")
    parser.add_argument("--states", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=1024)
    parser.add_argument("--seed_offset", type=int, default=10000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vqvae = load_vqvae(args.vqvae_checkpoint, device=device)
    dynamics = load_vq_dynamics(args.dynamics_checkpoint, device=device)

    planner = VQMPCPlanner(
        vqvae= vqvae,
        dynamics=dynamics,
        device=device,
        horizon=args.horizon,
        num_candidates=args.candidates
    )

    env = RescueGridEnv()

    total = 0
    matches = 0
    mpc_counts = Counter()
    oracle_counts = Counter()

    for i in tqdm(range(args.states), desc="Evaluating VQ-MPC first actions"):
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
    print("VQ-MPC action distribution")
    for name, count in mpc_counts.items():
        print(f"   {name}: {count}")

    
    print()
    print("Oracle action distribution")
    for name, count in oracle_counts.items():
        print(f"   {name}: {count}")

if __name__ == "__main__":
    main()
