import argparse

import torch

from planning.mpc_latent import LatentMPCPlanner
from planning.policies import RandomPolicy, GreedyPolicy, OracleShortestPathPolicy, PlannerPolicy
from planning.episode import evaluate_policy
from world_model.loading import load_latent_dynamics, load_tile_autoencoder
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--autoencoder_checkpoint", default="checkpoints/tile_autoencoder_latent128.pt")
    parser.add_argument("--dynamics_checkpoint", default="checkpoints/latent_dynamics_latent128.pt")
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=512)
    parser.add_argument("--seed_offset", type=int, default=10000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    autoencoder = load_tile_autoencoder(args.autoencoder_checkpoint, device=device, latent_dim=args.latent_dim)

    dynamics = load_latent_dynamics(args.dynamics_checkpoint, device=device, latent_dim=args.latent_dim)

    planner = LatentMPCPlanner(
        autoencoder=autoencoder,
        dynamics=dynamics,
        device=device,
        horizon=args.horizon,
        num_candidates=args.candidates,
    )

    policies = [
        ("random", RandomPolicy()),
        ("greedy", GreedyPolicy()),
        ("oracle_shortest_path", OracleShortestPathPolicy()),
        ("learned_mpc", PlannerPolicy(planner)),
    ]

    for name, policy in policies:
        evaluate_policy(name, policy, num_episodes=args.episodes, seed_offset=args.seed_offset)

if __name__ == "__main__":
    main()