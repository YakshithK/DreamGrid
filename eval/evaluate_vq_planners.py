import argparse

import torch

from planning.mpc_vq import VQMPCPlanner
from planning.policies import RandomPolicy, GreedyPolicy, OracleShortestPathPolicy, PlannerPolicy
from planning.episode import evaluate_policy
from world_model.loading import load_vqvae, load_vq_dynamics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vqvae_checkpoint", default="checkpoints/vqvae.pt")
    parser.add_argument("--dynamics_checkpoint", default="checkpoints/vq_dynamics.pt")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=1024)
    parser.add_argument("--seed_offset", type=int, default=10000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vqvae = load_vqvae(args.vqvae_checkpoint, device)
    dynamics = load_vq_dynamics(args.dynamics_checkpoint, device)

    planner = VQMPCPlanner(
        vqvae=vqvae,
        dynamics=dynamics,
        device=device,
        horizon=args.horizon,
        num_candidates=args.candidates,
    )

    policies = [
        ("random", RandomPolicy()),
        ("greedy", GreedyPolicy()),
        ("oracle_shortest_path", OracleShortestPathPolicy()),
        ("vq_mpc", PlannerPolicy(planner)),
    ]

    for name, policy in policies:
        evaluate_policy(
            name,
            policy,
            num_episodes=args.episodes,
            seed_offset=args.seed_offset,
        )


if __name__ == "__main__":
    main()