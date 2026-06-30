import argparse

import torch
from planning.mpc_spatial import SpatialMPCPlanner
from planning.policies import RandomPolicy, GreedyPolicy, OracleShortestPathPolicy, PlannerPolicy
from planning.episode import evaluate_policy
from world_model.loading import load_spatial_dynamics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/baselines/spatial_dynamics.pt")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=512)
    parser.add_argument("--seed_offset", type=int, default=10000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_spatial_dynamics(args.checkpoint, device)

    planner = SpatialMPCPlanner(
        model=model,
        device=device,
        horizon=args.horizon,
        num_candidates=args.candidates,
    )

    policies = [
        ("random", RandomPolicy()),
        ("greedy", GreedyPolicy()),
        ("oracle_shortest_path", OracleShortestPathPolicy()),
        ("spatial_mpc", PlannerPolicy(planner)),
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
