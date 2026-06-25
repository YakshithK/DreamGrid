import argparse
import random

import numpy as np
import torch
from tqdm import tqdm 

from env.constants import NUM_ACTIONS
from env.grid import RescueGridEnv
from env.pathfinding import shortest_path, action_between
from planning.mpc_latent import LatentMPCPlanner
from world_model.loading import load_latent_dynamics, load_tile_autoencoder

class RandomPolicy:
    def act(self, obs, env):
        return random.randrange(NUM_ACTIONS), {}


class GreedyPolicy:
    def act(self, obs, env):
        best_action = 4
        best_dist = manhattan(env.agent_post, env.goal_pos)

        for action, delta in env_action_deltas().items():
            nr = env.agent_post[0] + delta[0]
            nc = env.agent_post[1] + delta[1]
            pos = (nr, nc)

            if not env._in_bounds(pos):
                continue

            if env.grid[pos] == 1:
                continue

            dist = manhattan(pos, env.goal_pos)

            if dist < best_dist:
                best_dist = dist
                best_action = action

        return best_action, {}
    
class OracleShortestPathPolicy:
    def act(self, obs, env):
        path = shortest_path(env.agent_post, env.goal_pos, env.grid)
        
        if path is None or len(path) < 2:
            return random.randint(0, NUM_ACTIONS - 1), {"path_found": False}
        
        action = action_between(path[0], path[1])
        return action, {"path_found": True}


class LearnedMPCPolicy:
    def __init__(self, planner):
        self.planner = planner

    def act(self, obs, env):
        return self.planner.plan(obs)
    

def env_action_deltas():
    from env.constants import ACTION_DELTAS
    return ACTION_DELTAS

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def run_episode(env, policy, seed, max_steps=40):
    obs = env.reset(seed=seed)

    total_reward = 0.0
    collisions = 0
    success = False
    hazard_death = False
    planner_infos = []

    for step in range(max_steps):
        action, info = policy.act(obs, env)

        obs, reward, done, step_info = env.step(action)

        total_reward += reward
        collisions += int(step_info.get("collision", False))
        success = bool(step_info.get("success", False))

        planner_infos.append(info)

        if done:
            if not success and reward <= -10.0:
                hazard_death = True

            return {
                "success": success,
                "hazard_death": hazard_death,
                "timeout": False,
                "steps": step + 1,
                "total_reward": total_reward,
                "collisions": collisions,
                "planner_infos": planner_infos,
            }

    return {
        "success": success,
        "hazard_death": hazard_death,
        "timeout": True,
        "steps": max_steps,
        "total_reward": total_reward,
        "collisions": collisions,
        "planner_infos": planner_infos,
    }

def evaluate_policy(name, policy, num_episodes, seed_offset):
    env = RescueGridEnv(seed=seed_offset)

    results = []

    for i in tqdm(range(num_episodes), desc=name):
        seed = seed_offset + i
        result = run_episode(env, policy, seed)
        results.append(result)

    success_rate = np.mean([r["success"] for r in results])
    hazard_rate = np.mean([r["hazard_death"] for r in results])
    timeout_rate = np.mean([r["timeout"] for r in results])
    avg_steps = np.mean([r["steps"] for r in results])
    avg_reward = np.mean([r["total_reward"] for r in results])
    avg_collisions = np.mean([r["collisions"] for r in results])

    print()
    print(f"Policy: {name}")
    print(f"Success Rate: {success_rate:.3f}")
    print(f"Hazard Death Rate: {hazard_rate:.3f}")
    print(f"Timeout Rate: {timeout_rate:.3f}")
    print(f"Average Steps: {avg_steps:.2f}")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Collisions: {avg_collisions:.2f}")

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--autoencoder_checkpoint", default="checkpoints/tile_autoencoder_latent128.pt")
    parser.add_argument("--dynamics_checkpoint", default="checkpoints/latent_dynamics_latent128.pt")
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--candidates", type=int, default=128)
    parser.add_argument("--seed_offset", type=int, default=10000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    autoencoder = load_tile_autoencoder(args.autoencoder_checkpoint, device=device)

    dynamics = load_latent_dynamics(args.dynamics_checkpoint, device=device)

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
        ("learned_mpc", LearnedMPCPolicy(planner)),
    ]

    for name, policy in policies:
        evaluate_policy(name, policy, num_episodes=args.episodes, seed_offset=args.seed_offset)

if __name__ == "__main__":
    main()