import random

import numpy as np
from tqdm import tqdm

from env.grid import RescueGridEnv


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
            if success:
                hazard_death = False
                timeout = False
            elif reward <= -10.0:
                hazard_death = True
                timeout = False
            elif step_info.get("steps", step + 1) >= max_steps:
                hazard_death = False
                timeout = True
            else:
                hazard_death = False
                timeout = False

            return {
                "success": success,
                "hazard_death": hazard_death,
                "timeout": timeout,
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


def summarize_results(results):
    return {
        "success_rate": np.mean([r["success"] for r in results]),
        "hazard_rate": np.mean([r["hazard_death"] for r in results]),
        "timeout_rate": np.mean([r["timeout"] for r in results]),
        "avg_steps": np.mean([r["steps"] for r in results]),
        "avg_reward": np.mean([r["total_reward"] for r in results]),
        "avg_collisions": np.mean([r["collisions"] for r in results]),
    }


def print_summary(name, metrics):
    print()
    print(f"Policy: {name}")
    print(f"Success Rate: {metrics['success_rate']:.3f}")
    print(f"Hazard Death Rate: {metrics['hazard_rate']:.3f}")
    print(f"Timeout Rate: {metrics['timeout_rate']:.3f}")
    print(f"Average Steps: {metrics['avg_steps']:.2f}")
    print(f"Average Reward: {metrics['avg_reward']:.2f}")
    print(f"Average Collisions: {metrics['avg_collisions']:.2f}")


def evaluate_policy(name, policy, num_episodes, seed_offset):
    random.seed(seed_offset)

    env = RescueGridEnv(seed=seed_offset)
    results = []

    for i in tqdm(range(num_episodes), desc=name):
        seed = seed_offset + i
        results.append(run_episode(env, policy, seed))

    metrics = summarize_results(results)
    print_summary(name, metrics)

    return results, metrics