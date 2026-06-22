import argparse
import os

import numpy as np
from tqdm import tqdm

from env.constants import NUM_ACTIONS, MAX_STEPS
from env.grid import RescueGridEnv
from env.pathfinding import shortest_path, action_between

def choose_action(env, mode, rng):
    if mode == "random":
        return int(rng.integers(0, NUM_ACTIONS))
    
    path = shortest_path(env.grid, env.agent_pos, env.goal_pos)

    if path is None or len(path) <= 2:
        return int(rng.integers(0, NUM_ACTIONS))
    

    expert_action = action_between(path[0], path[1])

    if mode == "expert":
        return expert_action
    
    if mode == "noisy_expert":
        if rng.random() < 0.25:
            return int(rng.integers(0, NUM_ACTIONS))
        return expert_action
    
    raise ValueError(f"Unknown mode: {mode}")
    
def generate_split(output_path, num_transitions, seed):
    rng = np.random.default_rng(seed)
    env = RescueGridEnv(seed=seed)

    current_images = []
    actions = []
    next_images = []
    rewards = []
    dones = []
    collisions = []
    current_positions = []
    next_positions = []
    map_seeds = []

    modes = ["random", "expert", "noisy_expert"]

    probs = [0.4, 0.3, 0.3]

    obs = env.reset()
    current_map_seed = seed

    for _ in tqdm(range(num_transitions), desc=f"Generating {output_path}"):
        if env.done or env.steps >= MAX_STEPS:
            current_map_seed = int(rng.integers(0, 1_000_000_000))
            obs = env.reset(seed=current_map_seed)


        mode = rng.choice(modes, p=probs)
        action = choose_action(env, mode, rng)

        current_pos = env.agent_pos
        next_obs, reward, done, info = env.step(action)
        next_pos = info["agent_pos"]

        current_images.append(obs)
        actions.append(action)
        next_images.append(next_obs)
        rewards.append(reward)
        dones.append(done)
        collisions.append(info["collision"])
        current_positions.append(current_pos)
        next_positions.append(next_pos)
        map_seeds.append(current_map_seed)

        obs = next_obs


    np.savez_compressed(
        output_path,
        current_images=np.array(current_images, dtype=np.uint8),
        actions = np.array(actions, dtype=np.int64),
        next_images=np.array(next_images, dtype=np.uint8),
        rewards=np.array(rewards, dtype=np.float32),
        dones=np.array(dones, dtype=np.bool_),
        collisions=np.array(collisions, dtype=np.bool_),
        current_positions=np.array(current_positions, dtype=np.int64),
        next_positions=np.array(next_positions, dtype=np.int64),
        map_seeds=np.array(map_seeds, dtype=np.int64)
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data")
    parser.add_argument("--train", type=int, default=50000)
    parser.add_argument("--val", type=int, default=5000)
    parser.add_argument("--test", type=int, default=5000)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    generate_split(os.path.join(args.out_dir, "train_transitions.npz"), args.train, seed=1)
    generate_split(os.path.join(args.out_dir, "val_transitions.npz"), args.val, seed=2)
    generate_split(os.path.join(args.out_dir, "test_transitions.npz"), args.test, seed=3)

if __name__ == "__main__":
    main()