import argparse

import torch
from tqdm import tqdm

from env.constants import ACTION_NAMES, NUM_ACTIONS
from env.grid import RescueGridEnv
from env.tile_palette import image_to_tile_classes
from world_model.loading import load_spatial_dynamics
from eval.metrics import find_single_agent


def clone_env(env):
    cloned = RescueGridEnv()
    cloned.grid = env.grid.copy()
    cloned.agent_pos = env.agent_pos
    cloned.goal_pos = env.goal_pos
    cloned.steps = env.steps
    cloned.done = env.done
    return cloned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/spatial_dynamics.pt")
    parser.add_argument("--states", type=int, default=500)
    parser.add_argument("--seed_offset", type=int, default=10000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_spatial_dynamics(args.checkpoint, device)

    env = RescueGridEnv()

    total = 0
    single_agent = 0
    position_correct = 0
    collision_correct = 0

    per_action = {
        action: {
            "total": 0,
            "single_agent": 0,
            "position_correct": 0,
            "collision_correct": 0,
        }
        for action in range(NUM_ACTIONS)
    }

    with torch.no_grad():
        for i in tqdm(range(args.states), desc="Evaluating spatial dynamics"):
            obs = env.reset(seed=args.seed_offset + i)

            obs_tensor = (
                torch.as_tensor(obs, device=device, dtype=torch.float32)
                .permute(2, 0, 1)
                .unsqueeze(0)
                / 255.0
            )

            current_tiles = image_to_tile_classes(obs_tensor)

            for action in range(NUM_ACTIONS):
                probe_env = clone_env(env)
                true_next_obs, reward, done, info = probe_env.step(action)

                true_next_tensor = (
                    torch.as_tensor(true_next_obs, device=device, dtype=torch.float32)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    / 255.0
                )

                true_tiles = image_to_tile_classes(true_next_tensor)[0]

                action_tensor = torch.tensor([action], device=device)
                outputs = model(current_tiles, action_tensor)

                pred_tiles = outputs["next_tile_logits"].argmax(dim=1)[0]
                pred_collision = torch.sigmoid(outputs["collision_logit"])[0].item() > 0.5

                true_agent = find_single_agent(true_tiles)
                pred_agent = find_single_agent(pred_tiles)

                true_collision = bool(info["collision"])

                total += 1
                per_action[action]["total"] += 1

                if pred_agent is not None:
                    single_agent += 1
                    per_action[action]["single_agent"] += 1

                if pred_agent == true_agent:
                    position_correct += 1
                    per_action[action]["position_correct"] += 1

                if pred_collision == true_collision:
                    collision_correct += 1
                    per_action[action]["collision_correct"] += 1

    print()
    print(f"Total action probes: {total}")
    print(f"Single-agent rate: {single_agent / total:.4f}")
    print(f"Next-agent-position accuracy: {position_correct / total:.4f}")
    print(f"Collision accuracy: {collision_correct / total:.4f}")

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