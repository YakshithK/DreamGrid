import argparse
import os

import matplotlib.pyplot as plt
import torch


from env.constants import ACTION_NAMES
from env.grid import RescueGridEnv
from models.spatial_dynamics import SpatialDynamicsModel
from planning.mpc_spatial import SpatialMPCPlanner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/spatial_dynamics.pt")
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=40)
    parser.add_argument("--out_path", default="outputs/spatial_mpc_episode.png")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SpatialDynamicsModel().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    planner = SpatialMPCPlanner(
        model,
        device,
        args.horizon,
        args.candidates
    )

    env = RescueGridEnv()
    obs = env.reset(seed=args.seed)

    frames = [obs]
    titles = [f"t=0\nstart"]

    total_reward = 0.0
    final_status = "timeout"

    for step in range(args.max_steps):
        action, info = planner.plan(obs)

        obs, reward, done, step_info = env.step(action)

        total_reward += reward

        frames.append(obs)

        title = (
            f"t={step+1}\n"
            f"{ACTION_NAMES[action]}\n"
            f"r={reward:.2f}\n"
        )
        titles.append(title)

        if done:
            if step_info.get("success", False):
                final_status = "success"
            elif reward <= -10.0:
                final_status = "hazard"
            else:
                final_status = "done"
            break

    num_frames = len(frames)
    cols = 8
    rows = (num_frames + cols - 1) // cols

    fig, axes= plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.25))
    axes = axes.reshape(-1)

    for i, ax in enumerate(axes):
        ax.axis("off")

        if i >= num_frames:
            continue

        ax.imshow(frames[i])
        ax.set_title(titles[i], fontsize=8)

    fig.suptitle(
        (
            f"Spatial MPC episode | seed={args.seed} | "
            f"status={final_status} | total_reward={total_reward:.2f} |"
            f"steps={len(frames) - 1}"
        ),
        fontsize=12
    )

    plt.tight_layout()
    plt.savefig(args.out_path, dpi=300)
    print(f"Saved episode visualization to {args.out_path}")


if __name__ == "__main__":
    main()