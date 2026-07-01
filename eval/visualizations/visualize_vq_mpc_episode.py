import argparse
import os

import matplotlib.pyplot as plt
import torch

from env.constants import ACTION_NAMES, MAX_STEPS
from env.grid import RescueGridEnv
from planning.mpc_vq import VQMPCPlanner
from world_model.loading import load_vq_dynamics, load_vqvae


def format_plan(action_sequence, max_items=4):
    if action_sequence is None:
        return ""
    
    names = [ACTION_NAMES[a] for a in action_sequence[:max_items]]

    if len(action_sequence) > max_items:
        names.append("...")

    return " ".join(names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vqvae_checkpoint", default="checkpoints/final/vqvae.pt")
    parser.add_argument("--dynamics_checkpoint", default="checkpoints/final/vq_dynamics.pt")
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=1024)
    parser.add_argument("--max_steps", type=int, default=40)
    parser.add_argument("--out_dir", default="outputs/figures")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vqvae = load_vqvae(args.vqvae_checkpoint, device)
    dynamics = load_vq_dynamics(args.dynamics_checkpoint, device)

    planner = VQMPCPlanner(
        vqvae,
        dynamics,
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
        action, plan_info = planner.plan(obs)

        obs, reward, done, step_info = env.step(action)

        total_reward += reward

        best_sequence = plan_info.get("best_sequence", [])
        best_score = plan_info.get("best_score", 0.0)

        title = (
            f"t={step + 1} | {ACTION_NAMES[action]}\n"
            f"reward={reward:.2f} | score={best_score:.2f}\n"
            f"plan: {format_plan(best_sequence)}"
        )

        frames.append(obs)
        titles.append(title)

        if done:
            if step_info.get("success", False):
                final_status = "success"
            elif reward <= -10.0:
                final_status = "hazard"
            elif step_info.get("steps", 0) >= MAX_STEPS:
                final_status = "timeout"
            else:
                final_status = "done"
            break

    num_frames = len(frames)
    cols = 8
    rows = (num_frames + cols - 1) // cols

    fig, axes= plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.5))
    axes = axes.reshape(-1)

    for i, ax in enumerate(axes):
        ax.axis("off")

        if i >= num_frames:
            continue

        ax.imshow(frames[i])
        ax.set_title(titles[i], fontsize=8)

    fig.suptitle(
        (
            f"VQ-MPC episode | seed={args.seed} | "
            f"horizon={args.horizon} | candidates={args.candidates} | "
            f"status={final_status} | reward={total_reward:.2f} | "
            f"steps={len(frames) - 1}"
        ),
        fontsize=12,
    )

    out_path = os.path.join(
        args.out_dir,
        f"vq_mpc_episode_seed{args.seed}_h{args.horizon}_c{args.candidates}.png",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig(out_path, dpi=300)
    print(f"Saved VQ-MPC episode visualization to {out_path}")


if __name__ == "__main__":
    main()
