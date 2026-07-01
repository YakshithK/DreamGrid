import argparse
import os

import matplotlib.pyplot as plt
import torch

from env.constants import ACTION_NAMES, NUM_ACTIONS
from env.grid import RescueGridEnv
from env.tile_palette import tile_classes_to_image
from planning.scoring import score_tile_rollout
from world_model.loading import load_vq_dynamics, load_vqvae
from world_model.rollouts.vq import rollout_vq_model


def format_actions(actions, split=4):
    names = [ACTION_NAMES[int(a)] for a in actions]
    
    if len(names) <= split:
        return " ".join(names)
    

    return " ".join(names[:split]) + "\n" + " ".join(names[split:])


def image_for_plot(image_chw):
    return image_chw.permute(1, 2, 0).detach().cpu().clamp(0, 1)


def clone_env(env):
    cloned = RescueGridEnv()
    cloned.grid = env.grid.copy()
    cloned.agent_pos = env.agent_pos
    cloned.goal_pos = env.goal_pos
    cloned.steps = env.steps
    cloned.done = env.done

    return cloned


def rollout_real_env(env, actions):
    frames = []
    rewards = []
    dones = []
    infos = []

    for action in actions:
        obs, reward, done, info = env.step(int(action))
        frames.append(obs)
        rewards.append(float(reward))
        dones.append(bool(done))
        infos.append(info)

        if done:
            break

    return frames, rewards, dones, infos


def final_status(done, reward, info):
    if not done:
        return "continues"
    
    if info.get("success", False):
        return "success"

    if reward <= -10.0:
        return "hazard"
    
    return "done"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vqvae_checkpoint", default="checkpoints/final/vqvae.pt")
    parser.add_argument("--dynamics_checkpoint", default="checkpoints/final/vq_dynamics.pt")
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=1024)
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument('--out_dir', default="outputs/figures")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vqvae = load_vqvae(args.vqvae_checkpoint, device)
    dynamics = load_vq_dynamics(args.dynamics_checkpoint, device)

    env = RescueGridEnv()
    obs = env.reset(seed=args.seed)
    real_env = clone_env(env)

    obs_tensor = torch.as_tensor(obs, device=device, dtype=torch.float32)

    if obs_tensor.max() > 1.0:
        obs_tensor = obs_tensor / 255.0

    obs_tensor = obs_tensor.permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        encoded = vqvae(obs_tensor)
        start_codes = encoded["code_ids"]
        start_tiles = encoded["tile_logits"].argmax(dim=1)

        candidate_actions = torch.randint(
            low=0,
            high=NUM_ACTIONS -1,
            size=(args.candidates, args.horizon),
            device=device
        )

        start_batch = start_codes.repeat(args.candidates, 1, 1)

        rollout = rollout_vq_model(
            vqvae=vqvae,
            dynamics=dynamics,
            start_codes=start_batch,
            actions=candidate_actions
        )

        scores = score_tile_rollout(
            rollout=rollout,
            start_tiles=start_tiles,
        )

        top_k = min(args.top_k, args.candidates)
        top_scores, top_indices = torch.topk(scores, k=top_k)

        top_actions = candidate_actions[top_indices]
        top_tiles = rollout["pred_tiles"][top_indices]
        top_rewards = rollout["rewards"][top_indices]
        top_done_probs = rollout["done_probs"][top_indices]
        top_collision_probs = rollout["collision_probs"][top_indices]

        flat_tiles = top_tiles.reshape(top_k * args.horizon, 10, 10)
        flat_images = tile_classes_to_image(flat_tiles)
        imagined_images = flat_images.reshape(top_k, args.horizon, 3, 80, 80)

    best_actions = top_actions[0].detach().cpu().tolist()
    real_frames, real_rewards, real_dones, real_infos = rollout_real_env(real_env, best_actions)

    rows = top_k + 1
    cols = args.horizon + 1

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * 2.1, rows * 2.35)
    )

    if rows == 1:
        axes = axes[None, :]

    current_images = obs_tensor[0].detach().cpu()

    for row in range(rows):
        for col in range(cols):
            ax = axes[row, col]
            ax.axis("off")

    for rank in range(top_k):
        actions = top_actions[rank].detach().cpu().tolist()
        score = float(top_scores[rank].item())

        axes[rank, 0].imshow(image_for_plot(current_images))
        axes[rank, 0].set_title(
            f"candidate {rank + 1}\n"
            f"score={score:.2f}\n"
            f"{format_actions(actions)}",
            fontsize=7
        )

        for t in range(args.horizon):
            axes[rank, t+1].imshow(image_for_plot(imagined_images[rank, t]))

            reward = float(top_rewards[rank, t].item())
            done_p = float(top_done_probs[rank, t].item())
            collision_p = float(top_collision_probs[rank, t].item())

            axes[rank, t + 1].set_title(
                f"imagined t+{t + 1}\n"
                f"a={ACTION_NAMES[int(actions[t])]}\n"
                f"r={reward:.2f} d={done_p:.2f} c={collision_p:.2f}",
                fontsize=7
            )

    actual_row = top_k

    axes[actual_row, 0].imshow(image_for_plot(current_images))
    axes[actual_row, 0].set_title(
        f"actual rollout\n"
        "execute best plan",
        fontsize=7
    )

    for t in range(args.horizon):
        col = t + 1

        if t >= len(real_frames):
            axes[actual_row, col].set_title("actual stopped", fontsize=7)
            continue

        frame = torch.as_tensor(real_frames[t], dtype=torch.float32)

        if frame.max() > 1.0:
            frame = frame / 255.0

        frame = frame.permute(2, 0, 1)

        reward = real_rewards[t]
        done = real_dones[t]
        info = real_infos[t]
        status = final_status(done, reward, info)

        axes[actual_row, col].imshow(image_for_plot(frame))
        axes[actual_row, col].set_title(
            f"actual t+{t + 1}\n"
            f"a={ACTION_NAMES[int(best_actions[t])]}\n"
            f"r={reward:.2f} | {status}",
            fontsize=7
        )

    fig.suptitle(
        (
            f"VQ imagination | seed={args.seed} | "
            f"horizon={args.horizon} | candidates={args.candidates} | "
            f"top_k={args.top_k}"
        ),
        fontsize=12
    )

    out_path = os.path.join(
        args.out_dir,
        f"vq_imagination_seed{args.seed}_h{args.horizon}_c{args.candidates}_k{args.top_k}.png"
    )

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(out_path, dpi=260)
    print(f"Saved VQ imagination visualization to {out_path}")

if __name__ == "__main__":
    main()
