import numpy as np
import torch
from torch.utils.data import Dataset

class RolloutDataset(Dataset):
    def __init__(self, path, horizon=5):
        data = np.load(path)

        self.current_images = data["current_images"]
        self.next_images = data["next_images"]
        self.actions = data["actions"]
        self.rewards = data["rewards"]
        self.dones = data["dones"]
        self.collisions = data["collisions"]
        self.map_seeds = data["map_seeds"]

        self.horizon = horizon
        self.valid_starts = self._find_valid_starts()

    def _find_valid_starts(self):
        valid = []

        max_start = len(self.actions) - self.horizon

        for idx in range(max_start):
            start_seed = self.map_seeds[idx]

            ok = True

            for k in range(self.horizon):
                if self.map_seeds[idx + k] != start_seed:
                    ok = False
                    break

                if k < self.horizon - 1 and self.dones[idx + k]:
                    ok = False
                    break
            
            if ok:
                valid.append(idx)

        return valid
    
    def __len__(self):
        return len(self.valid_starts)
    
    def __getitem__(self, item):
        idx = self.valid_starts[item]

        start_image = self.current_images[idx].astype(np.float32) / 255.0
        start_image = torch.from_numpy(start_image).permute(2, 0, 1)

        actions = []
        true_imgs = []
        rewards = []
        dones = []
        collisions = []

        for k in range(self.horizon):
            step_idx = idx + k

            actions.append(self.actions[step_idx])

            true_next = self.next_images[step_idx].astype(np.float32) / 255.0
            true_next = torch.from_numpy(true_next).permute(2, 0, 1)
            true_imgs.append(true_next)

            rewards.append(self.rewards[step_idx])
            dones.append(self.dones[step_idx])
            collisions.append(self.collisions[step_idx])

        return {
            "start_image": start_image,
            "actions": torch.tensor(actions, dtype=torch.long),
            "true_images": torch.stack(true_imgs, dim=0),
            "rewards": torch.tensor(rewards, dtype=torch.float32),
            "dones": torch.tensor(dones, dtype=torch.float32),
            "collisions": torch.tensor(collisions, dtype=torch.float32)
        }