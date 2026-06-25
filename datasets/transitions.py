import numpy as np
import torch
from torch.utils.data import Dataset


class ImageActionTransitionDataset(Dataset):
    def __init__(self, path):
        data = np.load(path)

        self.current_images = data["current_images"]
        self.next_images = data["next_images"]
        self.actions = data["actions"]
        self.rewards = data["rewards"]
        self.dones = data["dones"]
        self.collisions = data["collisions"]

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        current = self.current_images[idx].astype(np.float32) / 255.0
        nxt = self.next_images[idx].astype(np.float32) / 255.0

        current = torch.from_numpy(current).permute(2, 0, 1)
        nxt = torch.from_numpy(nxt).permute(2, 0, 1)

        action = torch.tensor(self.actions[idx], dtype=torch.long)
        reward = torch.tensor(self.rewards[idx], dtype=torch.float32)
        done = torch.tensor(self.dones[idx], dtype=torch.float32)
        collision = torch.tensor(self.collisions[idx], dtype=torch.float32)

        return current, action, nxt, reward, done, collision

class PixelTransitionDataset(Dataset):
    def __init__(self, path):
        data =  np.load(path)

        self.current_images = data["current_images"]
        self.actions = data["actions"]
        self.next_images = data["next_images"]

    def __len__(self):
        return len(self.actions)
    
    def __getitem__(self, idx):
        current = self.current_images[idx].astype(np.float32) / 255.0
        nxt = self.next_images[idx].astype(np.float32) / 255.0
        action = self.actions[idx]

        current = torch.from_numpy(current).permute(2, 0, 1)
        nxt = torch.from_numpy(nxt).permute(2, 0, 1)
        action = torch.tensor(action, dtype=torch.long)

        return current, action, nxt