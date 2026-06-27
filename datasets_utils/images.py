import numpy as np

import torch
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, path):
        data = np.load(path)
        self.current_images = data['current_images']
        self.next_images = data['next_images']
        self.length = len(self.current_images) * 2

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        real_idx = idx // 2
        use_next = idx % 2 == 1

        if use_next:
            image = self.next_images[real_idx]
        else:
            image = self.current_images[real_idx]

        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        return image