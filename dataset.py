import torch
import numpy as np

from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, images, labels):
        super(Dataset, self).__init__()
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return {'image': torch.from_numpy(self.images[idx].reshape(1, 56, 56)).float(), 'label': self.labels[idx]}