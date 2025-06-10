import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class DEAPDataset(Dataset):
    def __init__(self, data_path, label_path, mode="2D"):
        self.data = np.load(data_path).astype(np.float32)
        self.labels = np.load(label_path).astype(np.float32)
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        if self.mode == "1D":
            x = x.reshape(1, -1)

        elif self.mode == "2D":
            x = x[..., np.newaxis]
            x = np.transpose(x, (2, 0, 1))

        elif self.mode == "3D":
            x = x[np.newaxis, :, :, :]

        return torch.tensor(x), torch.tensor(y)