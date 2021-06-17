import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler, Dataset
import numpy as np


class OutlierDataset(Dataset):

    def __init__(self, args, transform, train=True, is_normal=True):
        dataset_method = getattr(datasets, args.dataset)
        self.dataset = dataset_method(root="./data",
                                    train=train,
                                    download=True,
                                    transform=transform)
        labels = np.array(self.dataset.targets)
        self.indices = [i for i in range(len(labels)) if ((is_normal and labels[i] < 5) or ((not is_normal) and labels[i] >= 5))]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset.__getitem__(self.indices[idx])
