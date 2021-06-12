import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler, Dataset
import numpy as np

class CifarOutlierDataset(Dataset):

    def __init__(self, args, transform, train=True, is_normal=True):
        # todo super(CifarOutlierDataset, self).__init__() ?
        self.dataset = datasets.CIFAR10(root="./data",
                                    train=train,
                                    download=True,
                                    transform=transform)

        self.data = np.array(self.dataset.data)
        self.labels = np.array(self.dataset.targets)
        self.indices = [i for i in range(len(self.labels)) if ((is_normal and self.labels[i] < 5) or ((not is_normal) and self.labels[i] >= 5))]
        # self.data = self.data[indices]
        # self.labels = self.labels[indices]  #todo optimize memory


    def __len__(self):
        return len(self.indices)


    def __getitem__(self, idx):
        return self.dataset.__getitem__(self.indices[idx])
        # return self.data[idx], self.labels[idx]
