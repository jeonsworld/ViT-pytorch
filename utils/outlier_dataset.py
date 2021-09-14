from torch.utils.data import Dataset


class OutlierDataset(Dataset):

    def __init__(self, main_dataset, transform):
        self.dataset = main_dataset.dataset
        indices = main_dataset.indices
        final_indices = []
        for i in indices:
            if main_dataset.dataset.targets[i] >= 7:
                final_indices.append(i)
        self.indices = final_indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        item = self.dataset[self.indices[idx]]
        return (self.transform(item[0]), item[1])