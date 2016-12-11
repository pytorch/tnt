from . import dataset
from tqdm import tqdm

class ProgressBarDataset(dataset.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if idx == 0:
            self.pbar = tqdm(total=len(self.dataset))
        if idx >= len(self):
            self.pbar.close()
            raise IndexError("CustomRange index out of range")
        self.pbar.update()
        return self.dataset[idx]
