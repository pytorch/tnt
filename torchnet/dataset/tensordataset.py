from .dataset import Dataset
import torch
import numpy as np


class TensorDataset(Dataset):
    """
    Dataset from a tensor or array or list or dict.

    `TensorDataset` provides a way to create a dataset out of the data that is
    already loaded into memory. It accepts data in the following forms:

    tensor or numpy array
        `idx`th sample is `data[idx]`

    dict of tensors or numpy arrays
        `idx`th sample is `{k: v[idx] for k, v in data.items()}`

    list of tensors or numpy arrays
        `idx`th sample is `[v[idx] for v in data]`

    Purpose: Easy way to create a dataset out of standard data structures.

    Args:
        data (dict/list/tensor/ndarray): Data for the dataset.
    """

    def __init__(self, data):
        super(TensorDataset, self).__init__()

        if isinstance(data, dict):
            assert len(data) > 0, "Should have at least one element"
            # check that all fields have the same size
            n_elem = len(list(data.values())[0])
            for v in data.values():
                assert len(v) == n_elem, "All values must have the same size"
        elif isinstance(data, list):
            assert len(data) > 0, "Should have at least one element"
            n_elem = len(data[0])
            for v in data:
                assert len(v) == n_elem, "All elements must have the same size"

        self.data = data

    def __len__(self):
        if isinstance(self.data, dict):
            return len(list(self.data.values())[0])
        elif isinstance(self.data, list):
            return len(self.data[0])
        elif torch.is_tensor(self.data) or isinstance(self.data, np.ndarray):
            return len(self.data)

    def __getitem__(self, idx):
        super(TensorDataset, self).__getitem__(idx)
        if isinstance(self.data, dict):
            return {k: v[idx] for k, v in self.data.items()}
        elif isinstance(self.data, list):
            return [v[idx] for v in self.data]
        elif torch.is_tensor(self.data) or isinstance(self.data, np.ndarray):
            return self.data[idx]
