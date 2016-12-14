from . import dataset
import torch
import numpy as np

class TensorDataset(dataset.Dataset):
    '''
    Accept:
     * dict of tensors or numpy arrays
     * list of tensors or numpy arrays
     * tensor or numpy array
    '''
    def __init__(self, data):
        if isinstance(data, dict):
            assert len(data) > 0, "Should have at least one element"
            # check that all fields have the same size
            n_elem = len(data.values()[0])
            for v in data.values():
                assert len(v) == n_elem
        elif isinstance(data, list):
            assert len(data) > 0, "Should have at least one element"
            n_elem = len(data[0])
            for v in data:
                assert len(v) == n_elem
        self.data = data

    def __len__(self):
        if isinstance(self.data, dict):
            return len(self.data.values()[0])
        elif isinstance(self.data, list):
            return len(self.data[0])
        elif torch.is_tensor(self.data) or isinstance(self.data, np.ndarray):
            return len(self.data)

    def __getitem__(self, idx):
        if isinstance(self.data, dict):
            return {k: v[idx] for k,v in self.data.items()}
        elif isinstance(self.data, list):
            return [v[idx] for v in self.data]
        elif torch.is_tensor(self.data) or isinstance(self.data, np.ndarray):
            return self.data[idx]

