from . import dataset
import torch

class TensorDataset(dataset.Dataset):
    def __init__(self, data):
        assert isinstance(data, dict)
        assert len(data) > 0, "Should have at least one element"
        
        # check that all fields have the same size
        n_elem = len(data.values()[0])
        for v in data.values():
            assert len(v) == n_elem
        self.data = data

    def __len__(self):
        return len(self.data.values()[0])

    def __getitem__(self, idx):
        # sample = {}
        # for k,v in self.data:
        #     sample[k] = v[idx]
        return {k: v[idx] for k,v in self.data.items()}

