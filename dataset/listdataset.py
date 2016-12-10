from . import dataset
import torch

class ListDataset(dataset.Dataset):
    """
    list can be:
     * string, then read the file line by line
     * torch.LongTensor
     * any other iterable
    """
    def __init__(self, elem_list, load, path=''):
        if isinstance(elem_list, str):
            with open(elem_list) as f:
                self.list = [line[:-1] for line in f]
        elif torch.is_tensor(elem_list):
            assert isinstance(elem_list, torch.LongTensor), \
                    "Only torch.LongTensor supported as list"
            assert elem_list.dim() == 1
            self.list = elem_list
        else:
            # just assume iterable
            self.list = elem_list
        self.path = path 
        self.load = load

    def __len__(self):
        return torch.is_tensor(self.list) and self.list.numel() or len(self.list)

    def __getitem__(self, idx):
        assert idx >= 0 and idx < len(self)
        if self.path != '':
            return self.load("%s/%s" % (self.path, self.list[idx]))
        else:
            return self.load(self.list[idx])
            
