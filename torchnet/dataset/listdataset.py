from .dataset import Dataset
import torch


class ListDataset(Dataset):
    """
    Considering a `list` (can be a `string`, `torch.LongTensor`, or any other
    iterable) i-th sample of a dataset will be returned by `load(list[i])`,
    where `load()` is a closure provided by the user.

    If `path` is provided, list is assumed to be a list of string, and will
    each element `list[i]` will prefixed by `path/` when fed to `load()`.
    Purpose: many low or medium-scale datasets can be seen as a list of files
    (for example representing input samples). For this list of file, a target
    can be often inferred in a simple manner.
    """
    def __init__(self, elem_list, load, path=None):
        super(ListDataset, self).__init__()
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
        return len(self.list)

    def __getitem__(self, idx):
        super(ListDataset, self).__getitem__(idx)
        if self.path is not None:
            return self.load("%s/%s" % (self.path, self.list[idx]))
        else:
            return self.load(self.list[idx])
