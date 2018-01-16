from .dataset import Dataset


class ListDataset(Dataset):
    """
    Dataset which loads data from a list using given function.

    Considering a `elem_list` (can be an iterable or a `string` ) i-th sample
    of a dataset will be returned by `load(elem_list[i])`, where `load()`
    is a function provided by the user.

    If `path` is provided, `elem_list` is assumed to be a list of strings, and
    each element `elem_list[i]` will prefixed by `path/` when fed to `load()`.

    Purpose: many low or medium-scale datasets can be seen as a list of files
    (for example representing input samples). For this list of file, a target
    can be often inferred in a simple manner.

    Args:
        elem_list (iterable/str): List of arguments which will be passed to
            `load` function. It can also be a path to file with each line
            containing the arguments to `load`
        load (function, optional): Function which loads the data.
            i-th sample is returned by `load(elem_list[i])`. By default `load`
            is identity i.e, `lambda x: x`
        path (str, optional): Defaults to None. If a string is provided,
            `elem_list` is assumed to be a list of strings, and each element
            `elem_list[i]` will prefixed by this string when fed to `load()`.

    """

    def __init__(self, elem_list, load=lambda x: x, path=None):
        super(ListDataset, self).__init__()

        if isinstance(elem_list, str):
            with open(elem_list) as f:
                self.list = [line.replace('\n', '') for line in f]
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
