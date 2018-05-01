from itertools import islice, chain, repeat
import torch.utils.data


class MultiTaskDataLoader(object):
    '''Loads batches simultaneously from multiple datasets.

    The MultiTaskDataLoader is designed to make multi-task learning simpler. It is
    ideal for jointly training a model for multiple tasks or multiple datasets.
    MultiTaskDataLoader is initialzes with an iterable of :class:`Dataset` objects,
    and provides an iterator which will return one batch that contains an equal number
    of samples from each of the :class:`Dataset` s.

    Specifically, it returns batches of  ``[(B_0, 0), (B_1, 1), ..., (B_k, k)]``
    from datasets ``(D_0, ..., D_k)``, where each `B_i` has :attr:`batch_size` samples


    Args:
        datasets: A list of :class:`Dataset` objects to serve batches from
        batch_size: Each batch from each :class:`Dataset` will have this many samples
        use_all (bool): If True, then the iterator will return batches until all
            datasets are exhausted. If False, then iteration stops as soon as one dataset
            runs out
        loading_kwargs: These are passed to the children dataloaders


    Example:
        >>> train_loader = MultiTaskDataLoader([dataset1, dataset2], batch_size=3)
        >>> for ((datas1, labels1), task1), (datas2, labels2), task2) in train_loader:
        >>>     print(task1, task2)
        0 1
        0 1
        ...
        0 1

    '''

    def __init__(self, datasets, batch_size=1, use_all=False, **loading_kwargs):
        self.loaders = []
        self.batch_size = batch_size
        self.use_all = use_all
        self.loading_kwargs = loading_kwargs
        for dataset in datasets:
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                **self.loading_kwargs)
            self.loaders.append(loader)
        self.min_loader_size = min([len(l) for l in self.loaders])
        self.current_loader = 0

    def __iter__(self):
        '''Returns an iterator that simultaneously returns batches from each dataset.
        Specifically, it returns batches of
            [(B_0, 0), (B_1, 1), ..., (B_k, k)]
        from datasets
            (D_0, ..., D_k),

        '''
        return zip_batches(*[zip(iter(l), repeat(loader_num)) for loader_num, l in enumerate(self.loaders)],
                           use_all=self.use_all)

    def __len__(self):
        if self.use_all:
            return max([len(l) for loader in self.loaders])
        else:
            return self.min_loader_size


def zip_batches(*iterables, **kwargs):
    use_all = kwargs.pop('use_all', False)
    if use_all:
        try:
            from itertools import izip_longest as zip_longest
        except ImportError:
            from itertools import zip_longest
        return zip_longest(fillvalue=None, *iterables)
    else:
        return zip(*iterables)
