from   itertools import islice, chain, repeat
import torch.utils.data


class MultiTaskDataLoader(object):
    '''
        Serves batches from multiple datasets.

        Iterates over all the datasets (D0, ..., Dk), returning batches of 
            [(B_0, 0), (B_1, 1), ..., (B_k, k)]
        where each B_i has "batch_size" samples
    '''

    def __init__(self, datasets, batch_size=1, use_all=False, **loading_kwargs):
        '''
        Args:
            datasets: A list of datasets to serve batches from
            batch_size: Each batch from each dataset will have this many samples
            use_all: If True, then continues retufning batches until all datasets are exhausted
                     If False, then iteration stops as soon as one dataset runs out
            loading_kwargs: These are passed to each dataset's dataloader
        '''
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
        ''' 
            Iterates over all the datasets (D0, ..., Dk), returning batches of 
                [(B_0, 0), (B_1, 1), ..., (B_k, k)]
            where each B_i has "batch_size" samples
        '''
        return zip_batches(*[
            zip(iter(l), repeat(loader_num)) for loader_num, l in enumerate(self.loaders)
            ],
            use_all=self.use_all)

    def __len__(self):
        if self.use_all:
            return max([len(l) for loader in self.loaders])
        else:
            return self.min_loader_size


def zip_batches(*iterables, use_all=False):
    if use_all:
        try:
            from itertools import izip_longest as zip_longest
        except:
            from itertools import zip_longest 
        return zip_longest(fillvalue=None, *iterables)
    else:
        return zip(*iterables)
