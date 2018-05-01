import math
from .dataset import Dataset
from torchnet import transform


class BatchDataset(Dataset):
    """
    Dataset which batches the data from a given dataset.

    Given a `dataset`, `BatchDataset` merges samples from this dataset to
    form a new sample which can be interpreted as a batch of size `batchsize`.

    The `merge` function controls how the batching is performed. By default
    the occurrences are supposed to be tensors, and they aggregated along the
    first dimension.

    It is often important to shuffle examples while performing the batch
    operation. `perm(idx, size)` is a function which returns the shuffled index
    of the sample at position `idx` in the underlying dataset. For convenience,
    the `size` of the underlying dataset is also passed to the function. By
    default, the function is the identity.

    The underlying dataset size might or might not be always divisible by
    `batchsize`.  The optional `policy` string specify how to handle corner
    cases.

    Purpose: the concept of batch is problem dependent. In *torchnet*, it is up
    to the user to interpret a sample as a batch or not. When one wants to
    assemble samples from an existing dataset into a batch, then
    `BatchDataset` is suited for the job. Sometimes it is however more
    convenient to write a dataset from scratch providing "batched" samples.

    Args:
        dataset (Dataset): Dataset to be batched.
        batchsize (int): Size of the batch.
        perm (function, optional): Function used to shuffle the dataset before
            batching. `perm(idx, size)` should return the shuffled index of
            `idx` th sample. By default, the function is the identity.
        merge (function, optional): Function to control batching behaviour.
             `transform.makebatch(merge)` is used to make the batch. Default is
             None.
        policy (str, optional): Policy to handle the corner cases when the
            underlying dataset size is not divisible by `batchsize`. One of
            (`include-last`, `skip-last`, `divisible-only`).

            - `include-last` makes sure all samples of the underlying dataset
               will be seen, batches will be of size equal or inferior to
               `batchsize`.
            - `skip-last` will skip last examples of the underlying dataset if
               its size is not properly divisible. Batches will be always of
               size equal to `batchsize`.
            - `divisible-only` will raise an error if the underlying dataset
               has not a size divisible by `batchsize`.
        filter (function, optional): Function to filter the sample before
            batching. If `filter(sample)` is True, then sample is included for
            batching. Otherwise, it is excluded. By default, `filter(sample)`
            returns True for any `sample`.

    """

    def __init__(self,
                 dataset,
                 batchsize,
                 perm=lambda idx, size: idx,
                 merge=None,
                 policy='include-last',
                 filter=lambda sample: True):
        super(BatchDataset, self).__init__()
        self.dataset = dataset
        self.perm = perm
        self.batchsize = batchsize
        self.policy = policy
        self.filter = filter
        self.makebatch = transform.makebatch(merge)
        len(self)

    def __len__(self):
        if self.policy == 'include-last':
            return int(math.ceil(float(len(self.dataset) / self.batchsize)))
        elif self.policy == 'skip-last':
            return int(math.floor(float(len(self.dataset) / self.batchsize)))
        elif self.policy == 'divisible-only':
            assert len(self.dataset) % self.batchsize == 0, \
                'dataset size is not divisible by batch size'
            return len(self.dataset) / self.batchsize
        else:
            assert False, 'invalid policy (include-last | skip-last | \
                divisible-only expected)'

    def __getitem__(self, idx):
        super(BatchDataset, self).__getitem__(idx)
        maxidx = len(self.dataset)

        samples = []
        for i in range(0, self.batchsize):
            j = idx * self.batchsize + i
            if j >= maxidx:
                break

            j = self.perm(j, maxidx)
            sample = self.dataset[j]

            if self.filter(sample):
                samples.append(sample)

        samples = self.makebatch(samples)
        return samples
