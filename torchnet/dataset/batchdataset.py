import math
from .dataset import Dataset
from .. import transform


class BatchDataset(Dataset):
    """
    Given a `dataset`, `tnt.BatchDataset` merges samples from this dataset to
    form a new sample which can be interpreted as a batch (of size
    `batchsize`).
    The `merge` function controls how the batch is performed. It is a closure
    taking a Lua array as input containing all occurrences (for a given batch)
    of a field of the sample, and returning the aggregated version of these
    occurrences. By default the occurrences are supposed to be tensors, and
    they aggregated along the first dimension.
    More formally, if the i-th sample of the underlying dataset is written as:
    ```lua
    {input=<input_i>, target=<target_i>}
    ```
    assuming only two fields `input` and `target` in the sample, then `merge()`
    will be passed tables of the form:
    ```lua
    {<input_i_1>, <input_i_2>, ... <input_i_n>}
    ```
    or
    ```lua
    {<target_i_1>, <target_i_2>, ... <target_i_n>}
    ```
    with `n` being the batch size.
    It is often important to shuffle examples while performing the batch
    operation. `perm(idx, size)` is a closure which returns the shuffled index
    of the sample at position `idx` in the underlying dataset. For convenience,
    the `size` of the underlying dataset is also passed to the closure. By
    default, the closure is the identity.
    The underlying dataset size might or might not be always divisible by
    `batchsize`.  The optional `policy` string specify how to handle corner
    cases:
      - `include-last` makes sure all samples of the underlying dataset will be
        seen, batches will be of size equal or inferior to `batchsize`.
      - `skip-last` will skip last examples of the underlying dataset if it's
        size is not properly divisible. Batches will be always of size equal to
        `batchsize`.
      - `divisible-only` will raise an error if the underlying dataset has not
        a size divisible by `batchsize`.
    Purpose: the concept of batch is problem dependent. In *torchnet*, it is up
    to the user to interpret a sample as a batch or not. When one wants to
    assemble samples from an existing dataset into a batch, then
    `tnt.BatchDataset` is suited for the job. Sometimes it is however more
    convenient to write a dataset from scratch providing "batched" samples.
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
        samples = []
        maxidx = len(self.dataset)
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
