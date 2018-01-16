from six import iteritems
from .utils.table import canmergetensor as canmerge
from .utils.table import mergetensor as mergetensor


def compose(transforms):
    assert isinstance(transforms, list)
    for tr in transforms:
        assert callable(tr), 'list of functions expected'

    def composition(z):
        for tr in transforms:
            z = tr(z)
        return z
    return composition


def tablemergekeys():
    def mergekeys(tbl):
        mergetbl = {}
        if isinstance(tbl, dict):
            for idx, elem in tbl.items():
                for key, value in elem.items():
                    if key not in mergetbl:
                        mergetbl[key] = {}
                    mergetbl[key][idx] = value
        elif isinstance(tbl, list):
            for elem in tbl:
                for key, value in elem.items():
                    if key not in mergetbl:
                        mergetbl[key] = []
                    mergetbl[key].append(value)
        return mergetbl
    return mergekeys


def tableapply(f):
    return lambda d: dict(map(lambda kv: (kv[0], f(kv[1])), iteritems(d)))


def makebatch(merge=None):
    if merge:
        makebatch = compose([tablemergekeys(), merge])
    else:
        makebatch = compose([
            tablemergekeys(),
            tableapply(lambda field: mergetensor(field)
                       if canmerge(field) else field)
        ])

    return lambda samples: makebatch(samples)
