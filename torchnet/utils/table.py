import torch


def canmergetensor(tbl):
    if not isinstance(tbl, list):
        return False

    if torch.is_tensor(tbl[0]):
        sz = tbl[0].numel()
        for v in tbl:
            if v.numel() != sz:
                return False
        return True
    return False


def mergetensor(tbl):
    sz = [len(tbl)] + list(tbl[0].size())
    res = tbl[0].new(torch.Size(sz))
    for i, v in enumerate(tbl):
        res[i].copy_(v)
    return res
