import torch


def batch_mm(matrix, batch):
    """
    https://github.com/pytorch/pytorch/issues/14489
    """
    return torch.stack([matrix.mm(b) for b in batch], dim=0)


def dot(x, y, sparse=False):
    """Wrapper for torch.matmul (sparse vs dense)."""
    if sparse:
        return batch_mm(x, y)
    else:
        return torch.matmul(x, y)