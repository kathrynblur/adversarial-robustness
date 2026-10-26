"""
Data augmentation applied to a batch of images during training
"""

import torch


def random_flip(x):
    """
    Flip each image horizontally with probability 0.5
    """
    flip = torch.rand(x.size(0), device=x.device) < 0.5
    return torch.where(flip[:, None, None, None], x.flip(-1), x)
