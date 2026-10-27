"""
Shared test fixtures: small random datasets, so the tests need no downloads
"""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def make_loader(n, shape, num_classes, batch_size=16):
    x = torch.rand(n, *shape)
    y = torch.randint(0, num_classes, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)


@pytest.fixture
def separable_binary_loader():
    """
    Linearly separable binary data: the label is 1 if the mean of the left
    half of the image is larger than the mean of the right half
    """
    x = torch.rand(256, 1, 28, 28)
    y = (x[..., :14].mean(dim=(1, 2, 3)) > x[..., 14:].mean(dim=(1, 2, 3))).long()
    return DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)
