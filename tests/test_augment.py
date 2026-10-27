import torch

from advrobust.augment import random_flip


def test_random_flip_keeps_or_flips_each_image():
    x = torch.rand(32, 3, 8, 8)

    out = random_flip(x)

    assert out.shape == x.shape
    for original, augmented in zip(x, out):
        assert torch.equal(augmented, original) or torch.equal(augmented, original.flip(-1))


def test_random_flip_flips_some_but_not_all_images():
    x = torch.rand(64, 3, 8, 8)

    out = random_flip(x)
    n_flipped = sum(not torch.equal(a, b) for a, b in zip(out, x))

    assert 0 < n_flipped < 64
