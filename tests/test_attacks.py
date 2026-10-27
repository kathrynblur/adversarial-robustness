import pytest
import torch
import torch.nn.functional as F

from advrobust.attacks import (
    fgsm_attack,
    gaussian_noise_attack,
    pgd_attack,
    random_sign_attack,
)
from advrobust.models import BinaryCNNMNIST, MulticlassCNNMNIST
from advrobust.train import bce

EPSILONS = [0.01, 0.1, 0.3]


def batch(num_classes):
    x = torch.rand(8, 1, 28, 28)
    y = torch.randint(0, num_classes, (8,))
    return x, y


@pytest.mark.parametrize("epsilon", EPSILONS)
def test_fgsm_stays_within_epsilon_and_pixel_range(epsilon):
    model = MulticlassCNNMNIST()
    x, y = batch(10)

    x_adv = fgsm_attack(model, x, y, epsilon, F.cross_entropy)

    assert x_adv.shape == x.shape
    assert (x_adv - x).abs().max() <= epsilon + 1e-6
    assert x_adv.min() >= 0 and x_adv.max() <= 1


@pytest.mark.parametrize("epsilon", EPSILONS)
def test_pgd_stays_within_epsilon_and_pixel_range(epsilon):
    model = MulticlassCNNMNIST()
    x, y = batch(10)

    x_adv = pgd_attack(model, x, y, epsilon, F.cross_entropy, steps=5)

    assert x_adv.shape == x.shape
    assert (x_adv - x).abs().max() <= epsilon + 1e-6
    assert x_adv.min() >= 0 and x_adv.max() <= 1


def test_one_step_pgd_without_random_start_equals_fgsm():
    model = BinaryCNNMNIST()
    x, y = batch(2)

    x_fgsm = fgsm_attack(model, x, y, 0.1, bce)
    x_pgd = pgd_attack(model, x, y, 0.1, bce, steps=1, step_size=0.1, random_start=False)

    assert torch.allclose(x_fgsm, x_pgd)


@pytest.mark.parametrize("attack", [fgsm_attack, pgd_attack])
def test_attacks_restore_training_mode(attack):
    model = MulticlassCNNMNIST()
    x, y = batch(10)

    model.train()
    attack(model, x, y, 0.1, F.cross_entropy)
    assert model.training

    model.eval()
    attack(model, x, y, 0.1, F.cross_entropy)
    assert not model.training


@pytest.mark.parametrize("noise_fn", [gaussian_noise_attack, random_sign_attack])
@pytest.mark.parametrize("epsilon", EPSILONS)
def test_noise_baselines_stay_within_epsilon_and_pixel_range(noise_fn, epsilon):
    x = torch.rand(8, 1, 28, 28)

    x_noise = noise_fn(x, epsilon)

    assert x_noise.shape == x.shape
    assert (x_noise - x).abs().max() <= epsilon + 1e-6
    assert x_noise.min() >= 0 and x_noise.max() <= 1
