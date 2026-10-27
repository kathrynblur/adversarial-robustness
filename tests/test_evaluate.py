import pytest

from advrobust.attacks import gaussian_noise_attack
from advrobust.evaluate import evaluate_clean, evaluate_fgsm, evaluate_noise, evaluate_pgd
from advrobust.models import BinaryLogReg
from advrobust.train import train


@pytest.fixture
def trained_logreg(separable_binary_loader):
    return train(BinaryLogReg(), separable_binary_loader, binary=True, epochs=30, lr=0.05)


def test_accuracies_are_between_zero_and_one(trained_logreg, separable_binary_loader):
    for acc in [
        evaluate_clean(trained_logreg, separable_binary_loader, binary=True),
        evaluate_fgsm(trained_logreg, separable_binary_loader, 0.1, binary=True),
        evaluate_pgd(trained_logreg, separable_binary_loader, 0.1, binary=True, steps=3),
        evaluate_noise(trained_logreg, separable_binary_loader, 0.1, gaussian_noise_attack, binary=True),
    ]:
        assert 0.0 <= acc <= 1.0


def test_fgsm_lowers_accuracy(trained_logreg, separable_binary_loader):
    clean = evaluate_clean(trained_logreg, separable_binary_loader, binary=True)
    attacked = evaluate_fgsm(trained_logreg, separable_binary_loader, 0.2, binary=True)

    assert attacked < clean


def test_pgd_matches_fgsm_for_binary_logistic_regression(trained_logreg, separable_binary_loader):
    # for a binary linear model, FGSM already finds the worst-case perturbation
    fgsm = evaluate_fgsm(trained_logreg, separable_binary_loader, 0.05, binary=True)
    pgd = evaluate_pgd(trained_logreg, separable_binary_loader, 0.05, binary=True)

    assert abs(fgsm - pgd) <= 0.02
