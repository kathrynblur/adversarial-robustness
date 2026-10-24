"""
Evaluation: accuracy under FGSM attacks or random-noise perturbations

Binary models output probabilities (threshold 0.5)
Multiclass models output logits (argmax)
"""

import torch
import torch.nn.functional as F

from .attacks import fgsm_attack
from .train import bce
from .utils import device


def predict(model, x, binary):
    out = model(x)
    if binary:
        return (out > 0.5).long()
    return out.argmax(1)


def accuracy_under_perturbation(model, loader, perturb, binary):
    """
    perturb(x, y) returns the perturbed batch
    """
    model.eval()

    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        x_pert = perturb(x, y)

        with torch.no_grad():
            preds = predict(model, x_pert, binary)

        correct += (preds == y).sum().item()
        total += x.size(0)

    return correct / total


def evaluate_fgsm(model, loader, epsilon, binary):
    loss_fn = bce if binary else F.cross_entropy
    return accuracy_under_perturbation(
        model, loader,
        lambda x, y: fgsm_attack(model, x, y, epsilon, loss_fn),
        binary,
    )


def evaluate_noise(model, loader, epsilon, noise_fn, binary):
    return accuracy_under_perturbation(
        model, loader,
        lambda x, y: noise_fn(x, epsilon),
        binary,
    )
