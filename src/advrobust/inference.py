"""
Inference helpers for the API and the demo: load the MNIST models, preprocess an
uploaded image, predict, attack, and compute a robustness score

Everything runs on the CPU, so it works on any server
"""

import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .attacks import fgsm_attack, pgd_attack
from .models import MulticlassCNNMNIST

CPU = torch.device("cpu")

# epsilons tried when searching for the smallest perturbation that changes the prediction
SCORE_EPSILONS = [round(0.02 * i, 2) for i in range(1, 16)]  # 0.02 ... 0.30


def load_model(path):
    """
    Load a MulticlassCNNMNIST from a state_dict file
    """
    model = MulticlassCNNMNIST()
    model.load_state_dict(torch.load(path, map_location=CPU))
    return model.eval()


def preprocess(image):
    """
    Turn a PIL image into a (1, 1, 28, 28) tensor in [0, 1], like MNIST:
    grayscale, white digit on a black background
    """
    image = image.convert("L").resize((28, 28), Image.BILINEAR)
    x = torch.from_numpy(np.array(image, dtype=np.float32) / 255.0)

    # MNIST digits are white on black; invert dark-on-light images
    if x.mean() > 0.5:
        x = 1.0 - x

    return x.view(1, 1, 28, 28)


def preprocess_bytes(data):
    return preprocess(Image.open(io.BytesIO(data)))


def predict_proba(model, x):
    """
    Return the class probabilities for a single image, shape (10,)
    """
    with torch.no_grad():
        return F.softmax(model(x), dim=1)[0]


def attack(model, x, method, epsilon, pgd_steps=20):
    """
    Attack a single image. The attack targets the model's own current prediction
    """
    if method == "none" or epsilon == 0:
        return x

    y = predict_proba(model, x).argmax().view(1)

    if method == "fgsm":
        return fgsm_attack(model, x, y, epsilon, F.cross_entropy)
    if method == "pgd":
        return pgd_attack(model, x, y, epsilon, F.cross_entropy, steps=pgd_steps)

    raise ValueError(f"unknown attack: {method}")


def robustness_score(model, x, pgd_steps=20):
    """
    Smallest epsilon (from SCORE_EPSILONS) at which PGD changes the prediction

    Returns None if the prediction survives all epsilons up to 0.30
    """
    original = predict_proba(model, x).argmax().item()

    for eps in SCORE_EPSILONS:
        torch.manual_seed(0)  # PGD has a random start; fix it so the score is reproducible
        x_adv = attack(model, x, "pgd", eps, pgd_steps)
        if predict_proba(model, x_adv).argmax().item() != original:
            return eps

    return None
