"""
Training loop shared by all models

Binary models output probabilities, so they use binary cross-entropy loss
and a threshold of 0.5 for predictions

Multiclass models output logits, so they use cross-entropy loss and argmax
for predictions
"""

import torch
import torch.nn.functional as F

from .attacks import fgsm_attack
from .utils import device


def bce(probs, y):
    """
    Binary cross-entropy loss
    """
    return F.binary_cross_entropy(probs, y.float())


def reg_penalty(model, reg_type, reg_strength):
    """
    L1 or L2 penalty on the weights of a logistic regression model
    """
    w = model.linear.weight

    if reg_type == "l2":
        return reg_strength * 0.5 * (w ** 2).sum()
    if reg_type == "l1":
        return reg_strength * w.abs().sum()
    return 0.0


def train(
    model,
    train_loader,
    binary,
    epochs=15,
    lr=0.01,
    optimizer="sgd",
    reg_type=None,
    reg_strength=1e-4,
    adv_epsilon=None,
    adv_weight=0.5,
    augment=None,
    log_fn=None,
):
    """
    Train a model in place and return it

    reg_type: None, "l1" or "l2" (logistic regression only)
        augment: None, or a function that takes a batch x and returns an augmented batch
    adv_epsilon: None for standard training, a number for FGSM adversarial training
    log_fn: None, or a function that receives a dict of metrics after each epoch
            (for example wandb's run.log)
    """
    model = model.to(device)
    loss_fn = bce if binary else F.cross_entropy

    if optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(f"unknown optimizer: {optimizer}")

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        total_correct = 0
        total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            if augment is not None:
                x = augment(x)

            if adv_epsilon is not None:
                # create adversarial examples using the current model
                x_adv = fgsm_attack(model, x, y, adv_epsilon, loss_fn)

            out = model(x)
            loss = loss_fn(out, y)

            if adv_epsilon is not None:
                adv_loss = loss_fn(model(x_adv), y)
                loss = (1 - adv_weight) * loss + adv_weight * adv_loss

            if reg_type is not None:
                loss = loss + reg_penalty(model, reg_type, reg_strength)

            opt.zero_grad()
            loss.backward()
            opt.step()

            preds = (out > 0.5).long() if binary else out.argmax(1)

            total_loss += loss.item() * x.size(0)
            total_correct += (preds == y.long()).sum().item()
            total += x.size(0)

        if log_fn is not None:
            log_fn({"epoch": epoch, "train_loss": total_loss / total, "train_acc": total_correct / total})

        if epoch % 5 == 0 or epoch == 1:
                        print(
                "epoch", epoch,
                "| loss:", round(total_loss / total, 4),
                "| train acc:", round(total_correct / total, 4),
            )

    return model
