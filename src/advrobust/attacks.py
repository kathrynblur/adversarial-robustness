"""
Shared FGSM and random noise implementaions
"""

import torch


def fgsm_attack(model, x, y, epsilon, loss_fn):
    """
    Appliies the Fast Gradient Sign Method (FGSM) attack

    Works for both binary and multiclass models by passing the appropriate loss function
    """
    
    # NOTE: in normal training, backward() computes gradients w.r.t. the
    # model's WEIGHTS. Here we instead set requires_grad_(True) on the
    # INPUT, so backward() gives us the gradient w.r.t. the input pixels.
    
    was_training = model.training
    model.eval()

    x_adv = x.clone().detach().requires_grad_(True) 

    output = model(x_adv)
    loss = loss_fn(output, y)

    model.zero_grad() # clear any weight gradients
    loss.backward() # compute gradients
    grad = x_adv.grad # gradient of the loss w.r.t. each input pixel

    x_adv = x_adv + epsilon * grad.sign()
    x_adv = torch.clamp(x_adv, 0, 1).detach()

    if was_training: # if it was in train mode, go back
        model.train()

    return x_adv

def gaussian_noise_attack(x, epsilon):
    """
    Adds Gaussian noise as a random baseline

    The noise has standard deviation epsilon/3 and is clipped so that each pixel perturbation
    stays within [-epsilon, epsilon]
    """
    
    noise = torch.randn_like(x) * (epsilon / 3)
    noise = torch.clamp(noise, -epsilon, epsilon)

    x_adv = x + noise
    x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv.detach()

def random_sign_attack(x, epsilon):
    """
    Add random perturbations of either -epsilon or + epsilon to each pixel
    """
    noise = epsilon * torch.sign(torch.randn_like(x))

    x_adv = x + noise
    x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv.detach()

def pgd_attack(model, x, y, epsilon, loss_fn, steps=10, step_size=None, random_start=True):
    """
    Applies the Projected Gradient Descent (PGD) attack

    PGD repeats small FGSM-like steps. After each step, the image is projected
    back into the epsilon-ball around the original image and into [0, 1]

    step_size defaults to 2.5 * epsilon / steps (Madry et al., 2018)
    """
    if step_size is None:
        step_size = 2.5 * epsilon / steps

    was_training = model.training
    model.eval()

    x_orig = x.clone().detach()

    # start at a random point inside the epsilon-ball
    if random_start:
        x_adv = x_orig + torch.empty_like(x_orig).uniform_(-epsilon, epsilon)
        x_adv = torch.clamp(x_adv, 0, 1)
    else:
        x_adv = x_orig.clone()

    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)

        loss = loss_fn(model(x_adv), y)

        # gradient w.r.t. the input only (does not touch the weight gradients)
        grad = torch.autograd.grad(loss, x_adv)[0]

        x_adv = x_adv.detach() + step_size * grad.sign()

        # project back into the epsilon-ball and the valid pixel range
        x_adv = torch.max(torch.min(x_adv, x_orig + epsilon), x_orig - epsilon)
        x_adv = torch.clamp(x_adv, 0, 1)

    if was_training:
        model.train()

    return x_adv.detach()
