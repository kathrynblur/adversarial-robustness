"""
Training and evaluation functions

Binary models output probabilities, so they use binary cross-entropy loss
and a threshold of 0.5 for predictions

Multiclass models output logits, so they use cross-entropy loss and argmax
for predictions
"""

import torch
import torch.nn.functional as F

from .utils import device

from .models import (
    BinaryLogReg, MulticlassLogReg, BinaryCNNMNIST,
    MulticlassCNNMNIST, BinaryCNNSTL10, MulticlassCNNSTL10,
    BinaryResNetSTL10, MulticlassResNetSTL10
)

from .attacks import fgsm_attack

def train_binary_logreg(train_loader, reg_type=None, reg_strength=1e-4, epochs=15, lr=0.05, in_features=784):
    """
    Train a binary logistic regression model with optional L1 or L2 regularization

    The regularization penalty is added directly to the training loss
    """
    model = BinaryLogReg(in_features=in_features).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        total_correct = 0
        total = 0

        for x, y in train_loader: # for one batch
            x = x.to(device)
            y = y.to(device).float()

            # forward pass
            probs = model(x)
            loss = F.binary_cross_entropy(probs, y)

            # add regularization penalty
            w = model.linear.weight

            if reg_type == "l2":
                loss = loss + reg_strength * 0.5 * (w ** 2).sum()
            elif reg_type == "l1":
                loss = loss + reg_strength * w.abs().sum()

            # update model weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_correct += ((probs > 0.5).long() == y.long()).sum().item()
            total += x.size(0)

        if epoch % 5 == 0 or epoch == 1: # each epoch = 1 or divisible by 5
            print("epoch", epoch, "| loss:", round(total_loss / total, 4), "| train acc:", round(total_correct / total, 4))

    return model

@torch.no_grad() # don't track gradients
def eval_binary(model, loader):
    """
    Evaluate a binary model and return its accuracy
    """
    model.eval()

    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        preds = (model(x) > 0.5).long()

        correct += (preds == y).sum().item()
        total += x.size(0)

    return correct / total

def train_multiclass(train_loader, reg_type=None, reg_strength=1e-4, epochs=15, lr=0.05, in_features=784):
    """
    Train a multiclass logistic regression model
    
    Cross-entropy loss is used here for logits
    """
    model = MulticlassLogReg(in_features=in_features).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        total_correct = 0
        total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            # forward pass
            logits = model(x)
            loss = F.cross_entropy(logits, y)

            # add regularization penalty
            W = model.linear.weight

            if reg_type == "l2":
                loss = loss + reg_strength * 0.5 * (W ** 2).sum()
            elif reg_type == "l1":
                loss = loss + reg_strength * W.abs().sum()

            # update model weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)

        if epoch % 5 == 0 or epoch == 1:
            print("epoch", epoch, "| loss:", round(total_loss / total, 4), "| train acc:", round(total_correct / total, 4))

    return model

@torch.no_grad()
def eval_multiclass(model, loader):
    model.eval()

    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        preds = model(x).argmax(1)

        correct += (preds == y).sum().item()
        total += x.size(0)

    return correct / total

def train_binary_adversarial(train_loader, epsilon=0.15, adv_weight=0.5, epochs=15, lr=0.05, in_features=784):
    """
    Train binary logistic regression using FGSM adversarial training
    """
    model = BinaryLogReg(in_features=in_features).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        total_correct = 0
        total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device).float()

            # create adversarial examples using the current model
            x_adv = fgsm_attack(model, x, y, epsilon, bce)

            # predictions on clean and adversarial examples
            probs_clean = model(x)
            probs_adv = model(x_adv)

            clean_loss = bce(probs_clean, y)
            adv_loss = bce(probs_adv, y)

            loss = ((1 - adv_weight) * clean_loss + adv_weight * adv_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_correct += ((probs_clean > 0.5).long() == y.long()).sum().item()
            total += x.size(0)

        if epoch % 5 == 0 or epoch == 1:
            print("epoch", epoch, "| loss:", round(total_loss / total, 4), "| train acc (clean):", round(total_correct / total, 4))

    return model




def train_multiclass_adversarial(train_loader, epsilon=0.15, adv_weight=0.5, epochs=15, lr=0.05, in_features=784):
    model = MulticlassLogReg(in_features=in_features).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        total_correct = 0
        total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            # create adversarial examples using the current model
            x_adv = fgsm_attack(model, x, y, epsilon, F.cross_entropy)

            # predictions on clean and adversarial examples
            logits_clean = model(x)
            logits_adv = model(x_adv)

            clean_loss = F.cross_entropy(logits_clean, y)
            adv_loss = F.cross_entropy(logits_adv, y)

            loss = ((1 - adv_weight) * clean_loss + adv_weight * adv_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_correct += (logits_clean.argmax(1) == y).sum().item()
            total += x.size(0)

        if epoch % 5 == 0 or epoch == 1:
            print("epoch", epoch, "| loss:", round(total_loss / total, 4), "| train acc (clean):", round(total_correct / total, 4))

    return model

# CNN training functions

def train_binary_cnn(train_loader, dataset="mnist", epochs=15, lr=0.01):
    """
    Train a binary CNN for either MNIST or STL-10
    """
    if dataset == "mnist":
        model = BinaryCNNMNIST().to(device)
    elif dataset == "stl10":
        model = BinaryCNNSTL10().to(device)


    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        total_correct = 0
        total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device).float()

            probs = model(x)
            loss = F.binary_cross_entropy(probs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_correct += ((probs > 0.5).long() == y.long()).sum().item()
            total += x.size(0)

        if epoch % 5 == 0 or epoch == 1:
            print("epoch", epoch, "| loss:", round(total_loss / total, 4), "| train acc:", round(total_correct / total, 4))

    return model

def train_multiclass_cnn(train_loader, dataset="mnist", epochs=15, lr=0.01):
    """
    Train a multiclass CNN for either MNIST or STL-10
    """
    if dataset == "mnist":
        model = MulticlassCNNMNIST().to(device)
    elif dataset == "stl10":
        model = MulticlassCNNSTL10().to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        total_correct = 0
        total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)

        if epoch % 5 == 0 or epoch == 1:
            print("epoch", epoch, "| loss:", round(total_loss / total, 4), "| train acc:", round(total_correct / total, 4))

    return model

def train_binary_cnn_adversarial(train_loader, dataset="mnist", epsilon=0.15, adv_weight=0.5, epochs=15, lr=0.01):
    if dataset == "mnist":
        model = BinaryCNNMNIST().to(device)
    elif dataset == "stl10":
        model = BinaryCNNSTL10().to(device)


    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        total_correct = 0
        total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device).float()

            # create adversarial examples using the current model
            x_adv = fgsm_attack(
                model, x, y, epsilon, bce
            )

            probs_clean = model(x)
            probs_adv = model(x_adv)

            clean_loss = bce(probs_clean, y)
            adv_loss = bce(probs_adv, y)

            loss = ((1 - adv_weight) * clean_loss + adv_weight * adv_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_correct += ((probs_clean > 0.5).long() == y.long()).sum().item()
            total += x.size(0)

        if epoch % 5 == 0 or epoch == 1:
            print("epoch", epoch, "| loss:", round(total_loss / total, 4), "| train acc (clean):", round(total_correct / total, 4))

    return model

def train_multiclass_cnn_adversarial(train_loader, dataset="mnist", epsilon=0.15, adv_weight=0.5, epochs=15, lr=0.01):

    if dataset == "mnist":
        model = MulticlassCNNMNIST().to(device)
    elif dataset == "stl10":
        model = MulticlassCNNSTL10().to(device)


    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0
        total_correct = 0
        total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            # create adversarial examples using the current model
            x_adv = fgsm_attack(model, x, y, epsilon, F.cross_entropy)

            logits_clean = model(x)
            logits_adv = model(x_adv)

            clean_loss = F.cross_entropy(logits_clean, y)
            adv_loss = F.cross_entropy(logits_adv, y)

            loss = ((1 - adv_weight) * clean_loss + adv_weight * adv_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_correct += (logits_clean.argmax(1) == y).sum().item()
            total += x.size(0)

        if epoch % 5 == 0 or epoch == 1:
            print("epoch", epoch, "| loss:", round(total_loss / total, 4), "| train acc (clean):", round(total_correct / total, 4))

    return model

# Resnet training functions

def train_binary_resnet(train_loader, epochs=15, lr=1e-3):
    """
    Binary ResNet-18 model for STL-10
    """
    model = BinaryResNetSTL10().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        total_correct = 0
        total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device).float()

            probs = model(x)
            loss = F.binary_cross_entropy(probs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_correct += ((probs > 0.5).long() == y.long()).sum().item()
            total += x.size(0)

        if epoch % 5 == 0 or epoch == 1:
            print("epoch", epoch, "| loss:", round(total_loss / total, 4), "| train acc:", round(total_correct / total, 4))

    return model

def train_multiclass_resnet(train_loader, epochs=15, lr=1e-3):

    model = MulticlassResNetSTL10().to(device)

    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        total_correct = 0
        total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)

        if epoch % 5 == 0 or epoch == 1:
            print("epoch", epoch,"| loss:", round(total_loss / total, 4), "| train acc:", round(total_correct / total, 4))

    return model

def train_binary_resnet_adversarial(train_loader, epsilon=0.015, adv_weight=0.5, epochs=15, lr=1e-3):
    
    model = BinaryResNetSTL10().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        total_correct = 0
        total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device).float()

            # create adversarial examples using the current model
            x_adv = fgsm_attack(model, x, y, epsilon, bce)

            probs_clean = model(x)
            probs_adv = model(x_adv)

            clean_loss = bce(probs_clean, y)
            adv_loss = bce(probs_adv, y)

            loss = ((1 - adv_weight) * clean_loss + adv_weight * adv_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_correct += ((probs_clean > 0.5).long() == y.long()).sum().item()
            total += x.size(0)

        if epoch % 5 == 0 or epoch == 1:
            print("epoch", epoch, "| loss:", round(total_loss / total, 4), "| train acc (clean):", round(total_correct / total, 4))

    return model

def train_multiclass_resnet_adversarial(train_loader, epsilon=0.015, adv_weight=0.5, epochs=15, lr=1e-3):

    model = MulticlassResNetSTL10().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        total_correct = 0
        total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            # create adversarial examples using the current model
            x_adv = fgsm_attack(model, x, y, epsilon, F.cross_entropy)

            logits_clean = model(x)
            logits_adv = model(x_adv)

            clean_loss = F.cross_entropy(logits_clean, y)
            adv_loss = F.cross_entropy(logits_adv, y)

            loss = ((1 - adv_weight) * clean_loss + adv_weight * adv_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_correct += (logits_clean.argmax(1) == y).sum().item()
            total += x.size(0)

        if epoch % 5 == 0 or epoch == 1:
            print("epoch", epoch, "| loss:", round(total_loss / total, 4), "| train acc (clean):", round(total_correct / total, 4))

    return model


# Shared evaluation helpers

def bce(probs, y):
    """
    Binary cross-entropy loss
    """
    return F.binary_cross_entropy(probs, y.float())

def evaluate_fgsm_binary(model, loader, epsilon):

    model.eval()

    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        x_adv = fgsm_attack(model, x, y, epsilon, bce)

        with torch.no_grad():
            probs = model(x_adv)
            preds = (probs > 0.5).long()

        correct += (preds == y).sum().item()
        total += x.size(0)

    return correct / total

def evaluate_fgsm_multiclass(model, loader, epsilon):

    model.eval()

    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        x_adv = fgsm_attack(model, x, y, epsilon, F.cross_entropy)

        with torch.no_grad():
            logits = model(x_adv)
            preds = logits.argmax(1)

        correct += (preds == y).sum().item()
        total += x.size(0)

    return correct / total

def evaluate_noise_binary(model, loader, epsilon, noise_fn):

    model.eval()

    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        x_noise = noise_fn(x, epsilon)

        with torch.no_grad():
            probs = model(x_noise)
            preds = (probs > 0.5).long()

        correct += (preds == y).sum().item()
        total += x.size(0)

    return correct / total

def evaluate_noise_multiclass(model, loader, epsilon, noise_fn):
    model.eval()

    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        x_noise = noise_fn(x, epsilon)

        with torch.no_grad():
            logits = model(x_noise)
            preds = logits.argmax(1)

        correct += (preds == y).sum().item()
        total += x.size(0)

    return correct / total
