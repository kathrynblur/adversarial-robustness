"""
Data-loading utilities

Importing this:
    - fixes random seeds
    - selects the device
    - downloads/loads MNIST
    - creates the MNIST datasets and DataLoaders

STL-10 is NOT downloaded on import, call load_stl10() when you need it

"""

import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms

# Reproducibility
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download MNIST
transform = transforms.ToTensor()

train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# MNIST binary (3 vs 8) split

def make_binary_split(ds, digit_a=3, digit_b=8, flatten=True):
    """
    Filter the MNIST dataset down to two digits and relabel them to 0/1.

    flatten=True returns (N, 784) tensors for logistic regression.
    flatten=False returns (N, 1, 28, 28) tensors for a CNN.
    """
    # keep only digit_a and digit_b
    mask = (ds.targets == digit_a) | (ds.targets == digit_b) 

    # convert pixel values from [0, 255] to [0, 1]
    images = ds.data[mask].float().div(255.0)

    # digit_a becomes 0 and digit_b becomes 1
    labels = (ds.targets[mask] == digit_b).long()

    if flatten:
        images = images.reshape(images.size(0), -1)
    else:
        images = images.unsqueeze(1) # add channel dim 

    return TensorDataset(images, labels)


# Flattened
train_bin = make_binary_split(train_ds)
test_bin = make_binary_split(test_ds)

train_loader_bin = DataLoader(train_bin, batch_size=128, shuffle=True) 
test_loader_bin = DataLoader(test_bin, batch_size=128, shuffle=False)

# Image-shaped 
train_bin_img = make_binary_split(train_ds, flatten=False)
test_bin_img = make_binary_split(test_ds, flatten=False)

train_loader_bin_img = DataLoader(train_bin_img, batch_size=128, shuffle=True)
test_loader_bin_img = DataLoader(test_bin_img, batch_size=128, shuffle=False)

# MNIST multiclass loaders

# MulticlassLogReg flattens in its forward() and MulticlassCNN doesn't need flattening at all

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

# Validation split for regularization strength quick search

def get_binary_val_split(val_split=0.15, batch_size=128):
    """
    Split the binary MNIST training data into training and validation sets
    """
    n_val = int(len(train_bin) * val_split)
    n_train = len(train_bin) - n_val

    # split the training data into train and validation sets
    train_sub, val_sub = random_split(
        train_bin,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED) # for reproducibility
    )

    # create data loaders for both subsets
    train_loader_sub = DataLoader(
        train_sub, batch_size=batch_size, shuffle=True
    )
    val_loader_sub = DataLoader(
        val_sub, batch_size=batch_size, shuffle=False
    )

    return train_loader_sub, val_loader_sub

# Loading STL10


def make_stl_binary_split(ds, name_a="cat", name_b="dog", flatten=True):
    """
    Filter the STL-10 dataset down to two classes and relabel them to 0/1.

    """
    classes = ds.classes
    a = classes.index(name_a)
    b = classes.index(name_b)

    labels_all = torch.tensor(ds.labels)

    # keep only the two selected classes
    mask = (labels_all == a) | (labels_all == b)

    # convert pixel values from [0, 255] to [0, 1]
    images = torch.tensor(ds.data)[mask].float().div(255.0)

    if flatten:
        images = images.reshape(images.size(0), -1)

    # name_a becomes 0 and name_b becomes 1
    labels = (labels_all[mask] == b).long()

    return TensorDataset(images, labels)

def load_stl10():
    """
    Load the STL-10 train and test sets and create the datasets and
    DataLoaders for the binary and multiclass experiments.
    """
    
    stl_train = datasets.STL10(
        root="./data",
        split="train",
        download=True,
        transform=transform
    )

    stl_test = datasets.STL10(
        root="./data",
        split="test",
        download=True,
        transform=transform
    )

    STL_DIM = 3 * 96 * 96

    stl_name_a = "cat"
    stl_name_b = "dog"

    # binary cat vs dog datasets
    stl_train_bin = make_stl_binary_split(stl_train)
    stl_test_bin = make_stl_binary_split(stl_test)

    # image-shaped versions for CNNs
    stl_train_bin_img = make_stl_binary_split(
        stl_train, flatten=False
    )
    stl_test_bin_img = make_stl_binary_split(
        stl_test, flatten=False
    )

    # flattened binary loaders for logistic regression
    stl_train_loader_bin = DataLoader(
        stl_train_bin, batch_size=128, shuffle=True
    )
    stl_test_loader_bin = DataLoader(
        stl_test_bin, batch_size=128, shuffle=False
    )

    # image-shaped binary loaders for CNNs
    stl_train_loader_bin_img = DataLoader(
        stl_train_bin_img, batch_size=128, shuffle=True
    )
    stl_test_loader_bin_img = DataLoader(
        stl_test_bin_img, batch_size=128, shuffle=False
    )

    # full multiclass loaders
    stl_train_loader = DataLoader(
        stl_train, batch_size=128, shuffle=True
    )
    stl_test_loader = DataLoader(
        stl_test, batch_size=128, shuffle=False
    )

    return {
        "stl_train": stl_train,
        "stl_test": stl_test,
        "STL_DIM": STL_DIM,
        "stl_train_bin": stl_train_bin,
        "stl_test_bin": stl_test_bin,
        "stl_train_bin_img": stl_train_bin_img,
        "stl_test_bin_img": stl_test_bin_img,
        "stl_name_a": stl_name_a,
        "stl_name_b": stl_name_b,
        "stl_train_loader_bin": stl_train_loader_bin,
        "stl_test_loader_bin": stl_test_loader_bin,
        "stl_train_loader_bin_img": stl_train_loader_bin_img,
        "stl_test_loader_bin_img": stl_test_loader_bin_img,
        "stl_train_loader": stl_train_loader,
        "stl_test_loader": stl_test_loader
    }
    