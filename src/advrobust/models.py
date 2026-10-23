"""
Model definitions used throughout the project

Pretrained ResNet-18 models are used for STL-10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryLogReg(nn.Module):
    """
    Binary logistic regression model with one linear layer followed by a sigmoid.
    """
    def __init__(self, in_features=784):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.linear(x))
        return x.squeeze(-1)  # change shape from (N, 1) to (N,)

class MulticlassLogReg(nn.Module):
    """
    Multiclass logistic regression with one linear layer
    """
    def __init__(self, in_features=784):
        super().__init__()
        self.linear = nn.Linear(in_features, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear(x)  # returns logits

class BinaryCNNMNIST(nn.Module):
    """
    CNN for binary classification on MNIST

    The network has two convolutional layers, each followed by max pooling,
    then two fully connected layers.

    Input shape: (N, 1, 28, 28)
    After pooling: 28x28 -> 14x14 -> 7x7
    Output shape: (N,) probabilities.
    """
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)

        self.fc1 = nn.Linear(7 * 7 * 64, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return torch.sigmoid(x).squeeze(-1)

class MulticlassCNNMNIST(nn.Module):
    """
    CNN for multiclass classification on MNIST

    The network has two convolutional layers, each followed by max pooling,
    then two fully connected layers. Same architecture as
    BinaryCNNMNIST, except the final layer outputs 10 logits.

    Input shape: (N, 1, 28, 28)
    After pooling: 28x28 -> 14x14 -> 7x7
    Output shape: (N, 10) logits.
    """
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)

        self.fc1 = nn.Linear(7 * 7 * 64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class BinaryCNNSTL10(nn.Module):
    """
    CNN for binary classification on STL-10

    The network has three convolutional layers, each followed by max pooling,
    then two fully connected layers.

    Input shape: (N, 3, 96, 96)
    After pooling: 96x96 -> 48x48 -> 24x24 -> 12x12
    Output shape: (N,) probabilities.
    """
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=2)

        self.fc1 = nn.Linear(12 * 12 * 128, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2)) # 96 -> 48
        x = F.relu(F.max_pool2d(self.conv2(x), 2)) # 48 -> 24
        x = F.relu(F.max_pool2d(self.conv3(x), 2)) # 24 -> 12

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return torch.sigmoid(x).squeeze(-1)

class MulticlassCNNSTL10(nn.Module):
    """
    CNN for multiclass classification on STL-10.

    The network has three convolutional layers, each followed by max pooling,
    then two fully connected layers. Same architecture as
    BinaryCNNSTL10, except the final layer outputs 10 logits.

    Input shape: (N, 3, 96, 96)
    After pooling: 96x96 -> 48x48 -> 24x24 -> 12x12
    Output shape: (N, 10) logits.
    """
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=2)

        self.fc1 = nn.Linear(12 * 12 * 128, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# ResNet Implementation

class Normalize(nn.Module):
    """
    Normalize each image channel using the given mean and standard deviation.

    This is done so that the input images can stay
    in the [0, 1] range when applying FGSM
    """
    def __init__(self, mean, std):
        super().__init__()

        self.register_buffer(
            "mean", torch.tensor(mean).view(1, -1, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(std).view(1, -1, 1, 1)
        )

    def forward(self, x):
        return (x - self.mean) / self.std


# ImageNet normalization values used by pretrained ResNet-18, source:
# https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html#torchvision.models.ResNet18_Weights 
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def make_resnet_backbone():
    """
    Pretrained ResNet-18 and freeze its pretrained weights
    """
    import torchvision

    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    backbone = torchvision.models.resnet18(weights=weights)

    # freeze the pretrained weights
    for parameter in backbone.parameters():
        parameter.requires_grad = False

    return backbone

class BinaryResNetSTL10(nn.Module):
    """
    ResNet-18 for binary classification on STL-10

    Final layer is replaced with a new binary classification layer
    """
    def __init__(self):
        super().__init__()

        self.normalize = Normalize(IMAGENET_MEAN, IMAGENET_STD)
        self.backbone = make_resnet_backbone()

        self.backbone.fc = nn.Linear(
            self.backbone.fc.in_features, 1
        )

    def forward(self, x):
        x = self.normalize(x)
        x = self.backbone(x)
        x = torch.sigmoid(x)

        return x.squeeze(-1)

class MulticlassResNetSTL10(nn.Module):
    """
    ResNet-18 for multiclass classification on STL-10
    """
    def __init__(self):
        super().__init__()

        self.normalize = Normalize(IMAGENET_MEAN, IMAGENET_STD)
        self.backbone = make_resnet_backbone()

        self.backbone.fc = nn.Linear(
            self.backbone.fc.in_features, 10
        )

    def forward(self, x):
        x = self.normalize(x)
        x = self.backbone(x)

        return x