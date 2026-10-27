import pytest
import torch

from advrobust.models import (
    BinaryCNNMNIST,
    BinaryCNNSTL10,
    BinaryCNNSTL10BN,
    BinaryLogReg,
    MulticlassCNNMNIST,
    MulticlassCNNSTL10,
    MulticlassLogReg,
)

# ResNet models are not tested here, because they download pretrained weights

MNIST = (1, 28, 28)
STL10 = (3, 96, 96)


@pytest.mark.parametrize(
    "model, input_shape",
    [
        (BinaryLogReg(), MNIST),
        (BinaryLogReg(in_features=3 * 96 * 96), STL10),
        (BinaryCNNMNIST(), MNIST),
        (BinaryCNNSTL10(), STL10),
        (BinaryCNNSTL10BN(), STL10),
    ],
)
def test_binary_models_output_one_probability_per_image(model, input_shape):
    x = torch.rand(4, *input_shape)

    out = model(x)

    assert out.shape == (4,)
    assert (out >= 0).all() and (out <= 1).all()


@pytest.mark.parametrize(
    "model, input_shape",
    [
        (MulticlassLogReg(), MNIST),
        (MulticlassCNNMNIST(), MNIST),
        (MulticlassCNNSTL10(), STL10),
    ],
)
def test_multiclass_models_output_ten_logits_per_image(model, input_shape):
    x = torch.rand(4, *input_shape)

    out = model(x)

    assert out.shape == (4, 10)


def test_logreg_accepts_flat_and_image_inputs():
    model = BinaryLogReg()
    x = torch.rand(4, 1, 28, 28)

    assert torch.allclose(model(x), model(x.view(4, -1)))
