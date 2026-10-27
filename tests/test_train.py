import pytest
from conftest import make_loader

from advrobust.augment import random_flip
from advrobust.evaluate import evaluate_clean
from advrobust.models import BinaryCNNMNIST, BinaryLogReg, MulticlassCNNMNIST
from advrobust.train import train


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"reg_type": "l1"},
        {"reg_type": "l2"},
        {"adv_epsilon": 0.1},
        {"optimizer": "adam", "lr": 1e-3},
        {"augment": random_flip},
    ],
)
def test_train_runs_with_each_option(options):
    loader = make_loader(32, (1, 28, 28), 2)

    model = train(BinaryLogReg(), loader, binary=True, epochs=1, **options)

    assert isinstance(model, BinaryLogReg)


def test_train_runs_for_cnns():
    train(BinaryCNNMNIST(), make_loader(32, (1, 28, 28), 2), binary=True, epochs=1, adv_epsilon=0.1)
    train(MulticlassCNNMNIST(), make_loader(32, (1, 28, 28), 10), binary=False, epochs=1, adv_epsilon=0.1)


def test_unknown_optimizer_raises():
    with pytest.raises(ValueError):
        train(BinaryLogReg(), make_loader(16, (1, 28, 28), 2), binary=True, epochs=1, optimizer="rmsprop")


def test_log_fn_receives_metrics_after_each_epoch():
    logged = []
    train(BinaryLogReg(), make_loader(32, (1, 28, 28), 2), binary=True, epochs=3, log_fn=logged.append)

    assert [m["epoch"] for m in logged] == [1, 2, 3]
    assert all({"train_loss", "train_acc"} <= set(m) for m in logged)


def test_logistic_regression_learns_separable_data(separable_binary_loader):
    model = train(BinaryLogReg(), separable_binary_loader, binary=True, epochs=30, lr=0.05)

    assert evaluate_clean(model, separable_binary_loader, binary=True) > 0.9
