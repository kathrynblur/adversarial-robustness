import glob
import json

import pytest
import yaml
from conftest import make_loader

from advrobust import cli, models

CONFIGS = sorted(glob.glob("configs/*.yaml"))


@pytest.mark.parametrize("path", CONFIGS)
def test_configs_are_valid(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)

    assert {"name", "dataset", "task", "model", "train", "eval"} <= set(cfg)
    assert cfg["dataset"] in {"mnist", "stl10"}
    assert cfg["task"] in {"binary", "multiclass"}
    assert hasattr(models, cfg["model"]["name"])
    assert cfg["train"].get("augment") in cli.AUGMENTATIONS
    assert set(cfg["eval"].get("attacks", [])) <= {"fgsm", "pgd"}


def test_run_trains_evaluates_and_saves(tmp_path, monkeypatch):
    def fake_datasets(dataset, task):
        return make_loader(32, (1, 28, 28), 2).dataset, make_loader(16, (1, 28, 28), 2).dataset

    monkeypatch.setattr(cli, "get_datasets", fake_datasets)

    cfg = {
        "name": "test_run",
        "dataset": "mnist",
        "task": "binary",
        "output_dir": str(tmp_path),
        "model": {"name": "BinaryLogReg"},
        "train": {"epochs": 1, "lr": 0.05},
        "eval": {"epsilons": [0.0, 0.1], "attacks": ["fgsm", "pgd"], "pgd_steps": 2},
    }
    results = cli.run(cfg)

    assert set(results) == {"clean", "fgsm", "pgd"}
    assert len(results["fgsm"]) == len(results["pgd"]) == 2
    assert results["fgsm"][0] == results["clean"]

    saved = json.loads((tmp_path / "test_run" / "results.json").read_text())
    assert saved["fgsm"] == results["fgsm"]
    assert (tmp_path / "test_run" / "model.pt").exists()
