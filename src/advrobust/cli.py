"""
Run experiments from YAML config files

Usage:
    python -m advrobust.cli configs/mnist_binary_cnn.yaml
    python -m advrobust.cli configs/*.yaml
    python -m advrobust.cli configs/*.yaml --tracking wandb
"""

import argparse
import json
import os

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from . import models
from .augment import random_flip
from .evaluate import evaluate_clean, evaluate_fgsm, evaluate_pgd
from .train import train
from .utils import SEED, device, set_seed

AUGMENTATIONS = {None: None, "random_flip": random_flip}


def get_datasets(dataset, task):
    """
    Return (train_dataset, test_dataset) with image-shaped inputs

    Logistic regression models flatten the input in forward(), so they use the same datasets
    """
    if dataset == "mnist":
        from . import data  # imported here, because importing data downloads MNIST

        if task == "binary":
            return data.train_bin_img, data.test_bin_img
        return data.train_ds, data.test_ds

    if dataset == "stl10":
        from .data import load_stl10

        stl = load_stl10()
        if task == "binary":
            return stl["stl_train_bin_img"], stl["stl_test_bin_img"]
        return stl["stl_train"], stl["stl_test"]

    raise ValueError(f"unknown dataset: {dataset}")


def start_tracking(cfg):
    """
    Start a Weights & Biases run if the config asks for it, otherwise return None
    """
    if cfg.get("tracking", "none") != "wandb":
        return None

    import wandb  # imported here, so wandb is only needed when tracking is on

    return wandb.init(
        project=cfg.get("wandb_project", "adversarial-robustness"),
        name=cfg["name"],
        config=cfg,
    )


def log_results(wb_run, cfg, results):
    """
    Log the final accuracies and a robustness curve (accuracy vs. epsilon) to wandb
    """
    import wandb

    epsilons = cfg["eval"]["epsilons"]
    attacks = cfg["eval"].get("attacks", [])

    wb_run.summary["clean_acc"] = results["clean"]
    for attack in attacks:
        for eps, acc in zip(epsilons, results[attack]):
            wb_run.summary[f"{attack}_acc_eps_{eps}"] = acc

    if attacks:
        wb_run.log({
            "robustness_curve": wandb.plot.line_series(
                xs=epsilons,
                ys=[results[a] for a in attacks],
                keys=attacks,
                title=f"{cfg['name']}: accuracy vs. epsilon",
                xname="epsilon",
            )
        })


def run(cfg):
    """
    Train and evaluate one model as described by a config dict, and save the results
    """
    set_seed(cfg.get("seed", SEED))
    wb_run = start_tracking(cfg)
    binary = cfg["task"] == "binary"
    t = cfg["train"]
    e = cfg["eval"]

    train_ds, test_ds = get_datasets(cfg["dataset"], cfg["task"])

    # optional random subset of the test set (PGD is slow on large test sets)
    if e.get("max_samples"):
        g = torch.Generator().manual_seed(cfg.get("seed", SEED))
        idx = torch.randperm(len(test_ds), generator=g)[: e["max_samples"]]
        test_ds = Subset(test_ds, idx.tolist())

    train_loader = DataLoader(train_ds, batch_size=t.get("batch_size", 128), shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    model_cls = getattr(models, cfg["model"]["name"])
    model = model_cls(**cfg["model"].get("args", {}))

    print(f"=== {cfg['name']} ({device}) ===")
    model = train(
        model,
        train_loader,
        binary=binary,
        epochs=t.get("epochs", 15),
        lr=t.get("lr", 0.01),
        optimizer=t.get("optimizer", "sgd"),
        reg_type=t.get("reg_type"),
        reg_strength=t.get("reg_strength", 1e-4),
        adv_epsilon=t.get("adv_epsilon"),
        adv_weight=t.get("adv_weight", 0.5),
        augment=AUGMENTATIONS[t.get("augment")],
        log_fn=wb_run.log if wb_run is not None else None,
    )

    results = {"clean": evaluate_clean(model, test_loader, binary)}
    print("clean accuracy:", round(results["clean"], 4))

    for attack in e.get("attacks", []):
        accs = []
        for eps in e["epsilons"]:
            if eps == 0:
                acc = results["clean"]
            elif attack == "fgsm":
                acc = evaluate_fgsm(model, test_loader, eps, binary)
            elif attack == "pgd":
                acc = evaluate_pgd(model, test_loader, eps, binary, steps=e.get("pgd_steps", 10))
            else:
                raise ValueError(f"unknown attack: {attack}")
            accs.append(acc)
        results[attack] = accs
        print(attack, "accuracy:", [round(a, 3) for a in accs])

    out_dir = os.path.join(cfg.get("output_dir", "results"), cfg["name"])
    os.makedirs(out_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"config": cfg, "epsilons": e["epsilons"], **results}, f, indent=2)

    print("saved to", out_dir)

    if wb_run is not None:
        log_results(wb_run, cfg, results)
        wb_run.finish()

    return results


def main():
    parser = argparse.ArgumentParser(description="Run adversarial robustness experiments")
    parser.add_argument("configs", nargs="+", help="one or more YAML config files")
    parser.add_argument("--tracking", choices=["none", "wandb"], default=None,
                        help="override the tracking setting of the config files")
    args = parser.parse_args()

    for path in args.configs:
        with open(path) as f:
            cfg = yaml.safe_load(f)
        if args.tracking is not None:
            cfg["tracking"] = args.tracking
        run(cfg)


if __name__ == "__main__":
    main()
