"""
Make the README figure and results table from the CLI results in results/

Usage (from the repo root, after running the configs):
    python scripts/make_figures.py

Writes:
    figures/fgsm_vs_pgd.png   accuracy vs. epsilon, FGSM vs. PGD, standard vs. adversarially trained
    results/summary.md        markdown table of all runs, to paste into the README
"""

import glob
import json
import os

import matplotlib.pyplot as plt

STANDARD = "#2a78d6"  # blue
ADVERSARIAL = "#eb6834"  # orange

PANELS = [
    ("MNIST binary, logistic regression", "mnist_binary_logreg", "mnist_binary_logreg_adv"),
    ("MNIST multiclass, CNN", "mnist_multiclass_cnn", "mnist_multiclass_cnn_adv"),
    ("STL-10 binary, CNN + BatchNorm", "stl10_binary_cnn_bn", "stl10_binary_cnn_bn_adv"),
]


def load(name):
    with open(os.path.join("results", name, "results.json")) as f:
        return json.load(f)


def plot_figure(path="figures/fgsm_vs_pgd.png"):
    fig, axes = plt.subplots(1, len(PANELS), figsize=(15, 4.2))

    for ax, (title, std_name, adv_name) in zip(axes, PANELS):
        for name, color, label in [(std_name, STANDARD, "standard"), (adv_name, ADVERSARIAL, "adversarially trained")]:
            r = load(name)
            ax.plot(r["epsilons"], r["fgsm"], color=color, linestyle="--", linewidth=2, marker="o", markersize=5,
                    label=f"{label}, FGSM")
            ax.plot(r["epsilons"], r["pgd"], color=color, linestyle="-", linewidth=2, marker="s", markersize=5,
                    label=f"{label}, PGD")

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("epsilon (max change per pixel)")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)

    axes[0].set_ylabel("test accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print("saved", path)


def summary_table(path="results/summary.md"):
    rows = []
    for file in sorted(glob.glob("results/*/results.json")):
        with open(file) as f:
            r = json.load(f)
        eps = r["epsilons"]
        # report the attacks at the training epsilon of the adversarial models: 0.15 for MNIST, 0.015 for STL-10
        target = 0.15 if 0.15 in eps else 0.015 if 0.015 in eps else eps[len(eps) // 2]
        mid = eps.index(target)
        rows.append(
            f"| `{r['config']['name']}` | {r['clean']:.1%} | {eps[mid]} | "
            f"{r['fgsm'][mid]:.1%} | {r['pgd'][mid]:.1%} |"
        )

    table = "\n".join(
        ["| Experiment | Clean | epsilon | FGSM | PGD |", "|---|---|---|---|---|"] + rows
    )
    with open(path, "w") as f:
        f.write(table + "\n")
    print(table)
    print("\nsaved", path)


if __name__ == "__main__":
    plot_figure()
    summary_table()
