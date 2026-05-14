from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data import make_gaussian_mixture, make_loader, subset_dataset
from .model import SimpleMLP
from .train import evaluate, train_model
from .unlearning import approximate_unlearning, full_retrain


def _annotate_bars(ax, bars) -> None:
    """Add percentage labels to each bar in a chart."""

    for bar in bars:
        height = bar.get_height()
        label_offset = 0.015 if height >= 0 else -0.035
        vertical_align = "bottom" if height >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + label_offset,
            f"{height:.1%}",
            ha="center",
            va=vertical_align,
            fontsize=8,
        )


def _save_accuracy_comparison_plot(summary: dict, output_dir: Path) -> None:
    """Create the main demo visualization from the experiment summary."""

    # Left panel: direct accuracy comparison for baseline, retrain, and unlearning.
    methods = [
        ("baseline", "Baseline", "#1f77b4"),
        ("retrain", "Full retraining", "#ff7f0e"),
        ("approximate_unlearning", "Approx. unlearning", "#2ca02c"),
    ]
    splits = [("test", "Test"), ("retain", "Retained"), ("remove", "Removed")]
    x_positions = list(range(len(splits)))
    width = 0.24

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)

    ax = axes[0]
    for index, (method_key, label, color) in enumerate(methods):
        values = [summary[method_key]["metrics"][split_key]["accuracy"] for split_key, _ in splits]
        offsets = [position + (index - 1) * width for position in x_positions]
        bars = ax.bar(offsets, values, width=width, label=label, color=color)
        _annotate_bars(ax, bars)

    ax.set_title("Accuracy before and after unlearning")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([label for _, label in splits])
    ax.set_ylim(0.0, 1.08)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)

    # Right panel: accuracy deltas make the forgetting effect easier to see.
    ax = axes[1]
    comparison_methods = [
        ("retrain", "Full retraining - Baseline", "#ff7f0e"),
        ("approximate_unlearning", "Approx. unlearning - Baseline", "#2ca02c"),
    ]
    comparison_splits = [("test", "Test"), ("remove", "Removed")]
    x_positions = list(range(len(comparison_splits)))
    width = 0.32

    baseline_metrics = summary["baseline"]["metrics"]
    for index, (method_key, label, color) in enumerate(comparison_methods):
        values = [
            summary[method_key]["metrics"][split_key]["accuracy"] - baseline_metrics[split_key]["accuracy"]
            for split_key, _ in comparison_splits
        ]
        offsets = [position + (index - 0.5) * width for position in x_positions]
        bars = ax.bar(offsets, values, width=width, label=label, color=color)
        _annotate_bars(ax, bars)

    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("Accuracy change relative to baseline")
    ax.set_ylabel("Accuracy delta")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([label for _, label in comparison_splits])
    ax.set_ylim(-1.08, 1.08)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)

    fig.savefig(output_dir / "accuracy_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_experiment(output_dir: Path) -> dict:
    """Run the full baseline, retraining, and approximate unlearning demo."""

    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    # Seed model initialization so the demo results are reproducible.
    torch.manual_seed(7)

    # Harder unlearning scenario: remove every training example from class 1.
    splits = make_gaussian_mixture(removal_strategy="forget_class", remove_class=1)
    train_dataset = splits.train
    test_dataset = splits.test

    # Build the retained/removed training subsets used for evaluation and unlearning.
    retained_dataset = subset_dataset(train_dataset, splits.retain_indices)
    removed_dataset = subset_dataset(train_dataset, splits.remove_indices)

    # Separate loaders let each workflow train/evaluate on the right data split.
    train_loader = make_loader(train_dataset, batch_size=64, shuffle=True)
    retained_loader = make_loader(retained_dataset, batch_size=64, shuffle=True)
    test_loader = make_loader(test_dataset, batch_size=128, shuffle=False)
    removed_loader = make_loader(removed_dataset, batch_size=64, shuffle=False)

    model_factory = lambda: SimpleMLP()

    # 1. Baseline: train on all data before any unlearning request.
    baseline_model = model_factory()
    baseline_result = train_model(baseline_model, train_loader, test_loader, device=device, epochs=25, lr=0.01)
    baseline_model.load_state_dict(baseline_result.model_state)
    baseline_metrics = {
        "test": evaluate(baseline_model, test_loader, device),
        "retain": evaluate(baseline_model, retained_loader, device),
        "remove": evaluate(baseline_model, removed_loader, device),
    }

    # 2. Full retraining: ground-truth baseline after deleting removed examples.
    retrain_result = full_retrain(model_factory, retained_loader, test_loader, device=device, epochs=25, lr=0.01)
    retrain_model = model_factory()
    retrain_model.load_state_dict(retrain_result.model_state)
    retrain_metrics = {
        "test": evaluate(retrain_model, test_loader, device),
        "retain": evaluate(retrain_model, retained_loader, device),
        "remove": evaluate(retrain_model, removed_loader, device),
    }

    # 3. Approximate unlearning: start from baseline and perform faster retain/scrub updates.
    approx_result = approximate_unlearning(
        model_factory,
        baseline_result.model_state,
        retained_loader,
        test_loader,
        device=device,
        removed_train_loader=removed_loader,
        fine_tune_epochs=6,
        lr=0.002,
        forget_weight=0.2,
    )
    approx_model = model_factory()
    approx_model.load_state_dict(approx_result.model_state)
    approx_metrics = {
        "test": evaluate(approx_model, test_loader, device),
        "retain": evaluate(approx_model, retained_loader, device),
        "remove": evaluate(approx_model, removed_loader, device),
    }

    # Collect all metrics needed for the terminal demo, JSON artifact, and plot.
    summary = {
        "dataset": {
            "removal_strategy": splits.removal_strategy,
            "train_size": len(train_dataset),
            "test_size": len(test_dataset),
            "removed_points": len(splits.remove_indices),
            "retained_points": len(splits.retain_indices),
        },
        "baseline": {
            "train_seconds": baseline_result.seconds,
            "metrics": baseline_metrics,
        },
        "retrain": {
            "seconds": retrain_result.seconds,
            "metrics": retrain_metrics,
        },
        "approximate_unlearning": {
            "seconds": approx_result.seconds,
            "metrics": approx_metrics,
        },
        "speedup_retrain_over_approx": retrain_result.seconds / max(approx_result.seconds, 1e-9),
        "forgetting_gap": retrain_metrics["remove"]["accuracy"] - approx_metrics["remove"]["accuracy"],
        "retrain_removed_accuracy_drop": baseline_metrics["remove"]["accuracy"] - retrain_metrics["remove"]["accuracy"],
        "approx_removed_accuracy_drop": baseline_metrics["remove"]["accuracy"] - approx_metrics["remove"]["accuracy"],
    }

    # Persist results so the presentation can show both raw numbers and a chart.
    (output_dir / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _save_accuracy_comparison_plot(summary, output_dir)
    return summary
