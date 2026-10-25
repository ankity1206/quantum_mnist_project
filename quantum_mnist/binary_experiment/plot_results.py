
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from config import CONFIG, prepare_directories


def load_training_history(csv_path):
    """Load training metrics from the CSV file."""
    history = []

    with open(csv_path, "r", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            history.append({
                "model": row["model"],
                "epoch": int(row["epoch"]),
                "train_loss": float(row["train_loss"]),
                "train_accuracy": float(row["train_accuracy"]),
                "test_loss": float(row["test_loss"]),
                "test_accuracy": float(row["test_accuracy"]),
            })

    if not history:
        raise ValueError(f"No training records found in {csv_path}")

    return history


def plot_metric(history, metric, title, ylabel, filename):
    """Plot a metric for both models and save the figure."""
    models = sorted(set(row["model"] for row in history))

    plt.figure(figsize=(8, 5))

    for model_name in models:
        rows = sorted(
            [row for row in history if row["model"] == model_name],
            key=lambda row: row["epoch"],
        )

        epochs = [row["epoch"] for row in rows]
        values = [row[metric] for row in rows]

        plt.plot(
            epochs,
            values,
            marker="o",
            label=model_name,
        )

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.xticks(range(1, max(row["epoch"] for row in history) + 1))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = CONFIG.output_dir / filename
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot: {output_path}")

    plt.show()
    plt.close()


def main():
    prepare_directories()

    csv_path = CONFIG.output_dir / "training_history.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Training history not found: {csv_path}\n"
            "Run train.py first."
        )

    history = load_training_history(csv_path)

    plot_metric(
        history=history,
        metric="train_loss",
        title="Training Loss Comparison",
        ylabel="Training Loss",
        filename="training_loss.png",
    )

    plot_metric(
        history=history,
        metric="test_accuracy",
        title="Test Accuracy Comparison",
        ylabel="Test Accuracy",
        filename="test_accuracy.png",
    )


if __name__ == "__main__":
    main()
