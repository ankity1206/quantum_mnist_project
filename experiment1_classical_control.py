import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from config import CONFIG
from train import train_one_epoch


SEEDS = [42, 123, 2026]
EPOCHS = CONFIG.epochs
BATCH_SIZE = CONFIG.batch_size
LEARNING_RATE = CONFIG.learning_rate

BASE = CONFIG.project_dir / "outputs"
SOURCE = BASE / "fair_classical_vs_quantum"
OUT = BASE / "experiment1_classical_control"
OUT.mkdir(parents=True, exist_ok=True)


class NonlinearClassicalControl(nn.Module):
    """Parameter-matched classical nonlinear control: 8 -> 6 -> 16 -> 10."""

    def __init__(self, num_features=8, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 6),
            nn.Tanh(),
            nn.Linear(6, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, metrics, path):
    output = []
    for metric in metrics:
        values = np.array([float(r[metric]) for r in rows])
        output.append({
            "model": "nonlinear_classical_control",
            "metric": metric,
            "mean": float(values.mean()),
            "std_sample": float(values.std(ddof=1)),
            "seeds": len(values),
        })
    write_csv(path, output)


def main():
    manifest_path = SOURCE / "dataset_manifest.npz"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Dataset manifest not found: {manifest_path}"
        )

    # Reuse the exact saved features and labels from the previous benchmark.
    with np.load(manifest_path, allow_pickle=False) as data:
        X_train = torch.from_numpy(data["train_features"].copy()).float()
        y_train = torch.from_numpy(data["y_train"].copy()).long()
        X_test = torch.from_numpy(data["test_features"].copy()).float()
        y_test = torch.from_numpy(data["y_test"].copy()).long()

    assert X_train.shape == (2000, 8), X_train.shape
    assert X_test.shape == (1000, 8), X_test.shape
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    device = torch.device(
        "cuda" if CONFIG.device == "cuda" and torch.cuda.is_available()
        else "cpu"
    )

    # Count trainable parameters before starting.
    probe = NonlinearClassicalControl(CONFIG.num_qubits, CONFIG.num_classes)
    parameter_count = sum(p.numel() for p in probe.parameters() if p.requires_grad)
    assert parameter_count == 336, parameter_count

    print(f"Device: {device}")
    print(f"Train/test examples: {len(train_dataset)}/{len(test_dataset)}")
    print(f"Trainable parameters: {parameter_count}")
    print(f"Epochs={EPOCHS}, batch={BATCH_SIZE}, lr={LEARNING_RATE}")
    print(f"Output directory: {OUT}")

    results = []
    histories = []

    for seed in SEEDS:
        print("\n" + "=" * 65)
        print(f"NONLINEAR CLASSICAL CONTROL | seed={seed}")
        print("=" * 65)

        seed_everything(seed)
        generator = torch.Generator().manual_seed(seed)

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=generator,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )

        model = NonlinearClassicalControl(
            num_features=CONFIG.num_qubits,
            num_classes=CONFIG.num_classes,
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        start = time.time()

        for epoch in range(1, EPOCHS + 1):
            epoch_start = time.time()
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            epoch_seconds = time.time() - epoch_start

            row = {
                "model": "nonlinear_classical_control",
                "seed": seed,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "epoch_seconds": epoch_seconds,
            }
            histories.append(row)

            print(
                f"Epoch {epoch:02d}/{EPOCHS} | "
                f"loss={train_loss:.4f} | acc={train_acc:.2%} | "
                f"time={epoch_seconds:.2f}s"
            )

        training_seconds = time.time() - start

        # Evaluate the test set only after training has finished.
        model.eval()
        y_true, y_pred = [], []
        loss_sum, sample_count = 0.0, 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits = model(inputs)
                loss = criterion(logits, labels)

                loss_sum += loss.item() * len(labels)
                sample_count += len(labels)
                y_true.extend(labels.cpu().tolist())
                y_pred.extend(logits.argmax(dim=1).cpu().tolist())

        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        test_loss = loss_sum / sample_count

        tag = f"nonlinear_control_seed{seed}"
        torch.save(model.state_dict(), OUT / f"{tag}.pt")

        report = classification_report(
            y_true,
            y_pred,
            labels=list(range(CONFIG.num_classes)),
            output_dict=True,
            zero_division=0,
        )
        report_rows = [
            {"class": label, **values}
            for label, values in report.items()
            if isinstance(values, dict)
        ]
        write_csv(OUT / f"{tag}_report.csv", report_rows)

        np.savetxt(
            OUT / f"{tag}_confusion_matrix.csv",
            confusion_matrix(
                y_true, y_pred, labels=list(range(CONFIG.num_classes))
            ),
            delimiter=",",
            fmt="%d",
        )

        result = {
            "model": "nonlinear_classical_control",
            "seed": seed,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "test_loss": test_loss,
            "training_seconds": training_seconds,
            "device": str(device),
            "train_examples": len(train_dataset),
            "test_examples": len(test_dataset),
            "features": X_train.shape[1],
            "trainable_parameters": parameter_count,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "loss": "unweighted_cross_entropy",
        }
        results.append(result)

        # Save progress after every seed.
        write_csv(OUT / "runs.csv", results)
        write_csv(OUT / "training_history.csv", histories)

        print(
            f"FINAL seed={seed}: accuracy={accuracy:.2%}, "
            f"macro-F1={macro_f1:.4f}, loss={test_loss:.4f}, "
            f"time={training_seconds:.2f}s"
        )

    summarize(
        results,
        ["accuracy", "macro_f1", "test_loss", "training_seconds"],
        OUT / "summary_mean_std.csv",
    )

    # Combine run-level results for a direct three-model comparison.
    old_runs_path = SOURCE / "runs.csv"
    with open(old_runs_path, newline="") as f:
        old_runs = list(csv.DictReader(f))

    comparison = []
    for old in old_runs:
        comparison.append({
            "model": old["model"],
            "seed": int(old["seed"]),
            "accuracy": float(old["accuracy"]),
            "macro_f1": float(old["macro_f1"]),
            "test_loss": float(old["test_loss"]),
            "training_seconds": float(old["training_seconds"]),
        })

    for row in results:
        comparison.append({
            "model": row["model"],
            "seed": row["seed"],
            "accuracy": row["accuracy"],
            "macro_f1": row["macro_f1"],
            "test_loss": row["test_loss"],
            "training_seconds": row["training_seconds"],
        })

    write_csv(OUT / "comparison_all_runs.csv", comparison)

    comparison_summary = []
    for model_name in (
        "classical", "nonlinear_classical_control", "hybrid"
    ):
        model_rows = [r for r in comparison if r["model"] == model_name]
        for metric in (
            "accuracy", "macro_f1", "test_loss", "training_seconds"
        ):
            values = np.array([r[metric] for r in model_rows], dtype=float)
            comparison_summary.append({
                "model": model_name,
                "metric": metric,
                "mean": float(values.mean()),
                "std_sample": float(values.std(ddof=1)),
                "seeds": len(values),
            })

    write_csv(OUT / "comparison_mean_std.csv", comparison_summary)

    print("\nAll three control runs completed.")
    print(f"Results saved in: {OUT}")
    print("Main comparison: comparison_mean_std.csv")


if __name__ == "__main__":
    main()
