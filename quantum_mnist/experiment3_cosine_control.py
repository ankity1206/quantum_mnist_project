import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score

from config import CONFIG


SEEDS = [42, 123, 2026]
SOURCE = CONFIG.project_dir / "outputs" / "fair_classical_vs_quantum"
OUT = CONFIG.project_dir / "outputs" / "experiment3_cosine_control"
OUT.mkdir(parents=True, exist_ok=True)


class CosineFeatureControl(nn.Module):
    """
    Classical equivalent of the 3-layer, no-entanglement circuit.

    Each circuit output is:
        cos(x_i + theta_i1 + theta_i2 + theta_i3)

    The same 24 trainable phase parameters and the same
    8 -> 16 -> 10 classifier head are retained.
    """

    def __init__(self, num_features=8, num_classes=10, layers=3,
                 hidden_size=16):
        super().__init__()
        self.phase_weights = nn.Parameter(
            0.1 * torch.randn(layers, num_features)
        )
        self.classifier = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        phase = self.phase_weights.sum(dim=0)
        features = torch.cos(x + phase)
        return self.classifier(features)


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    manifest = SOURCE / "dataset_manifest.npz"
    if not manifest.exists():
        raise FileNotFoundError(manifest)

    with np.load(manifest, allow_pickle=False) as data:
        X_train = torch.from_numpy(data["train_features"].copy()).float()
        y_train = torch.from_numpy(data["y_train"].copy()).long()
        X_test = torch.from_numpy(data["test_features"].copy()).float()
        y_test = torch.from_numpy(data["y_test"].copy()).long()

    assert X_train.shape == (2000, 8)
    assert X_test.shape == (1000, 8)

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)

    # Use the same GPU preference as the classical benchmark.
    device = torch.device(
        "cuda" if CONFIG.device == "cuda" and torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")
    print(f"Train/test: {len(train_ds)}/{len(test_ds)}")
    print(f"Epochs={CONFIG.epochs}, batch={CONFIG.batch_size}, "
          f"lr={CONFIG.learning_rate}")
    print(f"Output: {OUT}")

    results = []
    histories = []

    for seed in SEEDS:
        print(f"\n{'=' * 65}")
        print(f"Classical cosine-feature control | seed={seed}")
        print(f"{'=' * 65}")

        seed_everything(seed)
        generator = torch.Generator().manual_seed(seed)

        train_loader = DataLoader(
            train_ds, batch_size=CONFIG.batch_size,
            shuffle=True, generator=generator
        )
        test_loader = DataLoader(
            test_ds, batch_size=CONFIG.batch_size, shuffle=False
        )

        model = CosineFeatureControl(
            num_features=8, num_classes=CONFIG.num_classes,
            layers=3, hidden_size=16
        ).to(device)

        parameter_count = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=CONFIG.learning_rate
        )

        start = time.time()

        for epoch in range(1, CONFIG.epochs + 1):
            model.train()
            loss_sum = 0.0
            correct = 0
            count = 0

            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                loss_sum += loss.item() * len(yb)
                correct += (logits.argmax(dim=1) == yb).sum().item()
                count += len(yb)

            train_loss = loss_sum / count
            train_acc = correct / count

            histories.append({
                "model": "cosine_feature_control",
                "seed": seed,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
            })

            print(
                f"Epoch {epoch:02d}/{CONFIG.epochs} | "
                f"loss={train_loss:.4f} | acc={train_acc:.2%}"
            )

        training_seconds = time.time() - start

        model.eval()
        y_true, y_pred = [], []
        loss_sum, count = 0.0, 0

        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss_sum += loss.item() * len(yb)
                count += len(yb)
                y_true.extend(yb.cpu().tolist())
                y_pred.extend(logits.argmax(dim=1).cpu().tolist())

        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        test_loss = loss_sum / count

        torch.save(
            model.state_dict(), OUT / f"cosine_control_seed{seed}.pt"
        )

        result = {
            "model": "cosine_feature_control",
            "seed": seed,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "test_loss": test_loss,
            "training_seconds": training_seconds,
            "device": str(device),
            "train_examples": len(train_ds),
            "test_examples": len(test_ds),
            "trainable_parameters": parameter_count,
            "epochs": CONFIG.epochs,
            "batch_size": CONFIG.batch_size,
            "learning_rate": CONFIG.learning_rate,
        }
        results.append(result)
        write_csv(OUT / "runs.csv", results)
        write_csv(OUT / "training_history.csv", histories)

        print(
            f"FINAL seed={seed}: accuracy={accuracy:.2%}, "
            f"macro-F1={macro_f1:.4f}, loss={test_loss:.4f}, "
            f"time={training_seconds:.1f}s, params={parameter_count}"
        )

    # Compare against existing results without retraining old models.
    comparison = []

    old_runs_path = SOURCE / "runs.csv"
    with open(old_runs_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["model"] == "classical":
                comparison.append({
                    "model": "classical_baseline",
                    "seed": int(row["seed"]),
                    "accuracy": float(row["accuracy"]),
                    "macro_f1": float(row["macro_f1"]),
                    "test_loss": float(row["test_loss"]),
                    "training_seconds": float(row["training_seconds"]),
                })

    ablation_runs_path = (
        CONFIG.project_dir / "outputs" /
        "experiment2_circuit_ablation" / "runs.csv"
    )
    with open(ablation_runs_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["model"] == "layers3_no_entanglement":
                comparison.append({
                    "model": "hybrid_3layer_no_entanglement",
                    "seed": int(row["seed"]),
                    "accuracy": float(row["accuracy"]),
                    "macro_f1": float(row["macro_f1"]),
                    "test_loss": float(row["test_loss"]),
                    "training_seconds": float(row["training_seconds"]),
                })

    for row in results:
        comparison.append({
            key: row[key] for key in (
                "model", "seed", "accuracy", "macro_f1",
                "test_loss", "training_seconds"
            )
        })

    write_csv(OUT / "comparison_all_runs.csv", comparison)

    summary = []
    for name in sorted(set(r["model"] for r in comparison)):
        group = [r for r in comparison if r["model"] == name]
        for metric in (
            "accuracy", "macro_f1", "test_loss", "training_seconds"
        ):
            values = np.array([float(r[metric]) for r in group])
            summary.append({
                "model": name,
                "metric": metric,
                "mean": float(values.mean()),
                "std_sample": (
                    float(values.std(ddof=1)) if len(values) > 1 else 0.0
                ),
                "seeds": len(values),
            })

    write_csv(OUT / "comparison_mean_std.csv", summary)

    print("\nExperiment complete.")
    print(f"Summary: {OUT / 'comparison_mean_std.csv'}")


if __name__ == "__main__":
    main()
