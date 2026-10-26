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
from models import HybridQuantumClassifier
from train import train_one_epoch


SEEDS = [42, 123, 2026]
VARIANTS = [
    {"name": "layers1_entangled", "layers": 1, "entanglement": True},
    {"name": "layers2_entangled", "layers": 2, "entanglement": True},
    {"name": "layers3_no_entanglement", "layers": 3, "entanglement": False},
]

BASE = CONFIG.project_dir / "outputs"
SOURCE = BASE / "fair_classical_vs_quantum"
OUT = BASE / "experiment2_circuit_ablation"
OUT.mkdir(parents=True, exist_ok=True)


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
    for model_name in sorted(set(r["model"] for r in rows)):
        model_rows = [r for r in rows if r["model"] == model_name]
        for metric in metrics:
            values = np.array([float(r[metric]) for r in model_rows])
            output.append({
                "model": model_name,
                "metric": metric,
                "mean": float(values.mean()),
                "std_sample": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "seeds": len(values),
            })
    write_csv(path, output)


def main():
    manifest_path = SOURCE / "dataset_manifest.npz"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    with np.load(manifest_path, allow_pickle=False) as data:
        X_train = torch.from_numpy(data["train_features"].copy()).float()
        y_train = torch.from_numpy(data["y_train"].copy()).long()
        X_test = torch.from_numpy(data["test_features"].copy()).float()
        y_test = torch.from_numpy(data["y_test"].copy()).long()

    assert X_train.shape == (2000, 8)
    assert X_test.shape == (1000, 8)
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    device = torch.device("cpu")  # Match the existing quantum benchmark.
    print(f"Device: {device}")
    print(f"Train/test: {len(train_dataset)}/{len(test_dataset)}")
    print(f"Epochs={CONFIG.epochs}, batch={CONFIG.batch_size}, lr={CONFIG.learning_rate}")
    print(f"Output: {OUT}")

    results, histories = [], []

    for variant in VARIANTS:
        for seed in SEEDS:
            name = variant["name"]
            layers = variant["layers"]
            entanglement = variant["entanglement"]
            tag = f"{name}_seed{seed}"

            print("\n" + "=" * 70)
            print(f"{name} | seed={seed}")
            print("=" * 70)

            seed_everything(seed)
            generator = torch.Generator().manual_seed(seed)
            train_loader = DataLoader(
                train_dataset,
                batch_size=CONFIG.batch_size,
                shuffle=True,
                generator=generator,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=CONFIG.batch_size,
                shuffle=False,
            )

            model = HybridQuantumClassifier(
                num_qubits=CONFIG.num_qubits,
                num_classes=CONFIG.num_classes,
                num_layers=layers,
                hidden_size=16,
                entanglement=entanglement,
            ).to(device)

            parameter_count = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            circuit_weight_count = model.quantum_layer.weights.numel()

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(
                model.parameters(), lr=CONFIG.learning_rate
            )

            start = time.time()
            for epoch in range(1, CONFIG.epochs + 1):
                epoch_start = time.time()
                train_loss, train_acc = train_one_epoch(
                    model, train_loader, criterion, optimizer, device
                )
                elapsed = time.time() - epoch_start

                histories.append({
                    "model": name,
                    "seed": seed,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "epoch_seconds": elapsed,
                })
                print(
                    f"Epoch {epoch:02d}/{CONFIG.epochs} | "
                    f"loss={train_loss:.4f} | acc={train_acc:.2%} | "
                    f"time={elapsed:.1f}s"
                )

            training_seconds = time.time() - start

            model.eval()
            y_true, y_pred = [], []
            loss_sum, sample_count = 0.0, 0

            with torch.no_grad():
                for inputs, labels in test_loader:
                    logits = model(inputs)
                    loss = criterion(logits, labels)
                    loss_sum += loss.item() * len(labels)
                    sample_count += len(labels)
                    y_true.extend(labels.tolist())
                    y_pred.extend(logits.argmax(dim=1).tolist())

            accuracy = accuracy_score(y_true, y_pred)
            macro_f1 = f1_score(
                y_true, y_pred, average="macro", zero_division=0
            )
            test_loss = loss_sum / sample_count

            torch.save(model.state_dict(), OUT / f"{tag}.pt")

            report = classification_report(
                y_true,
                y_pred,
                labels=list(range(CONFIG.num_classes)),
                output_dict=True,
                zero_division=0,
            )
            write_csv(
                OUT / f"{tag}_report.csv",
                [
                    {"class": label, **values}
                    for label, values in report.items()
                    if isinstance(values, dict)
                ],
            )
            np.savetxt(
                OUT / f"{tag}_confusion_matrix.csv",
                confusion_matrix(
                    y_true, y_pred, labels=list(range(CONFIG.num_classes))
                ),
                delimiter=",",
                fmt="%d",
            )

            result = {
                "model": name,
                "seed": seed,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "test_loss": test_loss,
                "training_seconds": training_seconds,
                "device": str(device),
                "train_examples": len(train_dataset),
                "test_examples": len(test_dataset),
                "features": CONFIG.num_qubits,
                "quantum_layers": layers,
                "entanglement": entanglement,
                "circuit_trainable_weights": circuit_weight_count,
                "total_trainable_parameters": parameter_count,
                "epochs": CONFIG.epochs,
                "batch_size": CONFIG.batch_size,
                "learning_rate": CONFIG.learning_rate,
                "loss": "unweighted_cross_entropy",
            }
            results.append(result)
            write_csv(OUT / "runs.csv", results)
            write_csv(OUT / "training_history.csv", histories)

            print(
                f"FINAL {tag}: accuracy={accuracy:.2%}, "
                f"macro-F1={macro_f1:.4f}, loss={test_loss:.4f}, "
                f"time={training_seconds / 60:.1f} min"
            )

    # Reuse the existing 3-layer entangled baseline, whose benchmark settings
    # match this experiment, instead of retraining it.
    with open(SOURCE / "runs.csv", newline="") as f:
        old_runs = list(csv.DictReader(f))

    comparison = []
    for row in old_runs:
        if row["model"] != "hybrid":
            continue
        if int(row["quantum_layers"]) != 3:
            continue
        comparison.append({
            "model": "layers3_entangled",
            "seed": int(row["seed"]),
            "accuracy": float(row["accuracy"]),
            "macro_f1": float(row["macro_f1"]),
            "test_loss": float(row["test_loss"]),
            "training_seconds": float(row["training_seconds"]),
            "quantum_layers": 3,
            "entanglement": True,
        })

    for row in results:
        comparison.append({
            key: row[key] for key in (
                "model", "seed", "accuracy", "macro_f1", "test_loss",
                "training_seconds", "quantum_layers", "entanglement"
            )
        })

    write_csv(OUT / "comparison_all_runs.csv", comparison)
    summarize(
        comparison,
        ["accuracy", "macro_f1", "test_loss", "training_seconds"],
        OUT / "comparison_mean_std.csv",
    )

    print("\nAblation runs complete.")
    print(f"Comparison: {OUT / 'comparison_mean_std.csv'}")


if __name__ == "__main__":
    main()
