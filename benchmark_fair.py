import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import CONFIG
from data_loader import load_mnist
from preprocessing import fit_preprocessor
from models import HybridQuantumClassifier, ClassicalBaseline
from train import train_one_epoch


# All outputs go to a new directory. Existing experiment folders are untouched.
OUT = CONFIG.project_dir / "outputs" / "fair_classical_vs_quantum"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 2026]
QUANTUM_LAYERS = 3
EPOCHS = CONFIG.epochs
BATCH_SIZE = CONFIG.batch_size
LEARNING_RATE = CONFIG.learning_rate


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


def main():
    # Fixed sample selection for every run. Training seeds below affect
    # initialization and batch order, not which images enter the benchmark.
    data_config = CONFIG
    print("Loading the fixed MNIST subset...")
    X_train_raw, y_train, X_test_raw, y_test = load_mnist(data_config)

    X_train, X_test, pca, scaler = fit_preprocessor(
        X_train_raw,
        X_test_raw,
        n_components=CONFIG.num_qubits,
    )

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert set(y_train.tolist()) == set(range(CONFIG.num_classes))
    assert set(y_test.tolist()) == set(range(CONFIG.num_classes))

    # Persist the exact split labels and preprocessing diagnostics.
    np.savez_compressed(
        OUT / "dataset_manifest.npz",
        y_train=y_train.numpy(),
        y_test=y_test.numpy(),
        train_features=X_train.numpy(),
        test_features=X_test.numpy(),
        explained_variance_ratio=pca.explained_variance_ratio_,
        scaler_min=scaler.data_min_,
        scaler_max=scaler.data_max_,
    )

    train_dataset = TensorDataset(X_train.float(), y_train.long())
    test_dataset = TensorDataset(X_test.float(), y_test.long())

    quantum_device = torch.device("cpu")
    classical_device = torch.device(
        "cuda" if CONFIG.device == "cuda" and torch.cuda.is_available()
        else "cpu"
    )

    print(f"Training examples: {len(train_dataset)}")
    print(f"Test examples: {len(test_dataset)}")
    print(f"Features: {CONFIG.num_qubits}")
    print(f"Quantum layers: {QUANTUM_LAYERS}")
    print(f"Epochs: {EPOCHS}; batch size: {BATCH_SIZE}")
    print(f"Quantum device: {quantum_device}")
    print(f"Classical device: {classical_device}")
    print(f"Output: {OUT}")

    all_results = []
    all_histories = []

    for seed in SEEDS:
        for model_kind in ("classical", "hybrid"):
            print("\n" + "=" * 68)
            print(f"Model={model_kind} | seed={seed}")
            print("=" * 68)

            seed_everything(seed)

            # Identical shuffle seed for both model types at each repetition.
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

            if model_kind == "classical":
                model = ClassicalBaseline(
                    num_features=CONFIG.num_qubits,
                    num_classes=CONFIG.num_classes,
                    hidden_size=16,
                )
                device = classical_device
            else:
                model = HybridQuantumClassifier(
                    num_qubits=CONFIG.num_qubits,
                    num_classes=CONFIG.num_classes,
                    num_layers=QUANTUM_LAYERS,
                    hidden_size=16,
                )
                device = quantum_device

            model = model.to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=LEARNING_RATE,
            )

            history = []
            start_time = time.time()

            for epoch in range(1, EPOCHS + 1):
                epoch_start = time.time()
                train_loss, train_acc = train_one_epoch(
                    model, train_loader, criterion, optimizer, device
                )
                elapsed = time.time() - epoch_start

                record = {
                    "model": model_kind,
                    "seed": seed,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "epoch_seconds": elapsed,
                }
                history.append(record)
                all_histories.append(record)

                print(
                    f"Epoch {epoch:02d}/{EPOCHS} | "
                    f"train loss={train_loss:.4f} | "
                    f"train accuracy={train_acc:.2%} | "
                    f"time={elapsed:.1f}s"
                )

            total_seconds = time.time() - start_time

            # Test set is used only once, after all training is complete.
            model.eval()
            y_true, y_pred = [], []
            loss_sum = 0.0
            sample_count = 0

            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
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
            mean_loss = loss_sum / sample_count

            tag = f"{model_kind}_seed{seed}"
            torch.save(model.state_dict(), OUT / f"{tag}.pt")

            report = classification_report(
                y_true,
                y_pred,
                labels=list(range(CONFIG.num_classes)),
                output_dict=True,
                zero_division=0,
            )
            report_rows = []
            for label, values in report.items():
                if isinstance(values, dict):
                    report_rows.append({"class": label, **values})
            write_csv(OUT / f"{tag}_report.csv", report_rows)

            np.savetxt(
                OUT / f"{tag}_confusion_matrix.csv",
                confusion_matrix(
                    y_true, y_pred,
                    labels=list(range(CONFIG.num_classes)),
                ),
                delimiter=",",
                fmt="%d",
            )

            result = {
                "model": model_kind,
                "seed": seed,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "test_loss": mean_loss,
                "training_seconds": total_seconds,
                "device": str(device),
                "train_examples": len(train_dataset),
                "test_examples": len(test_dataset),
                "features": CONFIG.num_qubits,
                "quantum_layers": (
                    QUANTUM_LAYERS if model_kind == "hybrid" else 0
                ),
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "loss": "unweighted_cross_entropy",
            }
            all_results.append(result)

            print(
                f"FINAL {tag}: accuracy={accuracy:.2%}, "
                f"macro-F1={macro_f1:.4f}, "
                f"test loss={mean_loss:.4f}, "
                f"training time={total_seconds / 60:.1f} min"
            )

            # Save progress after each run, so partial results survive
            # if a later run is interrupted.
            write_csv(OUT / "runs.csv", all_results)
            write_csv(OUT / "training_history.csv", all_histories)

    # Aggregate across seeds: mean and sample standard deviation.
    summary = []
    for kind in ("classical", "hybrid"):
        rows = [r for r in all_results if r["model"] == kind]
        for metric in ("accuracy", "macro_f1", "test_loss", "training_seconds"):
            values = np.array([r[metric] for r in rows], dtype=float)
            summary.append({
                "model": kind,
                "metric": metric,
                "mean": float(values.mean()),
                "std_sample": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "seeds": len(values),
            })

    write_csv(OUT / "summary_mean_std.csv", summary)

    print("\nAll runs complete.")
    print("Run-level results:", OUT / "runs.csv")
    print("Mean/std summary:", OUT / "summary_mean_std.csv")


if __name__ == "__main__":
    main()
