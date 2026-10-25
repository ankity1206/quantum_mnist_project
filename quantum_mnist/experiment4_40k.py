import csv
import random
import time
from dataclasses import replace

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score

from config import CONFIG
from data_loader import load_mnist
from preprocessing import fit_preprocessor


SEEDS = [42, 123, 2026]
SAMPLES_PER_CLASS = 4000
LAYERS = 3
HIDDEN_SIZE = 16

OUT = CONFIG.project_dir / "outputs" / "experiment4_40k"
OUT.mkdir(parents=True, exist_ok=True)

RUNS_FILE = OUT / "runs.csv"
HISTORY_FILE = OUT / "training_history.csv"

if RUNS_FILE.exists():
    raise FileExistsError(
        f"{RUNS_FILE} already exists. "
        "Move the old experiment folder before rerunning."
    )


class CosineFeatureControl(nn.Module):
    """Classical cosine feature map with a trainable classifier."""

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
        return self.classifier(torch.cos(x + phase))


class ClassicalBaseline(nn.Module):
    """The existing 8 -> 16 -> 10 classical MLP."""

    def __init__(self, num_features=8, num_classes=10, hidden_size=16):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        return self.network(x)


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


def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    all_true, all_pred = [], []
    loss_sum, count = 0.0, 0

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            loss_sum += loss.item() * len(yb)
            count += len(yb)
            all_true.extend(yb.cpu().tolist())
            all_pred.extend(logits.argmax(dim=1).cpu().tolist())

    return {
        "accuracy": accuracy_score(all_true, all_pred),
        "macro_f1": f1_score(
            all_true, all_pred, average="macro", zero_division=0
        ),
        "test_loss": loss_sum / count,
    }


def main():
    # Change only the training subset size. The test selection remains
    # 100 examples per class and uses the original test-selection seed.
    cfg = replace(
        CONFIG,
        train_samples_per_class=SAMPLES_PER_CLASS,
        test_samples_per_class=100,
    )

    print("Loading balanced MNIST subsets...")
    X_train_raw, y_train, X_test_raw, y_test = load_mnist(cfg)

    assert len(X_train_raw) == 40000
    assert len(X_test_raw) == 1000
    assert torch.equal(
        torch.bincount(y_train), torch.full((10,), 4000)
    )
    assert torch.equal(
        torch.bincount(y_test), torch.full((10,), 100)
    )

    print("Fitting PCA and angle scaler on training images only...")
    X_train, X_test, pca, scaler = fit_preprocessor(
        X_train_raw, X_test_raw, n_components=8
    )

    # Save the exact processed dataset and preprocessing metadata.
    np.savez_compressed(
        OUT / "dataset_manifest.npz",
        train_features=X_train.numpy(),
        y_train=y_train.numpy(),
        test_features=X_test.numpy(),
        y_test=y_test.numpy(),
        explained_variance_ratio=pca.explained_variance_ratio_,
        scaler_min=scaler.data_min_,
        scaler_max=scaler.data_max_,
    )

    print("Train shape:", tuple(X_train.shape))
    print("Test shape:", tuple(X_test.shape))
    print("Training class counts:", torch.bincount(y_train).tolist())
    print("Test class counts:", torch.bincount(y_test).tolist())
    print("PCA variance retained:",
          round(float(pca.explained_variance_ratio_.sum()), 4))

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)

    device = torch.device(
        "cuda" if CONFIG.device == "cuda" and torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)
    print(
        f"Epochs={CONFIG.epochs}, batch={CONFIG.batch_size}, "
        f"learning_rate={CONFIG.learning_rate}"
    )
    print("Output:", OUT)

    results, histories = [], []

    model_types = [
        ("classical_baseline", ClassicalBaseline),
        ("cosine_feature_control", CosineFeatureControl),
    ]

    for model_name, model_class in model_types:
        for seed in SEEDS:
            print(f"\n{'=' * 60}")
            print(f"{model_name} | seed={seed}")
            print(f"{'=' * 60}")

            seed_everything(seed)
            generator = torch.Generator().manual_seed(seed)

            train_loader = DataLoader(
                train_ds,
                batch_size=CONFIG.batch_size,
                shuffle=True,
                generator=generator,
                num_workers=0,
            )
            test_loader = DataLoader(
                test_ds,
                batch_size=CONFIG.batch_size,
                shuffle=False,
                num_workers=0,
            )

            if model_name == "classical_baseline":
                model = model_class(
                    num_features=8,
                    num_classes=10,
                    hidden_size=HIDDEN_SIZE,
                )
            else:
                model = model_class(
                    num_features=8,
                    num_classes=10,
                    layers=LAYERS,
                    hidden_size=HIDDEN_SIZE,
                )

            model = model.to(device)
            parameter_count = sum(
                p.numel() for p in model.parameters()
                if p.requires_grad
            )

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(
                model.parameters(), lr=CONFIG.learning_rate
            )

            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            for epoch in range(1, CONFIG.epochs + 1):
                model.train()
                loss_sum, correct, count = 0.0, 0, 0

                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    optimizer.step()

                    loss_sum += loss.item() * len(yb)
                    correct += (
                        logits.argmax(dim=1) == yb
                    ).sum().item()
                    count += len(yb)

                history_row = {
                    "model": model_name,
                    "seed": seed,
                    "epoch": epoch,
                    "train_loss": loss_sum / count,
                    "train_accuracy": correct / count,
                }
                histories.append(history_row)
                write_csv(HISTORY_FILE, histories)

                print(
                    f"Epoch {epoch:02d}/{CONFIG.epochs} | "
                    f"train loss={history_row['train_loss']:.4f} | "
                    f"train accuracy={history_row['train_accuracy']:.2%}"
                )

            if device.type == "cuda":
                torch.cuda.synchronize()
            training_seconds = time.perf_counter() - start

            metrics = evaluate(model, test_loader, device)

            torch.save(
                model.state_dict(),
                OUT / f"{model_name}_seed{seed}.pt",
            )

            result = {
                "model": model_name,
                "seed": seed,
                **metrics,
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
            write_csv(RUNS_FILE, results)

            print(
                f"FINAL: accuracy={metrics['accuracy']:.2%}, "
                f"macro-F1={metrics['macro_f1']:.4f}, "
                f"loss={metrics['test_loss']:.4f}, "
                f"time={training_seconds:.1f}s, "
                f"parameters={parameter_count}"
            )

    summary = []
    for model_name in sorted(set(r["model"] for r in results)):
        group = [r for r in results if r["model"] == model_name]
        for metric in (
            "accuracy", "macro_f1", "test_loss", "training_seconds"
        ):
            values = np.array([r[metric] for r in group], dtype=float)
            summary.append({
                "model": model_name,
                "metric": metric,
                "mean": float(values.mean()),
                "std_sample": float(values.std(ddof=1)),
                "seeds": len(values),
            })

    write_csv(OUT / "comparison_mean_std.csv", summary)

    print("\nExperiment complete.")
    print("Runs:", RUNS_FILE)
    print("Summary:", OUT / "comparison_mean_std.csv")


if __name__ == "__main__":
    main()
