from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "outputs" / "experiment4_40k"
OUTPUT_DIR = ROOT / "outputs" / "experiment5_parameter_match"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 2026]
EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MatchedClassicalMLP(nn.Module):
    """Classical MLP: 8 -> 17 -> 10, with 333 trainable parameters."""

    def __init__(self, num_features=8, hidden_size=17, num_classes=10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        return self.network(x)


class CosineFeatureControl(nn.Module):
    """Cosine-feature model: 338 trainable parameters."""

    def __init__(
        self, num_features=8, hidden_size=16, num_classes=10, layers=3
    ):
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, criterion):
    model.eval()
    all_labels = []
    all_predictions = []
    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(features)
            loss = criterion(logits, labels)

            count = labels.size(0)
            total_loss += loss.item() * count
            total_examples += count
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(logits.argmax(dim=1).cpu().numpy())

    return {
        "accuracy": accuracy_score(all_labels, all_predictions),
        "macro_f1": f1_score(
            all_labels, all_predictions, average="macro", zero_division=0
        ),
        "test_loss": total_loss / total_examples,
    }


def main():
    manifest_path = INPUT_DIR / "dataset_manifest.npz"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Could not find {manifest_path}. "
            "Check that Experiment 4 has completed."
        )

    data = np.load(manifest_path)

    X_train = torch.tensor(data["train_features"], dtype=torch.float32)
    y_train = torch.tensor(data["y_train"], dtype=torch.long)
    X_test = torch.tensor(data["test_features"], dtype=torch.float32)
    y_test = torch.tensor(data["y_test"], dtype=torch.long)

    if X_train.shape[0] != 40000:
        raise ValueError(f"Expected 40000 training examples; got {X_train.shape[0]}")
    if X_test.shape[0] != 1000:
        raise ValueError(f"Expected 1000 test examples; got {X_test.shape[0]}")

    print(f"Device: {DEVICE}")
    print(f"Training examples: {len(y_train):,}")
    print(f"Test examples: {len(y_test):,}")
    print(f"Input features: {X_train.shape[1]}")
    print(f"Output directory: {OUTPUT_DIR}")

    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=256,
        shuffle=False,
    )

    run_rows = []
    history_rows = []

    experiments = {
        "matched_classical_mlp": MatchedClassicalMLP,
        "cosine_feature_control": CosineFeatureControl,
    }

    for model_name, model_class in experiments.items():
        for seed in SEEDS:
            set_seed(seed)
            model = model_class().to(DEVICE)
            parameter_count = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )

            train_generator = torch.Generator().manual_seed(seed)
            train_loader = DataLoader(
                TensorDataset(X_train, y_train),
                batch_size=BATCH_SIZE,
                shuffle=True,
                generator=train_generator,
                num_workers=0,
            )

            optimizer = torch.optim.Adam(
                model.parameters(), lr=LEARNING_RATE
            )
            criterion = nn.CrossEntropyLoss()

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            for epoch in range(1, EPOCHS + 1):
                model.train()
                total_train_loss = 0.0
                total_train_examples = 0

                for features, labels in train_loader:
                    features = features.to(DEVICE)
                    labels = labels.to(DEVICE)

                    optimizer.zero_grad()
                    logits = model(features)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()

                    count = labels.size(0)
                    total_train_loss += loss.item() * count
                    total_train_examples += count

                epoch_train_loss = total_train_loss / total_train_examples
                metrics = evaluate(model, test_loader, criterion)

                history_rows.append({
                    "model": model_name,
                    "seed": seed,
                    "epoch": epoch,
                    "train_loss": epoch_train_loss,
                    **metrics,
                })

                print(
                    f"{model_name:24s} seed={seed:4d} "
                    f"epoch={epoch:02d}/{EPOCHS} "
                    f"train_loss={epoch_train_loss:.4f} "
                    f"test_acc={metrics['accuracy']:.4f}"
                )

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time

            final_metrics = evaluate(model, test_loader, criterion)
            run_rows.append({
                "model": model_name,
                "seed": seed,
                **final_metrics,
                "training_seconds": elapsed,
                "device": str(DEVICE),
                "train_examples": len(y_train),
                "test_examples": len(y_test),
                "trainable_parameters": parameter_count,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
            })

            checkpoint = OUTPUT_DIR / f"{model_name}_seed{seed}.pt"
            torch.save(model.state_dict(), checkpoint)

            print(
                f"Finished {model_name}, seed={seed}: "
                f"accuracy={final_metrics['accuracy']:.4f}, "
                f"macro-F1={final_metrics['macro_f1']:.4f}, "
                f"parameters={parameter_count}, time={elapsed:.1f}s\n"
            )

            pd.DataFrame(run_rows).to_csv(
                OUTPUT_DIR / "runs.csv", index=False
            )
            pd.DataFrame(history_rows).to_csv(
                OUTPUT_DIR / "training_history.csv", index=False
            )

    runs = pd.DataFrame(run_rows)
    runs.to_csv(OUTPUT_DIR / "runs.csv", index=False)

    metric_columns = [
        "accuracy", "macro_f1", "test_loss",
        "training_seconds", "trainable_parameters",
    ]
    summary = (
        runs.groupby("model")[metric_columns]
        .agg(["mean", "std"])
        .round(6)
    )
    summary.to_csv(OUTPUT_DIR / "comparison_mean_std.csv")

    print("\n=== RESULTS BY SEED ===")
    print(runs[[
        "model", "seed", "accuracy", "macro_f1",
        "test_loss", "training_seconds", "trainable_parameters"
    ]].to_string(index=False))

    print("\n=== MEAN AND STANDARD DEVIATION ACROSS SEEDS ===")
    print(summary)

    print(f"\nSaved all results to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
