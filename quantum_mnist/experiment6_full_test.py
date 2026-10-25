from pathlib import Path
from dataclasses import replace
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from config import CONFIG
from data_loader import load_mnist
from preprocessing import fit_preprocessor


ROOT = Path(__file__).resolve().parent
EXP4_DIR = ROOT / "outputs" / "experiment4_40k"
EXP5_DIR = ROOT / "outputs" / "experiment5_parameter_match"
OUT_DIR = ROOT / "outputs" / "experiment6_full_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 2026]
BATCH_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MatchedClassicalMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(8, 17),
            nn.ReLU(),
            nn.Linear(17, 10),
        )

    def forward(self, x):
        return self.network(x)


class CosineFeatureControl(nn.Module):
    def __init__(self):
        super().__init__()
        self.phase_weights = nn.Parameter(0.1 * torch.randn(3, 8))
        self.classifier = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 10),
        )

    def forward(self, x):
        phase = self.phase_weights.sum(dim=0)
        return self.classifier(torch.cos(x + phase))


def main():
    print("Device:", DEVICE)
    print("Output directory:", OUT_DIR)

    # Load the exact 40,000-example training subset used in Experiment 4.
    cfg = replace(
        CONFIG,
        train_samples_per_class=4000,
        test_samples_per_class=100,
    )
    X_train_raw, _, _, _ = load_mnist(cfg)

    manifest_path = EXP4_DIR / "dataset_manifest.npz"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    with np.load(manifest_path) as manifest:
        saved_train_features = manifest["train_features"].copy()
        y_train_saved = manifest["y_train"].copy()

    if len(X_train_raw) != 40000:
        raise RuntimeError(f"Expected 40000 training examples, got {len(X_train_raw)}")

    if not np.array_equal(y_train_saved, load_mnist(cfg)[1].numpy()):
        raise RuntimeError(
            "Training label order differs from the saved manifest. "
            "Stopping to avoid evaluating incompatible preprocessing."
        )

    print("\nReconstructing PCA and scaler from training data only...")
    # The selected 1,000-image subset is only needed to satisfy the helper;
    # the PCA/scaler are fitted on X_train_raw, not on test images.
    _, _, X_subset_raw, _ = load_mnist(cfg)
    X_train_features, _, pca, scaler = fit_preprocessor(
        X_train_raw, X_subset_raw, n_components=8
    )

    # Strict check that reconstructed training features match Experiment 4.
    reconstructed = X_train_features.numpy()
    max_abs_diff = float(
        np.max(np.abs(reconstructed - saved_train_features))
    )
    print(f"Maximum training-feature difference: {max_abs_diff:.10g}")

    if not np.allclose(
        reconstructed, saved_train_features, rtol=1e-5, atol=1e-6
    ):
        raise RuntimeError(
            "Reconstructed training features do not match the saved manifest. "
            "Evaluation aborted; do not interpret results."
        )

    print("Preprocessing verification passed.")

    # Load ALL official MNIST test examples, without class-balanced subsampling.
    test_dataset = datasets.MNIST(
        root=str(cfg.data_dir),
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    X_full_raw = np.stack(
        [image.reshape(-1).numpy() for image, _ in test_dataset]
    ).astype(np.float32)
    y_full = np.array([label for _, label in test_dataset], dtype=np.int64)

    if len(y_full) != 10000:
        raise RuntimeError(f"Expected 10000 official test images, got {len(y_full)}")

    # Transform with the reconstructed training-fitted PCA and scaler.
    full_pca = pca.transform(X_full_raw)
    full_angles = scaler.transform(full_pca).astype(np.float32)
    X_full = torch.from_numpy(full_angles)
    y_full_tensor = torch.from_numpy(y_full).long()

    print("Full test feature shape:", tuple(X_full.shape))
    print("Test class counts:", np.bincount(y_full, minlength=10).tolist())

    loader = DataLoader(
        TensorDataset(X_full, y_full_tensor),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    criterion = nn.CrossEntropyLoss()

    model_specs = [
        ("matched_classical_mlp", MatchedClassicalMLP),
        ("cosine_feature_control", CosineFeatureControl),
    ]
    run_rows = []
    per_class_rows = []

    for model_name, model_class in model_specs:
        for seed in SEEDS:
            checkpoint = EXP5_DIR / f"{model_name}_seed{seed}.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)

            model = model_class().to(DEVICE)
            state = torch.load(
                checkpoint, map_location=DEVICE, weights_only=True
            )
            model.load_state_dict(state)
            model.eval()

            all_labels = []
            all_predictions = []
            total_loss = 0.0
            total = 0

            start = time.perf_counter()
            with torch.no_grad():
                for features, labels in loader:
                    features = features.to(DEVICE)
                    labels = labels.to(DEVICE)
                    logits = model(features)
                    loss = criterion(logits, labels)

                    batch_n = labels.size(0)
                    total_loss += loss.item() * batch_n
                    total += batch_n
                    all_labels.extend(labels.cpu().numpy().tolist())
                    all_predictions.extend(
                        logits.argmax(dim=1).cpu().numpy().tolist()
                    )

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            accuracy = accuracy_score(all_labels, all_predictions)
            macro_f1 = f1_score(
                all_labels, all_predictions,
                average="macro", zero_division=0
            )
            avg_loss = total_loss / total

            run_rows.append({
                "model": model_name,
                "seed": seed,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "test_loss": avg_loss,
                "evaluation_seconds": elapsed,
                "test_examples": total,
                "device": str(DEVICE),
                "checkpoint": checkpoint.name,
            })

            precision, recall, f1, support = precision_recall_fscore_support(
                all_labels, all_predictions,
                labels=list(range(10)),
                zero_division=0,
            )
            for digit in range(10):
                per_class_rows.append({
                    "model": model_name,
                    "seed": seed,
                    "digit": digit,
                    "precision": precision[digit],
                    "recall": recall[digit],
                    "f1": f1[digit],
                    "support": int(support[digit]),
                })

            cm = confusion_matrix(
                all_labels, all_predictions, labels=list(range(10))
            )
            pd.DataFrame(
                cm,
                index=[f"actual_{i}" for i in range(10)],
                columns=[f"predicted_{i}" for i in range(10)],
            ).to_csv(OUT_DIR / f"confusion_{model_name}_seed{seed}.csv")

            print(
                f"{model_name:24s} seed={seed:4d} "
                f"accuracy={accuracy:.4f} macro-F1={macro_f1:.4f} "
                f"loss={avg_loss:.4f} time={elapsed:.1f}s"
            )

    runs = pd.DataFrame(run_rows)
    runs.to_csv(OUT_DIR / "full_test_runs.csv", index=False)

    summary = runs.groupby("model")[
        ["accuracy", "macro_f1", "test_loss", "evaluation_seconds"]
    ].agg(["mean", "std"])
    summary.to_csv(OUT_DIR / "full_test_summary.csv")

    pd.DataFrame(per_class_rows).to_csv(
        OUT_DIR / "per_class_metrics.csv", index=False
    )

    verification = {
        "test_examples": int(len(y_full)),
        "train_examples_used_for_preprocessing": int(len(X_train_raw)),
        "num_features": int(X_full.shape[1]),
        "max_abs_train_feature_difference": max_abs_diff,
        "preprocessing_verification_passed": True,
        "test_class_counts": np.bincount(y_full, minlength=10).tolist(),
        "device": str(DEVICE),
        "seeds": SEEDS,
    }
    with open(OUT_DIR / "verification.json", "w") as f:
        json.dump(verification, f, indent=2)

    print("\n=== FULL 10,000-IMAGE TEST RESULTS ===")
    print(runs[[
        "model", "seed", "accuracy", "macro_f1",
        "test_loss", "evaluation_seconds"
    ]].to_string(index=False))

    print("\n=== MEAN ± STANDARD DEVIATION ACROSS SEEDS ===")
    print(summary.round(5).to_string())

    print(f"\nAll results saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
