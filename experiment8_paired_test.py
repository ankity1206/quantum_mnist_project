from pathlib import Path
from dataclasses import replace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import binomtest
from torchvision import datasets, transforms

from config import CONFIG
from data_loader import load_mnist
from preprocessing import fit_preprocessor

ROOT = Path(__file__).resolve().parent
EXP4 = ROOT / "outputs" / "experiment4_40k"
EXP5 = ROOT / "outputs" / "experiment5_parameter_match"
OUT = ROOT / "outputs" / "experiment8_paired_test"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 2026]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MatchedClassicalMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(8, 17), nn.ReLU(), nn.Linear(17, 10)
        )

    def forward(self, x):
        return self.network(x)


class CosineFeatureControl(nn.Module):
    def __init__(self):
        super().__init__()
        self.phase_weights = nn.Parameter(0.1 * torch.randn(3, 8))
        self.classifier = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 10)
        )

    def forward(self, x):
        phase = self.phase_weights.sum(dim=0)
        return self.classifier(torch.cos(x + phase))


def predict(model, X, batch_size=512):
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = X[start:start + batch_size].to(DEVICE)
            predictions.extend(
                model(batch).argmax(dim=1).cpu().numpy().tolist()
            )
    return np.asarray(predictions, dtype=np.int64)


def holm_adjust(p_values):
    """Holm correction for the three per-seed hypothesis tests."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p), dtype=float)
    running_max = 0.0
    m = len(p)

    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running_max = max(running_max, value)
        adjusted[idx] = running_max

    return adjusted


def main():
    print("Device:", DEVICE)

    cfg = replace(
        CONFIG,
        train_samples_per_class=4000,
        test_samples_per_class=100,
    )
    X_train_raw, y_train, _, _ = load_mnist(cfg)

    with np.load(EXP4 / "dataset_manifest.npz") as data:
        saved_train = data["train_features"].copy()
        saved_y = data["y_train"].copy()

    if not np.array_equal(y_train.numpy(), saved_y):
        raise RuntimeError("Training labels differ from the saved manifest.")

    X_train_features, _, pca, scaler = fit_preprocessor(
        X_train_raw, X_train_raw[:1], n_components=8
    )
    max_diff = float(np.max(np.abs(X_train_features.numpy() - saved_train)))
    print("Max difference vs saved training features:", max_diff)

    if not np.allclose(
        X_train_features.numpy(), saved_train, rtol=1e-5, atol=1e-6
    ):
        raise RuntimeError("Preprocessing verification failed.")

    # Use all 10,000 official MNIST test examples in their original order.
    test_ds = datasets.MNIST(
        root=str(cfg.data_dir),
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    X_raw = np.stack(
        [img.reshape(-1).numpy() for img, _ in test_ds]
    ).astype(np.float32)
    y = np.array([label for _, label in test_ds], dtype=np.int64)

    X_angles = scaler.transform(pca.transform(X_raw)).astype(np.float32)
    X = torch.from_numpy(X_angles)

    if len(y) != 10000:
        raise RuntimeError(f"Expected 10000 test examples, got {len(y)}")

    rows = []
    prediction_rows = []

    for seed in SEEDS:
        models = {
            "matched_classical_mlp": MatchedClassicalMLP(),
            "cosine_feature_control": CosineFeatureControl(),
        }
        predictions = {}

        for name, model in models.items():
            checkpoint = EXP5 / f"{name}_seed{seed}.pt"
            model.load_state_dict(
                torch.load(checkpoint, map_location="cpu", weights_only=True)
            )
            model = model.to(DEVICE)
            predictions[name] = predict(model, X)

        mlp_correct = predictions["matched_classical_mlp"] == y
        cos_correct = predictions["cosine_feature_control"] == y

        # b: MLP wrong, cosine correct.
        # c: MLP correct, cosine wrong.
        b = int(np.sum(~mlp_correct & cos_correct))
        c = int(np.sum(mlp_correct & ~cos_correct))
        discordant = b + c

        p_value = (
            float(binomtest(min(b, c), n=discordant, p=0.5).pvalue)
            if discordant else 1.0
        )

        rows.append({
            "seed": seed,
            "test_examples": len(y),
            "mlp_accuracy": float(mlp_correct.mean()),
            "cosine_accuracy": float(cos_correct.mean()),
            "accuracy_difference_cosine_minus_mlp":
                float(cos_correct.mean() - mlp_correct.mean()),
            "mlp_wrong_cosine_correct_b": b,
            "mlp_correct_cosine_wrong_c": c,
            "discordant_examples": discordant,
            "mcnemar_exact_p": p_value,
        })

        for i in range(len(y)):
            prediction_rows.append({
                "seed": seed,
                "index": i,
                "actual": int(y[i]),
                "mlp_prediction": int(predictions["matched_classical_mlp"][i]),
                "cosine_prediction": int(predictions["cosine_feature_control"][i]),
                "mlp_correct": bool(mlp_correct[i]),
                "cosine_correct": bool(cos_correct[i]),
            })

        print(
            f"seed={seed}: MLP={mlp_correct.mean():.4%}, "
            f"cosine={cos_correct.mean():.4%}, "
            f"b={b}, c={c}, exact McNemar p={p_value:.6g}"
        )

    results = pd.DataFrame(rows)
    results["holm_adjusted_p"] = holm_adjust(results["mcnemar_exact_p"])
    results.to_csv(OUT / "paired_test_results.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(
        OUT / "per_image_predictions.csv", index=False
    )

    print("\n=== PAIRED TEST RESULTS ===")
    print(results.round(6).to_string(index=False))
    print("\nSaved to:", OUT)
    print(
        "\nInterpretation: b counts images only the cosine model gets right; "
        "c counts images only the MLP gets right. "
        "Holm-adjusted p-values account for testing three seeds."
    )
    print(
        "These tests are per seed. Do not treat the three runs as independent "
        "test sets; they reuse the same 10,000 images."
    )


if __name__ == "__main__":
    main()
