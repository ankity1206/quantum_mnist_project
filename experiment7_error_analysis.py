from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "outputs" / "experiment6_full_test"
OUTPUT = ROOT / "outputs" / "experiment7_error_analysis"
OUTPUT.mkdir(parents=True, exist_ok=True)

MODELS = ["matched_classical_mlp", "cosine_feature_control"]
METRICS = ["precision", "recall", "f1"]


def main():
    per_class_path = INPUT / "per_class_metrics.csv"
    if not per_class_path.exists():
        raise FileNotFoundError(
            f"Missing {per_class_path}. Run Experiment 6 first."
        )

    data = pd.read_csv(per_class_path)

    required = {"model", "seed", "digit", *METRICS, "support"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Average each digit's performance across the three model seeds.
    summary = (
        data.groupby(["model", "digit"])[METRICS]
        .agg(["mean", "std"])
        .round(4)
    )
    summary.to_csv(OUTPUT / "per_digit_mean_std.csv")

    # Side-by-side mean metrics and cosine-minus-MLP differences.
    means = data.groupby(["model", "digit"])[METRICS].mean()
    rows = []
    for digit in range(10):
        row = {"digit": digit}
        for metric in METRICS:
            mlp = means.loc[("matched_classical_mlp", digit), metric]
            cosine = means.loc[("cosine_feature_control", digit), metric]
            row[f"mlp_{metric}"] = mlp
            row[f"cosine_{metric}"] = cosine
            row[f"delta_{metric}"] = cosine - mlp
        rows.append(row)

    comparison = pd.DataFrame(rows).round(4)
    comparison.to_csv(OUTPUT / "per_digit_comparison.csv", index=False)

    print("\n=== PER-DIGIT PERFORMANCE: COSINE MINUS MLP ===")
    print(comparison[[
        "digit", "delta_precision", "delta_recall", "delta_f1"
    ]].to_string(index=False))

    print("\n=== PER-DIGIT F1 COMPARISON ===")
    print(comparison[[
        "digit", "mlp_f1", "cosine_f1", "delta_f1"
    ]].to_string(index=False))

    print("\n=== DIGITS WITH LARGEST F1 IMPROVEMENTS ===")
    print(
        comparison.sort_values("delta_f1", ascending=False)[
            ["digit", "mlp_f1", "cosine_f1", "delta_f1"]
        ].head(5).to_string(index=False)
    )

    print("\n=== DIGITS WHERE COSINE F1 IS LOWER ===")
    lower = comparison[comparison["delta_f1"] < 0]
    if lower.empty:
        print("None: cosine F1 is not lower for any digit on average.")
    else:
        print(lower[[
            "digit", "mlp_f1", "cosine_f1", "delta_f1"
        ]].to_string(index=False))

    # Measure seed-by-seed F1 differences for each digit.
    pivot = data.pivot_table(
        index=["digit", "seed"],
        columns="model",
        values="f1",
    )
    pivot["cosine_minus_mlp_f1"] = (
        pivot["cosine_feature_control"] -
        pivot["matched_classical_mlp"]
    )
    seed_deltas = pivot.reset_index()
    seed_deltas.to_csv(OUTPUT / "per_digit_seed_differences.csv", index=False)

    consistency = (
        seed_deltas.assign(
            cosine_wins=seed_deltas["cosine_minus_mlp_f1"] > 0
        )
        .groupby("digit")["cosine_wins"]
        .sum()
        .rename("seeds_cosine_f1_higher_out_of_3")
        .reset_index()
    )
    consistency.to_csv(OUTPUT / "per_digit_seed_consistency.csv", index=False)

    print("\n=== SEED CONSISTENCY BY DIGIT ===")
    print(consistency.to_string(index=False))

    # Aggregate confusion matrices across the three seeds for each model.
    # Entries are summed counts, not averaged probabilities.
    for model in MODELS:
        matrices = []
        for seed in [42, 123, 2026]:
            path = INPUT / f"confusion_{model}_seed{seed}.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            cm = pd.read_csv(path, index_col=0).to_numpy(dtype=np.int64)
            if cm.shape != (10, 10):
                raise ValueError(f"Unexpected confusion matrix shape in {path}")
            matrices.append(cm)

        total_cm = np.sum(matrices, axis=0)
        pd.DataFrame(
            total_cm,
            index=[f"actual_{i}" for i in range(10)],
            columns=[f"predicted_{i}" for i in range(10)],
        ).to_csv(OUTPUT / f"aggregated_confusion_{model}.csv")

        # Off-diagonal entries with the most mistakes.
        mistakes = []
        for actual in range(10):
            for predicted in range(10):
                if actual != predicted:
                    mistakes.append({
                        "actual_digit": actual,
                        "predicted_digit": predicted,
                        "errors_across_3_seeds": int(total_cm[actual, predicted]),
                    })

        top = pd.DataFrame(mistakes).sort_values(
            "errors_across_3_seeds", ascending=False
        ).head(10)
        top.to_csv(OUTPUT / f"top_confusions_{model}.csv", index=False)

        print(f"\n=== TOP CONFUSIONS: {model} ===")
        print(top.to_string(index=False))

    print(f"\nSaved analysis to: {OUTPUT}")


if __name__ == "__main__":
    main()
