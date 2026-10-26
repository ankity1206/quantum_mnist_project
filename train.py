
import csv
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import CONFIG, prepare_directories
from preprocessing import preprocess_mnist
from models import HybridQuantumClassifier, ClassicalBaseline


def set_seed(seed):
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train a model for one epoch and return loss and accuracy."""
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(inputs)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size

        predictions = torch.argmax(logits, dim=1)
        total_correct += (predictions == labels).sum().item()
        total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss, accuracy


def evaluate_model(model, loader, criterion, device):
    """Evaluate a model and return metrics and individual predictions."""
    model.eval()

    total_loss = 0.0
    total_samples = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            predictions = torch.argmax(logits, dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    average_loss = total_loss / total_samples
    accuracy = (
        np.array(all_labels) == np.array(all_predictions)
    ).mean()

    return (
        average_loss,
        float(accuracy),
        all_labels,
        all_predictions,
    )


def save_history(history, output_path):
    """Save training metrics to a CSV file."""
    if not history:
        return

    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(history[0].keys()),
        )
        writer.writeheader()
        writer.writerows(history)


def save_classification_report(
    true_labels,
    predictions,
    model_name,
    output_dir,
):
    """Save per-class precision, recall, F1 and support."""
    class_labels = list(range(CONFIG.num_classes))
    target_names = [str(i) for i in class_labels]

    report = classification_report(
        true_labels,
        predictions,
        labels=class_labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    output_path = output_dir / f"{model_name}_classification_report.csv"

    with open(output_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["class", "precision", "recall", "f1-score", "support"]
        )

        for name, metrics in report.items():
            if isinstance(metrics, dict):
                writer.writerow([
                    name,
                    metrics.get("precision", ""),
                    metrics.get("recall", ""),
                    metrics.get("f1-score", ""),
                    metrics.get("support", ""),
                ])
            else:
                writer.writerow([name, "", metrics, ""])

    print(f"Classification report saved: {output_path}")


def save_confusion_matrix(
    true_labels,
    predictions,
    model_name,
    output_dir,
):
    """Save the confusion matrix as CSV and PNG."""
    class_labels = list(range(CONFIG.num_classes))

    matrix = confusion_matrix(
        true_labels,
        predictions,
        labels=class_labels,
    )

    csv_path = output_dir / f"{model_name}_confusion_matrix.csv"
    np.savetxt(csv_path, matrix, delimiter=",", fmt="%d")

    figure, axis = plt.subplots(figsize=(8, 7))

    image = axis.imshow(matrix, interpolation="nearest")
    figure.colorbar(image, ax=axis)

    axis.set(
        title=f"{model_name.replace('_', ' ').title()} Confusion Matrix",
        xlabel="Predicted digit",
        ylabel="Actual digit",
        xticks=class_labels,
        yticks=class_labels,
    )

    threshold = matrix.max() / 2 if matrix.size else 0

    for row in range(CONFIG.num_classes):
        for col in range(CONFIG.num_classes):
            axis.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
                color="white" if matrix[row, col] > threshold else "black",
                fontsize=8,
            )

    figure.tight_layout()

    png_path = output_dir / f"{model_name}_confusion_matrix.png"
    figure.savefig(png_path, dpi=150)
    plt.close(figure)

    print(f"Confusion matrix CSV saved: {csv_path}")
    print(f"Confusion matrix image saved: {png_path}")


def save_training_plot(history, model_name, output_dir):
    """Plot training and test loss and accuracy over epochs."""
    epochs = [item["epoch"] for item in history]
    train_losses = [item["train_loss"] for item in history]
    test_losses = [item["test_loss"] for item in history]
    train_accuracies = [item["train_accuracy"] for item in history]
    test_accuracies = [item["test_accuracy"] for item in history]

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_losses, marker="o", label="Train loss")
    axes[0].plot(epochs, test_losses, marker="o", label="Test loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        epochs, train_accuracies, marker="o", label="Train accuracy"
    )
    axes[1].plot(
        epochs, test_accuracies, marker="o", label="Test accuracy"
    )
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    figure.suptitle(model_name.replace("_", " ").title())
    figure.tight_layout()

    output_path = output_dir / f"{model_name}_training_curves.png"
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

    print(f"Training curves saved: {output_path}")


def train_model(model, model_name, train_loader, test_loader, device, class_weights=None):
    """Train, evaluate and save one model."""
    print(f"\n{'=' * 60}")
    print(f"Training {model_name.replace('_', ' ').title()}")
    print(f"Device: {device}")
    print(f"{'=' * 60}")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG.learning_rate,
    )

    history = []

    for epoch in range(1, CONFIG.epochs + 1):
        epoch_start = time.time()

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        test_loss, test_accuracy, _, _ = evaluate_model(
            model,
            test_loader,
            criterion,
            device,
        )

        epoch_seconds = time.time() - epoch_start

        metrics = {
            "model": model_name,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "epoch_seconds": epoch_seconds,
        }
        history.append(metrics)

        print(
            f"Epoch {epoch:02d}/{CONFIG.epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train accuracy: {train_accuracy:.2%} | "
            f"Test loss: {test_loss:.4f} | "
            f"Test accuracy: {test_accuracy:.2%} | "
            f"Time: {epoch_seconds:.2f}s"
        )

    # Save trained model weights.
    model_path = CONFIG.output_dir / f"{model_name}_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\nModel weights saved: {model_path}")

    # Save training history and curves.
    history_path = CONFIG.output_dir / f"{model_name}_history.csv"
    save_history(history, history_path)
    save_training_plot(history, model_name, CONFIG.output_dir)

    # Final evaluation on the test set.
    test_loss, test_accuracy, true_labels, predictions = evaluate_model(
        model,
        test_loader,
        criterion,
        device,
    )

    print(f"\nFinal test results for {model_name}:")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.2%}")

    save_classification_report(
        true_labels,
        predictions,
        model_name,
        CONFIG.output_dir,
    )

    save_confusion_matrix(
        true_labels,
        predictions,
        model_name,
        CONFIG.output_dir,
    )

    return {
        "model": model_name,
        "device": str(device),
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "total_training_seconds": sum(
            item["epoch_seconds"] for item in history
        ),
        "model_path": str(model_path),
    }


def main():
    set_seed(CONFIG.random_seed)
    prepare_directories()

    print("Loading and preprocessing MNIST...")
    (
        X_train,
        y_train,
        X_test,
        y_test,
        _,
        _,
    ) = preprocess_mnist()

    print("Training feature shape:", X_train.shape)
    print("Training label shape:", y_train.shape)
    print("Test feature shape:", X_test.shape)
    print("Test label shape:", y_test.shape)

    # Verify that labels are compatible with ten-class classification.
    expected_classes = set(range(CONFIG.num_classes))
    actual_classes = set(y_train.tolist())

    if actual_classes != expected_classes:
        raise ValueError(
            f"Expected training labels {expected_classes}, "
            f"but received {actual_classes}"
        )

    train_dataset = TensorDataset(X_train.float(), y_train.long())
    test_dataset = TensorDataset(X_test.float(), y_test.long())

    # Use a seeded generator so shuffled batches are reproducible.
    generator = torch.Generator()
    generator.manual_seed(CONFIG.random_seed)

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

    # Quantum circuit simulation runs on CPU.
    quantum_device = torch.device("cpu")

    # The classical baseline can use CUDA when available.
    classical_device = torch.device(
        "cuda" if CONFIG.device == "cuda" and torch.cuda.is_available()
        else "cpu"
    )

    print("\nExperiment configuration:")
    print("Number of classes:", CONFIG.num_classes)
    print("Quantum qubits:", CONFIG.num_qubits)
    print("Quantum layers:", CONFIG.quantum_layers)
    print("Batch size:", CONFIG.batch_size)
    print("Epochs:", CONFIG.epochs)
    print("Learning rate:", CONFIG.learning_rate)
    print("Quantum model device:", quantum_device)
    print("Classical model device:", classical_device)
    print("Output directory:", CONFIG.output_dir)

    # Train the hybrid quantum model.
    hybrid_model = HybridQuantumClassifier()
    hybrid_results = train_model(
        hybrid_model,
        "hybrid_quantum",
        train_loader,
        test_loader,
        quantum_device,
    )

    # Reset the seed before initializing the baseline.
    set_seed(CONFIG.random_seed)
    classical_model = ClassicalBaseline()
    classical_results = train_model(
        classical_model,
        "classical_baseline",
        train_loader,
        test_loader,
        classical_device,
    )

    # Save a concise comparison of both models.
    summary_path = CONFIG.output_dir / "model_comparison.csv"
    results = [hybrid_results, classical_results]

    with open(summary_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'=' * 60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'=' * 60}")
    print(f"Hybrid quantum test accuracy: "
          f"{hybrid_results['test_accuracy']:.2%}")
    print(f"Classical baseline test accuracy: "
          f"{classical_results['test_accuracy']:.2%}")
    print(f"Comparison saved: {summary_path}")
    print(f"All multiclass results are in: {CONFIG.output_dir}")


if __name__ == "__main__":
    main()
