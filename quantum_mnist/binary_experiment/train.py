
import random
import time

import numpy as np
import torch
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


def calculate_accuracy(logits, labels):
    """Calculate classification accuracy as a fraction."""
    predictions = torch.argmax(logits, dim=1)
    return (predictions == labels).float().mean().item()


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train a model for one epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device).long()

        optimizer.zero_grad()

        logits = model(features)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (
            (torch.argmax(logits.detach(), dim=1) == labels)
            .sum()
            .item()
        )
        total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss, accuracy


def evaluate_model(model, loader, criterion, device):
    """Evaluate a model without updating its parameters."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device).long()

            logits = model(features)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (
                (torch.argmax(logits, dim=1) == labels)
                .sum()
                .item()
            )
            total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss, accuracy


def train_model(
    model,
    model_name,
    train_loader,
    test_loader,
    device,
    epochs,
):
    """Train a model and record its train/test metrics."""
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG.learning_rate,
    )

    history = []

    print(f"\n{'=' * 55}")
    print(f"Training: {model_name}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"{'=' * 55}")

    start_time = time.perf_counter()

    for epoch in range(epochs):
        epoch_start = time.perf_counter()

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        test_loss, test_accuracy = evaluate_model(
            model,
            test_loader,
            criterion,
            device,
        )

        elapsed = time.perf_counter() - epoch_start

        metrics = {
            "model": model_name,
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "epoch_seconds": elapsed,
        }
        history.append(metrics)

        print(
            f"Epoch {epoch + 1:02d}/{epochs:02d} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train accuracy: {train_accuracy:.2%} | "
            f"Test loss: {test_loss:.4f} | "
            f"Test accuracy: {test_accuracy:.2%} | "
            f"Time: {elapsed:.1f}s"
        )

    total_time = time.perf_counter() - start_time
    print(f"Total training time: {total_time:.1f}s")

    return model, history


def main():
    set_seed(CONFIG.random_seed)
    prepare_directories()

    # The quantum circuit simulator runs on the CPU in this implementation.
    # The classical baseline can use CUDA if it is available.
    quantum_device = torch.device("cpu")
    classical_device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Loading and preprocessing MNIST...")
    X_train, y_train, X_test, y_test, pca, scaler = (
        preprocess_mnist(CONFIG)
    )

    print("\nDataset summary")
    print(f"Training examples: {len(X_train)}")
    print(f"Test examples: {len(X_test)}")
    print(f"Features per example: {X_train.shape[1]}")
    print(f"Classes: {CONFIG.num_classes}")
    print(f"Quantum simulator device: {quantum_device}")
    print(f"Classical model device: {classical_device}")

    train_dataset = TensorDataset(X_train.float(), y_train.long())
    test_dataset = TensorDataset(X_test.float(), y_test.long())

    generator = torch.Generator().manual_seed(CONFIG.random_seed)

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

    # Start with a single epoch to check the runtime.
    # Increase CONFIG.epochs after the first run completes successfully.
    epochs = CONFIG.epochs

    hybrid_model = HybridQuantumClassifier()
    hybrid_model, hybrid_history = train_model(
        model=hybrid_model,
        model_name="Hybrid Quantum",
        train_loader=train_loader,
        test_loader=test_loader,
        device=quantum_device,
        epochs=epochs,
    )

    classical_model = ClassicalBaseline()
    classical_model, classical_history = train_model(
        model=classical_model,
        model_name="Classical Baseline",
        train_loader=train_loader,
        test_loader=test_loader,
        device=classical_device,
        epochs=epochs,
    )

    # Save model parameters for later evaluation.
    torch.save(
        hybrid_model.cpu().state_dict(),
        CONFIG.output_dir / "hybrid_model.pt",
    )
    torch.save(
        classical_model.cpu().state_dict(),
        CONFIG.output_dir / "classical_model.pt",
    )

    # Save training history as a CSV file.
    import csv

    history = hybrid_history + classical_history
    history_path = CONFIG.output_dir / "training_history.csv"

    with open(history_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(history[0].keys()),
        )
        writer.writeheader()
        writer.writerows(history)

    print(f"\n{'=' * 55}")
    print("FINAL TEST RESULTS")
    print(f"{'=' * 55}")

    print(
        f"Hybrid Quantum test accuracy: "
        f"{hybrid_history[-1]['test_accuracy']:.2%}"
    )
    print(
        f"Classical Baseline test accuracy: "
        f"{classical_history[-1]['test_accuracy']:.2%}"
    )
    print(f"\nSaved files to: {CONFIG.output_dir}")
    print("- hybrid_model.pt")
    print("- classical_model.pt")
    print("- training_history.csv")


if __name__ == "__main__":
    main()
