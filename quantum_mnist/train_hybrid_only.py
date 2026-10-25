import torch
from torch.utils.data import DataLoader, TensorDataset

from config import CONFIG, prepare_directories
from preprocessing import preprocess_mnist
from models import HybridQuantumClassifier
from train import set_seed, train_model


def main():
    set_seed(CONFIG.random_seed)
    prepare_directories()

    print("Loading and preprocessing MNIST...")
    X_train, y_train, X_test, y_test, _, _ = preprocess_mnist()

    expected_classes = set(range(CONFIG.num_classes))
    actual_classes = set(y_train.tolist())

    if actual_classes != expected_classes:
        raise ValueError(
            f"Expected labels {expected_classes}, got {actual_classes}"
        )

    train_dataset = TensorDataset(X_train.float(), y_train.long())
    test_dataset = TensorDataset(X_test.float(), y_test.long())

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

    # Statevector quantum simulation runs on CPU.
    quantum_device = torch.device("cpu")

    print("\nNonlinear-head hybrid experiment")
    print("Qubits:", CONFIG.num_qubits)
    print("Quantum layers:", CONFIG.quantum_layers)
    print("Training samples:", len(train_dataset))
    print("Test samples:", len(test_dataset))
    print("Epochs:", CONFIG.epochs)
    print("Learning rate:", CONFIG.learning_rate)
    print("Output directory:", CONFIG.output_dir)

    model = HybridQuantumClassifier()
    results = train_model(
        model,
        "hybrid_quantum",
        train_loader,
        test_loader,
        quantum_device,
    )

    print("\n" + "=" * 55)
    print("NONLINEAR-HEAD EXPERIMENT COMPLETE")
    print("=" * 55)
    print(f"Test accuracy: {results['test_accuracy']:.2%}")
    print(f"Training time: {results['total_training_seconds']:.1f} seconds")
    print("Model saved to:", results["model_path"])
    print("Results saved in:", CONFIG.output_dir)


if __name__ == "__main__":
    main()
