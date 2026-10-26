import torch
from torch.utils.data import DataLoader, TensorDataset

from config import CONFIG, prepare_directories
from preprocessing import preprocess_mnist
from models import HybridQuantumClassifier
from train import set_seed, train_model


def main():
    # Keep the existing configuration settings, but isolate this run's outputs.
    CONFIG.output_dir = (
        CONFIG.project_dir / "outputs" / "experiment_8qubits_3layers_weighted_loss"
    )

    CONFIG.quantum_layers = 3

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

    # Heuristic weights for an experiment, not inverse-frequency weights.
    # Class indices correspond to digits 0 through 9.
    class_weights = torch.tensor(
        [1.00, 1.00, 1.15, 1.25, 1.00, 1.00, 1.00, 1.00, 1.50, 1.25],
        dtype=torch.float32,
    )

    quantum_device = torch.device("cpu")

    print("\nWeighted-loss hybrid experiment")
    print("Random seed:", CONFIG.random_seed)
    print("Qubits:", CONFIG.num_qubits)
    print("Quantum layers:", CONFIG.quantum_layers)
    print("Training samples:", len(train_dataset))
    print("Test samples:", len(test_dataset))
    print("Epochs:", CONFIG.epochs)
    print("Learning rate:", CONFIG.learning_rate)
    print("Class weights:", class_weights.tolist())
    print("Output directory:", CONFIG.output_dir)

    model = HybridQuantumClassifier()

    results = train_model(
        model,
        "hybrid_quantum_weighted",
        train_loader,
        test_loader,
        quantum_device,
        class_weights=class_weights,
    )

    print("\n" + "=" * 55)
    print("WEIGHTED-LOSS EXPERIMENT COMPLETE")
    print("=" * 55)
    print(f"Test accuracy: {results['test_accuracy']:.2%}")
    print(f"Training time: {results['total_training_seconds']:.1f} seconds")
    print("Model saved to:", results["model_path"])
    print("Results saved in:", CONFIG.output_dir)


if __name__ == "__main__":
    main()
