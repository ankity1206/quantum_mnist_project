
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # Project directories
    project_dir: Path = Path(__file__).resolve().parent
    data_dir: Path = project_dir / "data"

    # Separate outputs for this experiment
    output_dir: Path = project_dir / "outputs" / "experiment_8qubits_nonlinear_head"

    # Reproducibility
    random_seed: int = 42

    # Dataset configuration
    num_classes: int = 10
    train_samples_per_class: int = 200
    test_samples_per_class: int = 100

    # Quantum circuit configuration
    num_qubits: int = 8
    quantum_layers: int = 2

    # Training configuration
    batch_size: int = 16
    epochs: int = 10
    learning_rate: float = 0.01

    # Device configuration
    device: str = "cuda"


CONFIG = Config()


def prepare_directories(config: Config = CONFIG) -> None:
    """Create directories required by the project."""
    config.data_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    prepare_directories()

    print("Project directory:", CONFIG.project_dir)
    print("Dataset directory:", CONFIG.data_dir)
    print("Output directory:", CONFIG.output_dir)
    print("Number of classes:", CONFIG.num_classes)
    print("Number of qubits:", CONFIG.num_qubits)
    print("Quantum layers:", CONFIG.quantum_layers)
