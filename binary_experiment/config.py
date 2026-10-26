
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # Project paths
    project_dir: Path = Path(__file__).resolve().parent
    data_dir: Path = Path(__file__).resolve().parent / "data"
    output_dir: Path = Path(__file__).resolve().parent / "outputs"

    # Dataset
    random_seed: int = 42
    num_classes: int = 2
    train_samples_per_class: int = 200
    test_samples_per_class: int = 100

    # Quantum model
    num_qubits: int = 4
    quantum_layers: int = 2

    # Training
    batch_size: int = 16
    epochs: int = 10
    learning_rate: float = 0.01

    # Runtime
    device: str = "cuda"


CONFIG = Config()


def prepare_directories(config: Config = CONFIG) -> None:
    """Create project directories if they do not exist."""
    config.data_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    prepare_directories()

    print("Project directory:", CONFIG.project_dir)
    print("Data directory:", CONFIG.data_dir)
    print("Output directory:", CONFIG.output_dir)
    print("Configured qubits:", CONFIG.num_qubits)
    print("Configured device:", CONFIG.device)
