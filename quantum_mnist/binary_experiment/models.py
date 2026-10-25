
import torch
from torch import nn

from config import CONFIG
from quantum_layer import QuantumLayer


class HybridQuantumClassifier(nn.Module):
    """Quantum feature extractor followed by a classical classifier."""

    def __init__(
        self,
        num_qubits=None,
        num_classes=None,
        num_layers=None,
    ):
        super().__init__()

        num_qubits = (
            CONFIG.num_qubits if num_qubits is None else num_qubits
        )
        num_classes = (
            CONFIG.num_classes if num_classes is None else num_classes
        )
        num_layers = (
            CONFIG.quantum_layers if num_layers is None else num_layers
        )

        self.quantum_layer = QuantumLayer(
            num_qubits=num_qubits,
            num_layers=num_layers,
        )

        self.classifier = nn.Linear(num_qubits, num_classes)

    def forward(self, inputs):
        quantum_features = self.quantum_layer(inputs)
        logits = self.classifier(quantum_features)
        return logits


class ClassicalBaseline(nn.Module):
    """Classical MLP using the same PCA features as the quantum model."""

    def __init__(
        self,
        num_features=None,
        num_classes=None,
        hidden_size=16,
    ):
        super().__init__()

        num_features = (
            CONFIG.num_qubits if num_features is None else num_features
        )
        num_classes = (
            CONFIG.num_classes if num_classes is None else num_classes
        )

        self.network = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, inputs):
        return self.network(inputs)


if __name__ == "__main__":
    torch.manual_seed(CONFIG.random_seed)

    # Test the models with two examples.
    sample_inputs = torch.rand(2, CONFIG.num_qubits)

    hybrid_model = HybridQuantumClassifier()
    classical_model = ClassicalBaseline()

    hybrid_logits = hybrid_model(sample_inputs)
    classical_logits = classical_model(sample_inputs)

    print("Input shape:", sample_inputs.shape)
    print("Hybrid output shape:", hybrid_logits.shape)
    print("Classical output shape:", classical_logits.shape)

    print("\nHybrid logits:")
    print(hybrid_logits.detach())

    print("\nClassical logits:")
    print(classical_logits.detach())

    assert hybrid_logits.shape == (2, CONFIG.num_classes)
    assert classical_logits.shape == (2, CONFIG.num_classes)

    print("\nBoth model forward tests passed.")
