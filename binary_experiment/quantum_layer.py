
import numpy as np
import torch
from torch import nn

from config import CONFIG
from quantum_circuit import (
    build_variational_circuit,
    exact_expectations,
)


class _QuantumFunction(torch.autograd.Function):
    """PyTorch autograd bridge for a Qiskit variational circuit."""

    @staticmethod
    def forward(ctx, inputs, weights, circuit, input_params, weight_params):
        if inputs.ndim != 2:
            raise ValueError("inputs must have shape (batch_size, num_qubits).")

        if inputs.shape[1] != len(input_params):
            raise ValueError(
                f"Expected {len(input_params)} input features, "
                f"got {inputs.shape[1]}."
            )

        if weights.ndim != 1 or weights.numel() != len(weight_params):
            raise ValueError(
                f"Expected {len(weight_params)} trainable weights."
            )

        # Qiskit simulation runs on CPU in this implementation.
        input_array = inputs.detach().cpu().numpy()
        weight_array = weights.detach().cpu().numpy()

        outputs = []

        for sample in input_array:
            values = exact_expectations(
                sample,
                weight_array,
                circuit,
                input_params,
                weight_params,
            )
            outputs.append(values)

        ctx.save_for_backward(inputs, weights)
        ctx.circuit = circuit
        ctx.input_params = input_params
        ctx.weight_params = weight_params

        return torch.tensor(
            np.asarray(outputs),
            dtype=inputs.dtype,
            device=inputs.device,
        )

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weights = ctx.saved_tensors

        input_array = inputs.detach().cpu().numpy()
        weight_array = weights.detach().cpu().numpy()
        grad_array = grad_output.detach().cpu().numpy()

        num_weights = len(ctx.weight_params)
        grad_weights = np.zeros(num_weights, dtype=np.float64)

        shift = np.pi / 2.0

        # Parameter-shift rule:
        # d f / d theta = [f(theta + pi/2) - f(theta - pi/2)] / 2
        #
        # Accumulate each sample's contribution to the batch gradient.
        for sample_index, sample in enumerate(input_array):
            upstream_gradient = grad_array[sample_index]

            for weight_index in range(num_weights):
                plus_weights = weight_array.copy()
                minus_weights = weight_array.copy()

                plus_weights[weight_index] += shift
                minus_weights[weight_index] -= shift

                plus_output = exact_expectations(
                    sample,
                    plus_weights,
                    ctx.circuit,
                    ctx.input_params,
                    ctx.weight_params,
                )

                minus_output = exact_expectations(
                    sample,
                    minus_weights,
                    ctx.circuit,
                    ctx.input_params,
                    ctx.weight_params,
                )

                derivative = (plus_output - minus_output) / 2.0

                # Chain rule: sum over the circuit's output features.
                grad_weights[weight_index] += np.dot(
                    upstream_gradient,
                    derivative,
                )

        return (
            None,  # Input features are fixed, so no input gradient needed.
            torch.tensor(
                grad_weights,
                dtype=weights.dtype,
                device=weights.device,
            ),
            None,  # Circuit object
            None,  # Input parameters
            None,  # Weight parameters
        )


class QuantumLayer(nn.Module):
    """Trainable quantum feature layer backed by local Qiskit simulation."""

    def __init__(
        self,
        num_qubits=None,
        num_layers=None,
    ):
        super().__init__()

        self.num_qubits = (
            CONFIG.num_qubits if num_qubits is None else num_qubits
        )
        self.num_layers = (
            CONFIG.quantum_layers if num_layers is None else num_layers
        )

        (
            self.circuit,
            self.input_params,
            self.weight_params,
        ) = build_variational_circuit(
            num_qubits=self.num_qubits,
            num_layers=self.num_layers,
        )

        # Register weights with PyTorch so optimizers can update them.
        self.weights = nn.Parameter(
            0.1 * torch.randn(
                len(self.weight_params),
                dtype=torch.float32,
            )
        )

    def forward(self, inputs):
        return _QuantumFunction.apply(
            inputs,
            self.weights,
            self.circuit,
            self.input_params,
            self.weight_params,
        )


if __name__ == "__main__":
    torch.manual_seed(CONFIG.random_seed)

    layer = QuantumLayer()

    # Two samples are enough for a small gradient smoke test.
    sample_inputs = torch.tensor(
        [
            [0.4930, 1.7233, 1.9382, 0.9886],
            [0.7, 1.2, 2.0, 0.5],
        ],
        dtype=torch.float32,
    )

    print("Input shape:", sample_inputs.shape)

    outputs = layer(sample_inputs)
    print("Output shape:", outputs.shape)
    print("Quantum outputs:\n", outputs.detach())

    # A simple differentiable test loss.
    loss = outputs.square().mean()
    loss.backward()

    print("\nTest loss:", loss.item())
    print("Weight gradient:", layer.weights.grad)
    print("Gradient shape:", layer.weights.grad.shape)

    assert outputs.shape == (2, CONFIG.num_qubits)
    assert layer.weights.grad is not None
    assert torch.isfinite(layer.weights.grad).all()

    print("\nQuantum layer forward/backward test passed.")
