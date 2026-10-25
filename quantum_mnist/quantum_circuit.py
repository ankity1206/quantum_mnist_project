
import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, Pauli
from qiskit_aer import AerSimulator

from config import CONFIG


def build_feature_circuit(features):
    """Build the original measured feature-encoding circuit."""
    features = np.asarray(features, dtype=float).reshape(-1)

    if len(features) != CONFIG.num_qubits:
        raise ValueError(
            f"Expected {CONFIG.num_qubits} features, got {len(features)}."
        )

    circuit = QuantumCircuit(CONFIG.num_qubits, CONFIG.num_qubits)

    for qubit, angle in enumerate(features):
        circuit.ry(float(angle), qubit)

    for qubit in range(CONFIG.num_qubits - 1):
        circuit.cx(qubit, qubit + 1)

    circuit.measure(range(CONFIG.num_qubits), range(CONFIG.num_qubits))
    return circuit


def simulate_counts(circuit, shots=2048, seed=42):
    """Simulate a measured circuit using local Qiskit Aer."""
    simulator = AerSimulator()

    compiled = transpile(circuit, simulator, seed_transpiler=seed)

    result = simulator.run(
        compiled,
        shots=shots,
        seed_simulator=seed,
    ).result()

    return result.get_counts()


def probabilities_to_z_expectations(counts, num_qubits=4):
    """Calculate Pauli-Z expectations from measurement counts."""
    total_shots = sum(counts.values())
    expectations = np.zeros(num_qubits, dtype=float)

    for bitstring, count in counts.items():
        bitstring = bitstring.replace(" ", "")

        for qubit in range(num_qubits):
            measured_bit = int(bitstring[-(qubit + 1)])
            z_value = 1.0 if measured_bit == 0 else -1.0
            expectations[qubit] += z_value * count / total_shots

    return expectations


def build_variational_circuit(
    num_qubits=None,
    num_layers=None,
    entanglement=True,
):
    """
    Build a parameterized quantum circuit.

    Returns:
        circuit: Parameterized Qiskit circuit
        input_params: Parameters representing input features
        weight_params: Trainable rotation parameters
    """
    if num_qubits is None:
        num_qubits = CONFIG.num_qubits

    if num_layers is None:
        num_layers = CONFIG.quantum_layers

    if num_qubits < 2:
        raise ValueError("This entangling circuit requires at least 2 qubits.")

    if num_layers < 1:
        raise ValueError("num_layers must be at least 1.")

    input_params = ParameterVector("x", num_qubits)
    weight_params = ParameterVector(
        "theta",
        num_qubits * num_layers,
    )

    circuit = QuantumCircuit(num_qubits)

    # Encode the classical input.
    for qubit in range(num_qubits):
        circuit.ry(input_params[qubit], qubit)

    # Trainable variational layers.
    for layer in range(num_layers):

        # Each qubit receives its own adjustable rotation.
        for qubit in range(num_qubits):
            parameter_index = layer * num_qubits + qubit
            circuit.ry(weight_params[parameter_index], qubit)

        # Optional entanglement ablation. When disabled, keep all
        # trainable rotations but omit every CNOT gate.
        if entanglement:
            for qubit in range(num_qubits - 1):
                circuit.cx(qubit, qubit + 1)

            # Close the entanglement chain into a ring.
            circuit.cx(num_qubits - 1, 0)

    return circuit, input_params, weight_params


def exact_expectations(
    features,
    weights,
    circuit,
    input_params,
    weight_params,
):
    """
    Calculate exact Pauli-Z expectation values with a statevector.

    No finite-shot sampling is used. This function is useful for
    testing the circuit before implementing training gradients.
    """
    features = np.asarray(features, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)

    if len(features) != len(input_params):
        raise ValueError(
            f"Expected {len(input_params)} input features, "
            f"got {len(features)}."
        )

    if len(weights) != len(weight_params):
        raise ValueError(
            f"Expected {len(weight_params)} trainable weights, "
            f"got {len(weights)}."
        )

    bindings = {
        parameter: float(value)
        for parameter, value in zip(input_params, features)
    }

    bindings.update({
        parameter: float(value)
        for parameter, value in zip(weight_params, weights)
    })

    bound_circuit = circuit.assign_parameters(bindings)

    state = Statevector.from_instruction(bound_circuit)

    expectations = []

    for qubit in range(len(input_params)):
        # Qiskit Pauli labels place qubit 0 at the rightmost character.
        label = (
            "I" * (len(input_params) - qubit - 1)
            + "Z"
            + "I" * qubit
        )

        value = state.expectation_value(Pauli(label))
        expectations.append(float(np.real(value)))

    return np.asarray(expectations)


if __name__ == "__main__":
    example_features = np.array([0.4930, 1.7233, 1.9382, 0.9886])

    # Test the original shot-based circuit.
    measured_circuit = build_feature_circuit(example_features)
    counts = simulate_counts(measured_circuit)

    print("--- Original circuit ---")
    print("Measurement counts:", counts)
    print(
        "Shot-based Z expectations:",
        np.round(probabilities_to_z_expectations(counts), 4),
    )

    # Test the new variational circuit.
    circuit, input_params, weight_params = build_variational_circuit()

    rng = np.random.default_rng(CONFIG.random_seed)
    weights = rng.uniform(
        -0.1,
        0.1,
        size=len(weight_params),
    )

    outputs = exact_expectations(
        example_features,
        weights,
        circuit,
        input_params,
        weight_params,
    )

    print("\n--- Variational circuit ---")
    print(circuit.draw(output="text"))
    print("Number of input parameters:", len(input_params))
    print("Number of trainable parameters:", len(weight_params))
    print("Trainable weights:", np.round(weights, 4))
    print("Exact Z expectations:", np.round(outputs, 4))

    assert outputs.shape == (CONFIG.num_qubits,)
    assert np.all(outputs >= -1.000001)
    assert np.all(outputs <= 1.000001)

    print("\nAll variational circuit checks passed.")
