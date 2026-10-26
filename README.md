# Quantum MNIST Classification

This project compares classical machine learning models with a hybrid quantum-classical model for handwritten digit classification using the MNIST dataset.

The project also evaluates a classical cosine-feature model to investigate whether similar performance can be achieved without quantum simulation.

## Models

- **Classical MLP:** A standard neural network.
- **Hybrid Quantum Classifier:** A variational quantum circuit combined with a classical neural network.
- **Cosine-Feature Control:** A classical model that uses cosine-based features.
- **Parameter-Matched MLP:** A classical neural network with a parameter count similar to the cosine-feature model.

## Main Results

Using all 10,000 official MNIST test images:

| Model | Mean Accuracy |
|---|---:|
| Quantum + Classical  | 69.4% |
| Parameter-matched MLP | 83.82% |
| Cosine-feature model | 86.31% |

The cosine-feature model performed better than the parameter-matched MLP across all three training seeds.

The experiments do **not** establish quantum advantage. The cosine-feature model is classical and reproduces the feature transformation of the tested circuit without entanglement.


## Structure

- `config.py` — Experiment configuration
- `data_loader.py` — MNIST data loading and sampling
- `preprocessing.py` — PCA and feature scaling
- `models.py` — Classical and hybrid model definitions
- `quantum_circuit.py` — Quantum circuit construction
- `quantum_layer.py` — Quantum simulation and gradients
- `experiment*.py` — Experiment scripts
- `outputs/` — Saved metrics, model checkpoints and experiment results

## Final

- Quantum circuit simulation can be much slower than classical inference.
- The reported results are specific to the preprocessing, model configurations, and random seeds used in these experiments.
