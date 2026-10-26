# Quantum MNIST Classification

## Overview

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
| Parameter-matched MLP | 83.82% |
| Cosine-feature model | 86.31% |

The cosine-feature model performed better than the parameter-matched MLP across all three training seeds.

The experiments do **not** establish quantum advantage. The cosine-feature model is classical and reproduces the feature transformation of the tested circuit without entanglement.

## Requirements

- Python 3.10
- PyTorch
- Qiskit
- Qiskit Aer
- torchvision
- NumPy
- pandas
- scikit-learn
- Matplotlib

## Running Experiments

Run the experiment scripts from the project directory. For example:

```bash
python experiment4_40k.py
python experiment5_parameter_match.py
python experiment6_full_test.py
python experiment7_error_analysis.py
python experiment8_paired_test.py
```

These scripts use saved outputs from earlier experiments where required. Run the experiments in order when generating results from scratch.


For correct predictions, uploaded images must be converted to the same format used during training. The application should load the saved model checkpoint and training-fitted PCA and scaler rather than fitting new preprocessing on each image.

## Project Structure

- `config.py` — Experiment configuration
- `data_loader.py` — MNIST data loading and sampling
- `preprocessing.py` — PCA and feature scaling
- `models.py` — Classical and hybrid model definitions
- `quantum_circuit.py` — Quantum circuit construction
- `quantum_layer.py` — Quantum simulation and gradients
- `experiment*.py` — Experiment scripts
- `outputs/` — Saved metrics, model checkpoints, and experiment results

## Notes

- Quantum circuit simulation can be much slower than classical inference.
- The reported results are specific to the preprocessing, model configurations, and random seeds used in these experiments.
