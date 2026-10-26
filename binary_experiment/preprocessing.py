
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from config import CONFIG, Config
from data_loader import load_mnist


def fit_preprocessor(
    X_train: torch.Tensor,
    X_test: torch.Tensor,
    n_components: int = 4,
):
    """
    Fit PCA and an angle scaler using training data only.

    Returns:
        X_train_features: Tensor of shape (N_train, n_components)
        X_test_features:  Tensor of shape (N_test, n_components)
        pca: Fitted PCA transformer
        scaler: Fitted angle scaler
    """
    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError("Expected flattened images with shape (N, features).")

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Training and test feature dimensions must match.")

    if not 1 <= n_components <= min(X_train.shape):
        raise ValueError(
            f"n_components must be between 1 and "
            f"{min(X_train.shape)}."
        )

    # sklearn operates on CPU NumPy arrays.
    train_np = X_train.detach().cpu().numpy()
    test_np = X_test.detach().cpu().numpy()

    # Step 1: Reduce 784 pixels to n_components features.
    pca = PCA(
        n_components=n_components,
        svd_solver="full",
    )

    train_pca = pca.fit_transform(train_np)
    test_pca = pca.transform(test_np)

    # Step 2: Map the PCA features to rotation angles in [0, pi].
    # clip=True keeps transformed test angles inside that range.
    scaler = MinMaxScaler(
        feature_range=(0.0, np.pi),
        clip=True,
    )

    train_angles = scaler.fit_transform(train_pca)
    test_angles = scaler.transform(test_pca)

    # Return CPU tensors; we'll move tensors to the appropriate device
    # later when needed.
    X_train_features = torch.tensor(
        train_angles, dtype=torch.float32
    )
    X_test_features = torch.tensor(
        test_angles, dtype=torch.float32
    )

    return X_train_features, X_test_features, pca, scaler


def preprocess_mnist(config: Config = CONFIG):
    """Load MNIST and prepare PCA features for the quantum circuit."""
    X_train, y_train, X_test, y_test = load_mnist(config)

    X_train_features, X_test_features, pca, scaler = fit_preprocessor(
        X_train,
        X_test,
        n_components=config.num_qubits,
    )

    return (
        X_train_features,
        y_train,
        X_test_features,
        y_test,
        pca,
        scaler,
    )


if __name__ == "__main__":
    (
        X_train,
        y_train,
        X_test,
        y_test,
        pca,
        scaler,
    ) = preprocess_mnist()

    print("\n--- Preprocessing results ---")
    print("Training features:", X_train.shape)
    print("Test features:", X_test.shape)
    print("Training labels:", y_train.shape)
    print("Test labels:", y_test.shape)

    print(
        "Explained variance ratio:",
        np.round(pca.explained_variance_ratio_, 4),
    )
    print(
        "Total variance retained:",
        round(float(pca.explained_variance_ratio_.sum()), 4),
    )

    print(
        "Training angle range:",
        round(X_train.min().item(), 4),
        "to",
        round(X_train.max().item(), 4),
    )
    print(
        "Test angle range:",
        round(X_test.min().item(), 4),
        "to",
        round(X_test.max().item(), 4),
    )

    print("\nFirst example's four angles:")
    print(X_train[0])

    print("\nFirst example's label:", y_train[0].item())
