import torch
from torchvision import datasets, transforms

from config import CONFIG, Config


def load_mnist(config: Config = CONFIG):
    """
    Load MNIST from local files only.

    Returns:
        X_train: float tensor of shape (N_train, 784)
        y_train: integer tensor of shape (N_train,)
        X_test:  float tensor of shape (N_test, 784)
        y_test:  integer tensor of shape (N_test,)
    """
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root=str(config.data_dir),
        train=True,
        download=False,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root=str(config.data_dir),
        train=False,
        download=True,
        transform=transform,
    )

    X_train, y_train = _select_samples(
        train_dataset,
        config.num_classes,
        config.train_samples_per_class,
        config.random_seed,
    )

    X_test, y_test = _select_samples(
        test_dataset,
        config.num_classes,
        config.test_samples_per_class,
        config.random_seed + 1,
    )

    return X_train, y_train, X_test, y_test


def _select_samples(dataset, num_classes, samples_per_class, seed):
    """Select a reproducible, balanced subset of the requested classes."""
    generator = torch.Generator().manual_seed(seed)

    targets = dataset.targets
    selected_indices = []

    for label in range(num_classes):
        class_indices = torch.where(targets == label)[0]

        if len(class_indices) < samples_per_class:
            raise ValueError(
                f"Requested {samples_per_class} samples for digit {label}, "
                f"but only {len(class_indices)} are available."
            )

        permutation = torch.randperm(
            len(class_indices), generator=generator
        )
        chosen = class_indices[permutation[:samples_per_class]]
        selected_indices.append(chosen)

    indices = torch.cat(selected_indices)

    # Shuffle the combined subset so labels aren't grouped by class.
    indices = indices[
        torch.randperm(len(indices), generator=generator)
    ]

    images = []
    labels = []

    for index in indices.tolist():
        image, label = dataset[index]
        images.append(image.reshape(-1))
        labels.append(label)

    X = torch.stack(images).float()
    y = torch.tensor(labels, dtype=torch.long)

    return X, y


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_mnist()

    print("Training images:", X_train.shape)
    print("Training labels:", y_train.shape)
    print("Test images:", X_test.shape)
    print("Test labels:", y_test.shape)
    print("Pixel range:", X_train.min().item(), X_train.max().item())
    print("Training class counts:", torch.bincount(y_train))
    print("Test class counts:", torch.bincount(y_test))
