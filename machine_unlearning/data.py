from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class DatasetSplits:
    train: TensorDataset
    test: TensorDataset
    retain_indices: torch.Tensor
    remove_indices: torch.Tensor
    removal_strategy: str


def make_gaussian_mixture(
    *,
    seed: int = 7,
    train_size: int = 1200,
    test_size: int = 400,
    remove_fraction: float = 0.12,
    removal_strategy: Literal["subset", "forget_class"] = "subset",
    remove_class: int = 1,
) -> DatasetSplits:
    generator = torch.Generator().manual_seed(seed)

    centers = torch.tensor([[-2.0, -2.0], [2.0, 2.0]], dtype=torch.float32)
    cov_scale = 0.9

    def sample_split(size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = torch.randint(0, 2, (size,), generator=generator)
        noise = torch.randn(size, 2, generator=generator) * cov_scale
        features = centers[labels] + noise
        return features, labels

    train_features, train_labels = sample_split(train_size)
    test_features, test_labels = sample_split(test_size)

    class_indices = torch.where(train_labels == remove_class)[0]
    if len(class_indices) == 0:
        raise ValueError(f"No training examples found for remove_class={remove_class}.")

    if removal_strategy == "forget_class":
        remove_indices = class_indices
    elif removal_strategy == "subset":
        remove_count = max(1, int(len(class_indices) * remove_fraction))
        shuffled = class_indices[torch.randperm(len(class_indices), generator=generator)]
        remove_indices = shuffled[:remove_count]
    else:
        raise ValueError(f"Unsupported removal_strategy={removal_strategy!r}.")

    retain_mask = torch.ones(train_size, dtype=torch.bool)
    retain_mask[remove_indices] = False
    retain_indices = torch.where(retain_mask)[0]

    train_dataset = TensorDataset(train_features, train_labels)
    test_dataset = TensorDataset(test_features, test_labels)
    return DatasetSplits(
        train=train_dataset,
        test=test_dataset,
        retain_indices=retain_indices,
        remove_indices=remove_indices,
        removal_strategy=removal_strategy,
    )


def make_loader(dataset: TensorDataset, batch_size: int = 64, shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def subset_dataset(dataset: TensorDataset, indices: torch.Tensor) -> TensorDataset:
    features, labels = dataset.tensors
    return TensorDataset(features[indices], labels[indices])
