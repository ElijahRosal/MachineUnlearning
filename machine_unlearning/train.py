from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class TrainResult:
    model_state: Dict[str, torch.Tensor]
    history: list[dict[str, float]]
    seconds: float


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        logits = model(features)
        loss = criterion(logits, labels)
        predictions = torch.argmax(logits, dim=1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)
    return {
        "loss": total_loss / max(1, total_samples),
        "accuracy": total_correct / max(1, total_samples),
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    device: torch.device,
    epochs: int = 30,
    lr: float = 0.01,
) -> TrainResult:
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    start = perf_counter()
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_samples = 0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            predictions = torch.argmax(logits, dim=1)
            running_loss += loss.item() * labels.size(0)
            running_correct += (predictions == labels).sum().item()
            running_samples += labels.size(0)

        train_metrics = {
            "epoch": float(epoch),
            "train_loss": running_loss / max(1, running_samples),
            "train_accuracy": running_correct / max(1, running_samples),
        }
        test_metrics = evaluate(model, test_loader, device)
        train_metrics.update({f"test_{key}": value for key, value in test_metrics.items()})
        history.append(train_metrics)

    seconds = perf_counter() - start
    return TrainResult(model_state={key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, history=history, seconds=seconds)


def load_state(model: nn.Module, state: Dict[str, torch.Tensor], device: torch.device) -> nn.Module:
    model.load_state_dict(state)
    return model.to(device)
