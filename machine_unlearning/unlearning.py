from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader

from .train import TrainResult, evaluate, load_state, train_model


@dataclass
class UnlearningResult:
    strategy: str
    seconds: float
    metrics: Dict[str, float]
    model_state: Dict[str, torch.Tensor]


def full_retrain(
    model_factory,
    retained_train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    device: torch.device,
    epochs: int = 30,
    lr: float = 0.01,
) -> UnlearningResult:
    model = model_factory()
    result = train_model(model, retained_train_loader, test_loader, device=device, epochs=epochs, lr=lr)
    metrics = evaluate(load_state(model_factory(), result.model_state, device), test_loader, device)
    return UnlearningResult(
        strategy="full_retrain",
        seconds=result.seconds,
        metrics=metrics,
        model_state=result.model_state,
    )


def approximate_unlearning(
    model_factory,
    base_state: Dict[str, torch.Tensor],
    retained_train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    device: torch.device,
    removed_train_loader: DataLoader | None = None,
    fine_tune_epochs: int = 5,
    lr: float = 0.002,
    forget_weight: float = 0.2,
) -> UnlearningResult:
    model = load_state(model_factory(), base_state, device)
    start = perf_counter()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    for _ in range(fine_tune_epochs):
        model.train()
        for features, labels in retained_train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        if removed_train_loader is not None:
            for features, labels in removed_train_loader:
                features = features.to(device)
                labels = (1 - labels).to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(features)
                loss = forget_weight * criterion(logits, labels)
                loss.backward()
                optimizer.step()
    seconds = perf_counter() - start
    metrics = evaluate(model, test_loader, device)
    return UnlearningResult(
        strategy="approximate_unlearning",
        seconds=seconds,
        metrics=metrics,
        model_state={key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
    )
