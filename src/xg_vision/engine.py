from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm.auto import tqdm

from .metrics import binary_metrics


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def resolve_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(data: dict[str, Any], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def load_checkpoint(path: str | Path, map_location: torch.device | str = "cpu") -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    max_grad_norm: float | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    amp_enabled = scaler is not None and scaler.is_enabled()

    for batch in tqdm(loader, desc="train", leave=False):
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True).float()
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(x)
            loss = criterion(logits, y)

        if amp_enabled and scaler is not None:
            scaler.scale(loss).backward()
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        batch_size = y.numel()
        total_loss += float(loss.detach().cpu()) * batch_size
        total_items += batch_size
    return total_loss / max(total_items, 1)


@torch.no_grad()
def predict_batches(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
    threshold: float = 0.5,
) -> tuple[dict[str, Any], pd.DataFrame]:
    model.eval()
    y_true: list[float] = []
    y_prob: list[float] = []
    sample_ids: list[str] = []
    shot_dirs: list[str] = []
    total_loss = 0.0
    total_items = 0

    for batch in tqdm(loader, desc="eval", leave=False):
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True).float()
        logits = model(x)
        prob = torch.sigmoid(logits)

        if criterion is not None:
            loss = criterion(logits, y)
            total_loss += float(loss.detach().cpu()) * y.numel()
            total_items += y.numel()

        y_true.extend(y.detach().cpu().numpy().tolist())
        y_prob.extend(prob.detach().cpu().numpy().tolist())
        sample_ids.extend([str(item) for item in batch["sample_id"]])
        shot_dirs.extend([str(item) for item in batch["shot_dir"]])

    metrics = binary_metrics(y_true, y_prob, threshold=threshold)
    if criterion is not None:
        metrics["loss"] = total_loss / max(total_items, 1)

    predictions = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "shot_dir": shot_dirs,
            "label": [int(value) for value in y_true],
            "xg": y_prob,
            "prediction": [int(value >= threshold) for value in y_prob],
        }
    )
    return metrics, predictions
