from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "data": {
        "dataset_root": "Dataset_final/Dataset_final/Frames_Bons_Definitius",
        "splits_file": None,
        "split_strategy": "group",
        "split_ratios": [0.70, 0.15, 0.15],
        "image_size": [224, 398],
        "sequence_length": 6,
        "include_shot_frame": True,
        "seed": 42,
    },
    "training": {
        "epochs": 40,
        "batch_size": 16,
        "num_workers": 0,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "dropout": 0.25,
        "early_stopping_patience": 8,
        "max_grad_norm": 1.0,
        "use_amp": True,
        "threshold": 0.5,
        "augment": True,
    },
    "model": {
        "name": "cnn",
        "feature_dim": 256,
        "cnn_channels": [32, 64, 128, 256],
        "classifier_hidden_dim": 128,
        "lstm_hidden_dim": 256,
        "lstm_layers": 1,
        "attention_heads": 4,
        "attention_layers": 2,
        "attention_ff_dim": 512,
    },
    "output": {"run_dir": "runs"},
}


def deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively update a config dictionary."""
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    if path is None:
        return config

    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return deep_update(config, loaded)


def save_config(config: Mapping[str, Any], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=False)


def set_by_dotted_key(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor = config
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value
