from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import load_config, save_config
from .data import (
    ShotFrameDataset,
    create_split_manifest,
    find_frames_root,
    load_manifest,
    summarize_manifest,
)
from .engine import load_checkpoint, predict_batches, resolve_device, save_json, seed_everything, train_one_epoch
from .metrics import best_threshold, binary_metrics, metric_for_selection
from .models import build_model, canonical_model_name, model_input_type, model_kwargs_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train visual xG models.")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config path.")
    parser.add_argument("--model", choices=["cnn", "lstm", "attention", "cnn_lstm", "cnn_attention"], default=None)
    parser.add_argument("--dataset-root", default=None, help="Frames_Bons_Definitius or an outer dataset folder.")
    parser.add_argument("--splits-file", default=None, help="Existing or target split manifest CSV.")
    parser.add_argument("--output-dir", default=None, help="Run directory. Defaults to runs/<model>_<timestamp>.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, help="Example: cuda, cuda:0, cpu.")
    parser.add_argument("--backbone", choices=["auto", "custom", "resnet18"], default=None)
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet weights for torchvision backbones.")
    parser.add_argument("--fine-tune-backbone", action="store_true", help="Train the full pretrained backbone.")
    parser.add_argument("--threshold", default=None, help="Classification threshold or 'auto'.")
    parser.add_argument("--selection-metric", default=None, help="Metric for best checkpoint: average_precision, auc_roc, f1.")
    parser.add_argument("--no-augment", action="store_true", help="Disable train augmentations.")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision.")
    parser.add_argument("--limit-samples", type=int, default=None, help="Small debug limit per split.")
    return parser.parse_args()


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.model is not None:
        config["model"]["name"] = canonical_model_name(args.model)
    if args.dataset_root is not None:
        config["data"]["dataset_root"] = args.dataset_root
    if args.splits_file is not None:
        config["data"]["splits_file"] = args.splits_file
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    if args.num_workers is not None:
        config["training"]["num_workers"] = args.num_workers
    if args.seed is not None:
        config["data"]["seed"] = args.seed
    if args.backbone is not None:
        config["model"]["backbone"] = args.backbone
    if args.no_pretrained:
        config["model"]["pretrained"] = False
    if args.fine_tune_backbone:
        config["model"]["freeze_backbone"] = False
    if args.threshold is not None:
        config["training"]["threshold"] = args.threshold
    if args.selection_metric is not None:
        config["training"]["selection_metric"] = args.selection_metric
    if args.no_augment:
        config["training"]["augment"] = False
    if args.no_amp:
        config["training"]["use_amp"] = False
    return config


def make_loader(
    dataset: ShotFrameDataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def positive_weight(labels: list[int], strategy: str = "sqrt", scale: float = 1.0) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0:
        return 1.0
    ratio = negatives / positives
    if strategy == "none":
        weight = 1.0
    elif strategy == "linear":
        weight = ratio
    elif strategy == "sqrt":
        weight = ratio**0.5
    else:
        raise ValueError("pos_weight_strategy must be one of: none, sqrt, linear")
    return float(weight * scale)


def make_scaler(device: torch.device, enabled: bool):
    if device.type != "cuda" or not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=True)


def evaluate_with_threshold(
    model,
    loader,
    device: torch.device,
    criterion,
    threshold_config: str | float,
    threshold_metric: str,
) -> tuple[dict[str, Any], pd.DataFrame, float]:
    base_metrics, predictions = predict_batches(model, loader, device, criterion, threshold=0.5)
    if str(threshold_config).lower() == "auto":
        threshold, metrics = best_threshold(predictions["label"], predictions["xg"], metric=threshold_metric)
    else:
        threshold = float(threshold_config)
        metrics = binary_metrics(predictions["label"], predictions["xg"], threshold=threshold)
    if "loss" in base_metrics:
        metrics["loss"] = base_metrics["loss"]
    predictions["prediction"] = (predictions["xg"] >= threshold).astype(int)
    return metrics, predictions, threshold


def main() -> None:
    args = parse_args()
    config = apply_cli_overrides(load_config(args.config), args)
    model_name = canonical_model_name(config["model"]["name"])
    config["model"]["name"] = model_name

    seed = int(config["data"].get("seed", 42))
    seed_everything(seed)
    device = resolve_device(args.device)

    frames_root = find_frames_root(config["data"].get("dataset_root"))
    config["data"]["dataset_root"] = str(frames_root)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) if args.output_dir else Path(config["output"]["run_dir"]) / f"{model_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    splits_file = config["data"].get("splits_file")
    if splits_file is None:
        splits_path = run_dir / "splits.csv"
    else:
        splits_path = Path(splits_file)
    if not splits_path.exists():
        df = create_split_manifest(
            frames_root,
            splits_path,
            ratios=config["data"].get("split_ratios", (0.70, 0.15, 0.15)),
            seed=seed,
            strategy=config["data"].get("split_strategy", "group"),
        )
    else:
        df = load_manifest(splits_path)
    config["data"]["splits_file"] = str(splits_path.resolve())

    summary = summarize_manifest(df)
    summary.to_csv(run_dir / "split_summary.csv", index=False)
    save_config(config, run_dir / "config.yaml")

    input_type = model_input_type(model_name)
    data_kwargs = {
        "frames_root": frames_root,
        "manifest": df,
        "model_input": input_type,
        "image_size": config["data"].get("image_size", (224, 398)),
        "sequence_length": int(config["data"].get("sequence_length", 6)),
        "include_shot_frame": bool(config["data"].get("include_shot_frame", True)),
        "limit": args.limit_samples,
    }
    train_ds = ShotFrameDataset(split="train", augment=bool(config["training"].get("augment", True)), **data_kwargs)
    val_ds = ShotFrameDataset(split="val", augment=False, **data_kwargs)
    test_ds = ShotFrameDataset(split="test", augment=False, **data_kwargs)

    batch_size = int(config["training"].get("batch_size", 16))
    num_workers = int(config["training"].get("num_workers", 0))
    pin_memory = device.type == "cuda"
    train_loader = make_loader(train_ds, batch_size, num_workers, True, pin_memory)
    val_loader = make_loader(val_ds, batch_size, num_workers, False, pin_memory)
    test_loader = make_loader(test_ds, batch_size, num_workers, False, pin_memory)

    model_kwargs = model_kwargs_from_config(config, model_name)
    model = build_model(model_name, **model_kwargs).to(device)
    model_kwargs["backbone"] = getattr(model, "resolved_backbone", model_kwargs.get("backbone", "custom"))
    pos_weight = positive_weight(
        train_ds.labels,
        strategy=str(config["training"].get("pos_weight_strategy", "sqrt")),
        scale=float(config["training"].get("pos_weight_scale", 1.0)),
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(config["training"].get("learning_rate", 3e-4)),
        weight_decay=float(config["training"].get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    scaler = make_scaler(device, bool(config["training"].get("use_amp", True)))

    print(summary.to_string(index=False))
    print(f"Training {model_name} on {device} -> {run_dir}")
    print(f"Backbone: {model_kwargs.get('backbone', 'custom')}")
    print(f"Train positive weight: {pos_weight:.3f}")

    best_score = float("-inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    threshold_config = config["training"].get("threshold", "auto")
    threshold_metric = str(config["training"].get("threshold_metric", "f1"))
    selection_metric = str(config["training"].get("selection_metric", "average_precision"))
    patience = int(config["training"].get("early_stopping_patience", 8))
    epochs = int(config["training"].get("epochs", 40))
    max_grad_norm = config["training"].get("max_grad_norm", 1.0)
    max_grad_norm = float(max_grad_norm) if max_grad_norm is not None else None

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler=scaler,
            max_grad_norm=max_grad_norm,
        )
        val_metrics, val_predictions, selected_threshold = evaluate_with_threshold(
            model,
            val_loader,
            device,
            criterion,
            threshold_config,
            threshold_metric,
        )
        score = metric_for_selection(val_metrics, selection_metric)
        scheduler.step(score)

        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

        print(
            f"epoch {epoch:03d} | loss {train_loss:.4f} | "
            f"val_ap {val_metrics['average_precision']:.4f} | "
            f"val_auc {val_metrics['auc_roc']:.4f} | val_f1 {val_metrics['f1']:.4f} | "
            f"thr {selected_threshold:.3f}"
        )

        checkpoint = {
            "model_name": model_name,
            "model_kwargs": model_kwargs,
            "state_dict": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "threshold": selected_threshold,
            "val_metrics": val_metrics,
            "split_summary": summary.to_dict("records"),
        }
        torch.save(checkpoint, run_dir / "last.pt")

        if score > best_score:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            torch.save(checkpoint, run_dir / "best.pt")
            val_predictions.to_csv(run_dir / "val_predictions.csv", index=False)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
                break

    best_checkpoint = load_checkpoint(run_dir / "best.pt", map_location=device)
    model.load_state_dict(best_checkpoint["state_dict"])
    test_threshold = float(best_checkpoint.get("threshold", 0.5))
    test_metrics, test_predictions = predict_batches(model, test_loader, device, criterion, test_threshold)
    test_predictions.to_csv(run_dir / "test_predictions.csv", index=False)

    result = {
        "best_epoch": int(best_epoch),
        "best_validation_score": float(best_score),
        "selection_metric": selection_metric,
        "threshold": test_threshold,
        "best_val_metrics": best_checkpoint.get("val_metrics", {}),
        "test_metrics": test_metrics,
    }
    save_json(result, run_dir / "metrics.json")
    print("Test metrics:")
    print(pd.Series(test_metrics).to_string())


if __name__ == "__main__":
    main()
