from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import ShotFrameDataset, find_frames_root, load_manifest
from .engine import load_checkpoint, predict_batches, resolve_device, save_json
from .models import build_model, model_input_type


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained xG checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt or last.pt.")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--splits-file", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    config = checkpoint["config"]
    model_name = checkpoint["model_name"]

    frames_root = find_frames_root(args.dataset_root or config["data"]["dataset_root"])
    splits_file = args.splits_file or config["data"]["splits_file"]
    df = load_manifest(splits_file)
    dataset = ShotFrameDataset(
        frames_root=frames_root,
        manifest=df,
        split=args.split,
        model_input=model_input_type(model_name),
        image_size=config["data"].get("image_size", (224, 398)),
        sequence_length=int(config["data"].get("sequence_length", 6)),
        include_shot_frame=bool(config["data"].get("include_shot_frame", True)),
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size or int(config["training"].get("batch_size", 16)),
        shuffle=False,
        num_workers=args.num_workers if args.num_workers is not None else int(config["training"].get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )

    model = build_model(model_name, **checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    metrics, predictions = predict_batches(
        model,
        loader,
        device,
        criterion=None,
        threshold=float(checkpoint.get("threshold", 0.5)),
    )

    out_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(metrics, out_dir / f"{args.split}_metrics.json")
    predictions.to_csv(out_dir / f"{args.split}_predictions_eval.csv", index=False)
    print(pd.Series(metrics).to_string())


if __name__ == "__main__":
    main()
