from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xg_vision.config import load_config
from xg_vision.data import ShotFrameDataset, create_split_manifest, find_frames_root, load_manifest, summarize_manifest
from xg_vision.models import build_model, model_input_type, model_kwargs_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight project smoke test.")
    parser.add_argument("--dataset-root", default="Dataset_final")
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    frames_root = find_frames_root(args.dataset_root)
    config["data"]["dataset_root"] = str(frames_root)
    print(f"frames_root={frames_root}")

    with tempfile.TemporaryDirectory() as temp_dir:
        manifest_path = Path(temp_dir) / "splits.csv"
        create_split_manifest(frames_root, manifest_path, seed=42, strategy="group")
        manifest = load_manifest(manifest_path)
        print(summarize_manifest(manifest).to_string(index=False))

        for model_name in ("cnn", "lstm", "attention"):
            config["model"]["name"] = model_name
            input_type = model_input_type(model_name)
            dataset = ShotFrameDataset(
                frames_root=frames_root,
                manifest=manifest,
                split="train",
                model_input=input_type,
                image_size=config["data"]["image_size"],
                sequence_length=config["data"]["sequence_length"],
                include_shot_frame=config["data"]["include_shot_frame"],
                augment=False,
                limit=2,
            )
            loader = DataLoader(dataset, batch_size=2, shuffle=False)
            batch = next(iter(loader))
            model = build_model(model_name, **model_kwargs_from_config(config, model_name))
            with torch.no_grad():
                logits = model(batch["x"])
            assert logits.shape == (2,), f"Unexpected logits shape for {model_name}: {logits.shape}"
            print(f"{model_name}: input={tuple(batch['x'].shape)} logits={tuple(logits.shape)}")

    print("smoke test ok")


if __name__ == "__main__":
    main()
