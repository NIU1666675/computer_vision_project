from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from .data import FrameTransform, SHOT_FRAME, iter_shot_dirs, load_rgb, sequence_frame_names
from .engine import load_checkpoint, resolve_device
from .models import build_model, model_input_type


def load_model(checkpoint_path: str | Path, device: torch.device):
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model = build_model(checkpoint["model_name"], **checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def tensor_from_shot_dir(shot_dir: str | Path, checkpoint: dict) -> torch.Tensor:
    shot_path = Path(shot_dir)
    config = checkpoint["config"]
    model_name = checkpoint["model_name"]
    transform = FrameTransform(config["data"].get("image_size", (224, 398)), augment=False)

    if model_input_type(model_name) == "single":
        return transform(load_rgb(shot_path / SHOT_FRAME))

    names = sequence_frame_names(
        int(config["data"].get("sequence_length", 6)),
        bool(config["data"].get("include_shot_frame", True)),
    )
    frames = [transform(load_rgb(shot_path / name)) for name in names]
    return torch.stack(frames, dim=0)


@torch.no_grad()
def predict_shot_dirs(
    checkpoint_path: str | Path,
    shot_dirs: list[str | Path],
    device_arg: str | None = None,
) -> pd.DataFrame:
    device = resolve_device(device_arg)
    model, checkpoint = load_model(checkpoint_path, device)
    threshold = float(checkpoint.get("threshold", 0.5))
    rows = []
    for shot_dir in shot_dirs:
        x = tensor_from_shot_dir(shot_dir, checkpoint).unsqueeze(0).to(device)
        prob = float(torch.sigmoid(model(x)).detach().cpu().item())
        label_path = Path(shot_dir) / "label.txt"
        label = int(label_path.read_text(encoding="utf-8").strip()) if label_path.exists() else None
        rows.append(
            {
                "sample_id": Path(shot_dir).name,
                "shot_dir": str(Path(shot_dir).resolve()),
                "label": label,
                "xg": prob,
                "prediction": int(prob >= threshold),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict xG for one shot folder or a folder of shots.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--shot-dir", default=None, help="A single shot directory.")
    parser.add_argument("--input-dir", default=None, help="Directory containing shot directories.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-csv", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.shot_dir and not args.input_dir:
        raise SystemExit("Provide --shot-dir or --input-dir")
    shot_dirs = [Path(args.shot_dir)] if args.shot_dir else list(iter_shot_dirs(args.input_dir))
    predictions = predict_shot_dirs(args.checkpoint, shot_dirs, args.device)
    if args.output_csv:
        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(out_path, index=False)
    print(predictions.to_string(index=False))


if __name__ == "__main__":
    main()
