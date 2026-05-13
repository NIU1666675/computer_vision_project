from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from .data import SHOT_FRAME
from .predict import predict_shot_dirs


def save_overlay(frame_path: str | Path, xg: float, output_path: str | Path, label: int | None = None) -> None:
    image = Image.open(frame_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    text = f"xG {xg:.3f}"
    if label is not None:
        text += f" | label {label}"
    font = ImageFont.load_default()
    box = draw.textbbox((0, 0), text, font=font)
    width = box[2] - box[0]
    height = box[3] - box[1]
    draw.rectangle((8, 8, 20 + width, 20 + height), fill=(0, 0, 0, 175))
    draw.text((14, 13), text, fill=(255, 255, 255, 255), font=font)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay predicted xG on shot frames.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint used when --predictions-csv is omitted.")
    parser.add_argument("--shot-dir", default=None, help="Shot directory for direct prediction.")
    parser.add_argument("--predictions-csv", default=None, help="CSV with shot_dir and xg columns.")
    parser.add_argument("--output-dir", default="outputs/visualizations")
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.predictions_csv:
        predictions = pd.read_csv(args.predictions_csv)
    else:
        if not args.checkpoint or not args.shot_dir:
            raise SystemExit("Provide --predictions-csv or both --checkpoint and --shot-dir")
        predictions = predict_shot_dirs(args.checkpoint, [args.shot_dir], args.device)

    out_dir = Path(args.output_dir)
    rows = predictions.head(args.limit) if args.limit else predictions
    for _, row in rows.iterrows():
        shot_dir = Path(row["shot_dir"])
        output_path = out_dir / f"{shot_dir.name}_xg.jpg"
        label = None if pd.isna(row.get("label")) else int(row.get("label"))
        save_overlay(shot_dir / SHOT_FRAME, float(row["xg"]), output_path, label)
        print(output_path)


if __name__ == "__main__":
    main()
