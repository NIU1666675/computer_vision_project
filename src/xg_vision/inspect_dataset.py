from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image

from .data import create_split_manifest, find_frames_root, samples_to_frame, scan_samples, summarize_manifest
from .engine import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the shot-frame dataset.")
    parser.add_argument("--dataset-root", default="Dataset_final")
    parser.add_argument("--output-dir", default="outputs/dataset_report")
    parser.add_argument("--write-splits", action="store_true")
    parser.add_argument("--split-strategy", default="group", choices=["group", "sample"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = find_frames_root(args.dataset_root)
    samples = scan_samples(root)
    df = samples_to_frame(samples)

    sizes: dict[str, int] = {}
    for sample in samples:
        with Image.open(sample.shot_dir / "frame_shot.jpg") as image:
            key = f"{image.size[0]}x{image.size[1]}"
            sizes[key] = sizes.get(key, 0) + 1

    report = {
        "frames_root": str(root),
        "samples": len(samples),
        "goals": int(df["label"].sum()),
        "non_goals": int((df["label"] == 0).sum()),
        "goal_rate": float(df["label"].mean()),
        "matches": int(df["match_id"].nunique()),
        "image_sizes": sizes,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "samples.csv", index=False)
    save_json(report, out_dir / "summary.json")

    print(pd.Series(report).to_string())
    if args.write_splits:
        split_df = create_split_manifest(
            root,
            out_dir / "splits.csv",
            seed=args.seed,
            strategy=args.split_strategy,
        )
        summary = summarize_manifest(split_df)
        summary.to_csv(out_dir / "split_summary.csv", index=False)
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
