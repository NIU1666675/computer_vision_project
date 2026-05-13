from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN, CNN+LSTM and CNN+attention sequentially.")
    parser.add_argument("--dataset-root", default="Dataset_final")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-root", default="runs/all_models")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    shared: list[str] = [
        sys.executable,
        "-m",
        "xg_vision.train",
        "--config",
        args.config,
        "--dataset-root",
        args.dataset_root,
    ]
    if args.epochs is not None:
        shared += ["--epochs", str(args.epochs)]
    if args.batch_size is not None:
        shared += ["--batch-size", str(args.batch_size)]
    if args.num_workers is not None:
        shared += ["--num-workers", str(args.num_workers)]
    if args.device is not None:
        shared += ["--device", args.device]
    if args.no_augment:
        shared += ["--no-augment"]
    if args.no_amp:
        shared += ["--no-amp"]

    for model in ("cnn", "lstm", "attention"):
        cmd = shared + ["--model", model, "--output-dir", str(output_root / model)]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
