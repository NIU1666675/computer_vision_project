from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset


SHOT_FRAME = "frame_shot.jpg"
PREV_FRAME_PATTERN = "frame_prev_{idx}.jpg"
SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class ShotSample:
    sample_id: str
    shot_dir: Path
    rel_dir: str
    label: int
    match_id: str
    half: str | None
    shot_index: int | None


def is_frames_root(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    for child in path.iterdir():
        if child.is_dir() and (child / "label.txt").exists():
            return True
    return False


def find_frames_root(dataset_root: str | Path | None = None) -> Path:
    """Resolve the shot-frame dataset even if the outer Dataset_final is passed."""
    candidates: list[Path] = []
    if dataset_root is not None:
        root = Path(dataset_root)
        candidates.extend(
            [
                root,
                root / "Frames_Bons_Definitius",
                root / "Dataset_final" / "Frames_Bons_Definitius",
                root / "Dataset_final" / "Dataset_final" / "Frames_Bons_Definitius",
            ]
        )
    candidates.extend(
        [
            Path("Dataset_final") / "Frames_Bons_Definitius",
            Path("Dataset_final") / "Dataset_final" / "Frames_Bons_Definitius",
        ]
    )

    for candidate in candidates:
        if is_frames_root(candidate):
            return candidate.resolve()

    checked = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find Frames_Bons_Definitius. Checked:\n{checked}")


def parse_sample_name(name: str) -> tuple[str, str | None, int | None]:
    match = re.match(r"^(?P<match>.+)_(?P<half>[12])_224p_shot_(?P<shot>\d+)$", name)
    if match is None:
        return name, None, None
    return match.group("match"), match.group("half"), int(match.group("shot"))


def sequence_frame_names(sequence_length: int = 6, include_shot_frame: bool = True) -> list[str]:
    prev_frames = [PREV_FRAME_PATTERN.format(idx=idx) for idx in range(4, -1, -1)]
    full_sequence = prev_frames + ([SHOT_FRAME] if include_shot_frame else [])
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if sequence_length > len(full_sequence):
        raise ValueError(
            f"sequence_length={sequence_length} needs more frames than available "
            f"({len(full_sequence)})."
        )
    return full_sequence[-sequence_length:]


def scan_samples(frames_root: str | Path, strict: bool = True) -> list[ShotSample]:
    root = find_frames_root(frames_root)
    samples: list[ShotSample] = []
    errors: list[str] = []
    expected_frames = [SHOT_FRAME] + [PREV_FRAME_PATTERN.format(idx=i) for i in range(5)]

    for shot_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        label_path = shot_dir / "label.txt"
        if not label_path.exists():
            errors.append(f"{shot_dir.name}: missing label.txt")
            continue
        try:
            label = int(label_path.read_text(encoding="utf-8").strip())
        except ValueError:
            errors.append(f"{shot_dir.name}: invalid label")
            continue
        if label not in (0, 1):
            errors.append(f"{shot_dir.name}: label must be 0 or 1")
            continue

        missing = [frame for frame in expected_frames if not (shot_dir / frame).exists()]
        if missing:
            errors.append(f"{shot_dir.name}: missing frames {missing}")
            continue

        match_id, half, shot_index = parse_sample_name(shot_dir.name)
        samples.append(
            ShotSample(
                sample_id=shot_dir.name,
                shot_dir=shot_dir,
                rel_dir=shot_dir.relative_to(root).as_posix(),
                label=label,
                match_id=match_id,
                half=half,
                shot_index=shot_index,
            )
        )

    if strict and errors:
        preview = "\n".join(errors[:20])
        raise ValueError(f"Dataset validation found {len(errors)} issue(s):\n{preview}")
    if not samples:
        raise ValueError(f"No valid samples found in {root}")
    return samples


def samples_to_frame(samples: Sequence[ShotSample]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": sample.sample_id,
                "rel_dir": sample.rel_dir,
                "label": sample.label,
                "match_id": sample.match_id,
                "half": sample.half,
                "shot_index": sample.shot_index,
            }
            for sample in samples
        ]
    )


def _normalise_ratios(ratios: Sequence[float]) -> tuple[float, float, float]:
    if len(ratios) != 3:
        raise ValueError("split_ratios must have exactly three values: train, val, test")
    total = float(sum(ratios))
    if total <= 0:
        raise ValueError("split_ratios must sum to a positive value")
    return tuple(float(ratio) / total for ratio in ratios)  # type: ignore[return-value]


def _sample_stratified_split(df: pd.DataFrame, ratios: Sequence[float], seed: int) -> pd.Series:
    from sklearn.model_selection import train_test_split

    train_ratio, val_ratio, test_ratio = _normalise_ratios(ratios)
    labels = df["label"].to_numpy()
    idx = np.arange(len(df))
    train_idx, holdout_idx = train_test_split(
        idx,
        train_size=train_ratio,
        random_state=seed,
        stratify=labels,
    )
    holdout_labels = labels[holdout_idx]
    val_share = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        holdout_idx,
        train_size=val_share,
        random_state=seed,
        stratify=holdout_labels,
    )
    split = pd.Series(index=df.index, dtype="object")
    split.iloc[train_idx] = "train"
    split.iloc[val_idx] = "val"
    split.iloc[test_idx] = "test"
    return split


def _group_stratified_split(df: pd.DataFrame, ratios: Sequence[float], seed: int) -> pd.Series:
    ratios = _normalise_ratios(ratios)
    rng = random.Random(seed)
    groups = []
    for match_id, group in df.groupby("match_id", sort=False):
        groups.append(
            {
                "match_id": match_id,
                "indices": list(group.index),
                "total": len(group),
                "positive": int(group["label"].sum()),
            }
        )

    n_total = len(df)
    n_positive = int(df["label"].sum())
    target_total = {split: ratios[i] * n_total for i, split in enumerate(SPLIT_NAMES)}
    target_positive = {split: ratios[i] * n_positive for i, split in enumerate(SPLIT_NAMES)}
    best_assignment: dict[str, str] | None = None
    best_cost = float("inf")

    # Randomized greedy search keeps whole matches together while balancing label rates.
    for _ in range(400):
        order = groups[:]
        rng.shuffle(order)
        order.sort(key=lambda item: (item["positive"] + rng.random() * 0.25, item["total"]), reverse=True)
        total_count = {split: 0 for split in SPLIT_NAMES}
        positive_count = {split: 0 for split in SPLIT_NAMES}
        assignment: dict[str, str] = {}

        for group in order:
            best_split = min(
                SPLIT_NAMES,
                key=lambda split: _global_split_cost(
                    total_count,
                    positive_count,
                    target_total,
                    target_positive,
                    n_total,
                    n_positive,
                    candidate_split=split,
                    candidate_total=group["total"],
                    candidate_positive=group["positive"],
                ),
            )
            assignment[group["match_id"]] = best_split
            total_count[best_split] += group["total"]
            positive_count[best_split] += group["positive"]

        cost = _global_split_cost(total_count, positive_count, target_total, target_positive, n_total, n_positive)
        if cost < best_cost:
            best_cost = cost
            best_assignment = assignment

    if best_assignment is None:
        raise RuntimeError("Could not create group split assignment")
    return df["match_id"].map(best_assignment)


def _global_split_cost(
    total_count: dict[str, int],
    positive_count: dict[str, int],
    target_total: dict[str, float],
    target_positive: dict[str, float],
    n_total: int,
    n_positive: int,
    candidate_split: str | None = None,
    candidate_total: int = 0,
    candidate_positive: int = 0,
) -> float:
    n_negative = max(n_total - n_positive, 1)
    positive_scale = max(n_positive, 1)
    cost = 0.0
    for split in SPLIT_NAMES:
        total = total_count[split] + (candidate_total if split == candidate_split else 0)
        positive = positive_count[split] + (candidate_positive if split == candidate_split else 0)
        negative = total - positive
        target_negative = target_total[split] - target_positive[split]
        total_error = (total - target_total[split]) / max(n_total, 1)
        positive_error = (positive - target_positive[split]) / positive_scale
        negative_error = (negative - target_negative) / n_negative
        overfill = max(total - target_total[split], 0.0) / max(n_total, 1)
        cost += total_error**2 + 5.0 * positive_error**2 + negative_error**2 + 2.0 * overfill**2
    return cost


def create_split_manifest(
    frames_root: str | Path,
    output_path: str | Path,
    ratios: Sequence[float] = (0.70, 0.15, 0.15),
    seed: int = 42,
    strategy: str = "group",
) -> pd.DataFrame:
    samples = scan_samples(frames_root)
    df = samples_to_frame(samples)
    strategy = strategy.lower()
    if strategy == "group":
        df["split"] = _group_stratified_split(df, ratios, seed)
    elif strategy == "sample":
        df["split"] = _sample_stratified_split(df, ratios, seed)
    else:
        raise ValueError("strategy must be 'group' or 'sample'")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def load_manifest(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"sample_id", "rel_dir", "label", "match_id", "split"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    return df


class FrameTransform:
    def __init__(
        self,
        image_size: Sequence[int] = (224, 398),
        augment: bool = False,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.image_size = tuple(int(value) for value in image_size)
        self.augment = augment
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        height, width = self.image_size
        image = image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
        if self.augment:
            image = self._augment(image)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return (tensor - self.mean) / self.std

    def _augment(self, image: Image.Image) -> Image.Image:
        if random.random() < 0.45:
            image = self._random_resized_crop(image)
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < 0.35:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
        if random.random() < 0.35:
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.15))
        if random.random() < 0.25:
            image = ImageEnhance.Color(image).enhance(random.uniform(0.90, 1.10))
        if random.random() < 0.12:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8)))
        return image

    def _random_resized_crop(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        scale = random.uniform(0.90, 1.0)
        crop_w = max(int(width * scale), 1)
        crop_h = max(int(height * scale), 1)
        left = random.randint(0, max(width - crop_w, 0))
        top = random.randint(0, max(height - crop_h, 0))
        cropped = image.crop((left, top, left + crop_w, top + crop_h))
        out_h, out_w = self.image_size
        return cropped.resize((out_w, out_h), Image.Resampling.BILINEAR)


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


class ShotFrameDataset(Dataset):
    def __init__(
        self,
        frames_root: str | Path,
        manifest: str | Path | pd.DataFrame,
        split: str,
        model_input: str = "single",
        image_size: Sequence[int] = (224, 398),
        sequence_length: int = 6,
        include_shot_frame: bool = True,
        augment: bool = False,
        limit: int | None = None,
    ) -> None:
        self.frames_root = find_frames_root(frames_root)
        self.model_input = model_input
        self.sequence_length = sequence_length
        self.include_shot_frame = include_shot_frame
        self.transform = FrameTransform(image_size=image_size, augment=augment)
        self.sequence_names = sequence_frame_names(sequence_length, include_shot_frame)

        if isinstance(manifest, pd.DataFrame):
            df = manifest.copy()
        else:
            df = load_manifest(manifest)
        if split not in SPLIT_NAMES:
            raise ValueError(f"split must be one of {SPLIT_NAMES}")
        df = df[df["split"] == split].reset_index(drop=True)
        if limit is not None:
            df = df.head(limit).copy()
        if df.empty:
            raise ValueError(f"No samples for split '{split}'")
        self.rows = df.to_dict("records")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        shot_dir = self.frames_root / str(row["rel_dir"])
        label = torch.tensor(float(row["label"]), dtype=torch.float32)

        if self.model_input == "single":
            x = self.transform(load_rgb(shot_dir / SHOT_FRAME))
        elif self.model_input == "sequence":
            frames = [self.transform(load_rgb(shot_dir / name)) for name in self.sequence_names]
            x = torch.stack(frames, dim=0)
        else:
            raise ValueError("model_input must be 'single' or 'sequence'")

        return {
            "x": x,
            "y": label,
            "sample_id": str(row["sample_id"]),
            "shot_dir": str(shot_dir),
        }

    @property
    def labels(self) -> list[int]:
        return [int(row["label"]) for row in self.rows]


def summarize_manifest(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in SPLIT_NAMES:
        part = df[df["split"] == split]
        rows.append(
            {
                "split": split,
                "samples": len(part),
                "goals": int(part["label"].sum()),
                "non_goals": int((part["label"] == 0).sum()),
                "goal_rate": float(part["label"].mean()) if len(part) else 0.0,
                "matches": int(part["match_id"].nunique()) if "match_id" in part else 0,
            }
        )
    return pd.DataFrame(rows)


def iter_shot_dirs(path: str | Path) -> Iterable[Path]:
    root = Path(path)
    if (root / "label.txt").exists() or (root / SHOT_FRAME).exists():
        yield root
        return
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / SHOT_FRAME).exists():
            yield child
