from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PitchDimensions:
    length: float = 105.0
    width: float = 68.0
    goal_width: float = 7.32

    @property
    def goal_center(self) -> np.ndarray:
        return np.array([self.length, self.width / 2.0], dtype=np.float32)

    @property
    def goal_posts(self) -> tuple[np.ndarray, np.ndarray]:
        half_goal = self.goal_width / 2.0
        return (
            np.array([self.length, self.width / 2.0 - half_goal], dtype=np.float32),
            np.array([self.length, self.width / 2.0 + half_goal], dtype=np.float32),
        )


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("Install the homography extras: pip install -e .[homography]") from exc
    return cv2


def compute_homography(image_points: Iterable[Iterable[float]], pitch_points: Iterable[Iterable[float]]) -> np.ndarray:
    """Estimate image-to-pitch homography from at least four point pairs."""
    cv2 = _require_cv2()
    src = np.asarray(list(image_points), dtype=np.float32)
    dst = np.asarray(list(pitch_points), dtype=np.float32)
    if src.shape[0] < 4 or dst.shape[0] < 4:
        raise ValueError("At least four image/pitch point pairs are required")
    matrix, mask = cv2.findHomography(src, dst, method=cv2.RANSAC)
    if matrix is None:
        raise ValueError("Could not estimate homography from the provided points")
    return matrix.astype(np.float32)


def project_points(points: Iterable[Iterable[float]], homography: np.ndarray) -> np.ndarray:
    points_array = np.asarray(list(points), dtype=np.float32)
    if points_array.size == 0:
        return points_array.reshape(0, 2)
    ones = np.ones((points_array.shape[0], 1), dtype=np.float32)
    homogeneous = np.concatenate([points_array, ones], axis=1)
    projected = homogeneous @ homography.T
    projected = projected[:, :2] / np.clip(projected[:, 2:3], 1e-8, None)
    return projected


def bbox_bottom_centers(boxes: pd.DataFrame) -> np.ndarray:
    required = {"x1", "y1", "x2", "y2"}
    missing = required.difference(boxes.columns)
    if missing:
        raise ValueError(f"Detection boxes missing columns: {sorted(missing)}")
    x = (boxes["x1"].to_numpy(dtype=np.float32) + boxes["x2"].to_numpy(dtype=np.float32)) / 2.0
    y = boxes["y2"].to_numpy(dtype=np.float32)
    return np.stack([x, y], axis=1)


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)))


def goal_angle(shooter_xy: np.ndarray, pitch: PitchDimensions = PitchDimensions()) -> float:
    left_post, right_post = pitch.goal_posts
    shooter = np.asarray(shooter_xy, dtype=np.float32)
    v1 = left_post - shooter
    v2 = right_post - shooter
    denom = max(float(np.linalg.norm(v1) * np.linalg.norm(v2)), 1e-8)
    cosine = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def _same_side(p1: np.ndarray, p2: np.ndarray, a: np.ndarray, b: np.ndarray) -> bool:
    cp1 = np.cross(b - a, p1 - a)
    cp2 = np.cross(b - a, p2 - a)
    return bool(cp1 * cp2 >= 0)


def point_in_triangle(point: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    point = np.asarray(point, dtype=np.float32)
    return _same_side(point, a, b, c) and _same_side(point, b, a, c) and _same_side(point, c, a, b)


def geometry_features(
    sample_id: str,
    shooter_xy: Iterable[float],
    defenders_xy: Iterable[Iterable[float]],
    goalkeeper_xy: Iterable[float] | None = None,
    pitch: PitchDimensions = PitchDimensions(),
) -> dict[str, float | int | str]:
    shooter = np.asarray(shooter_xy, dtype=np.float32)
    defenders = np.asarray(list(defenders_xy), dtype=np.float32).reshape(-1, 2)
    goal_center = pitch.goal_center
    left_post, right_post = pitch.goal_posts

    if len(defenders):
        defender_distances = np.linalg.norm(defenders - shooter.reshape(1, 2), axis=1)
        nearest_defender = float(defender_distances.min())
        defenders_in_cone = int(
            sum(point_in_triangle(defender, shooter, left_post, right_post) for defender in defenders)
        )
    else:
        nearest_defender = float("nan")
        defenders_in_cone = 0

    features: dict[str, float | int | str] = {
        "sample_id": sample_id,
        "shot_x": float(shooter[0]),
        "shot_y": float(shooter[1]),
        "distance_to_goal": euclidean(shooter, goal_center),
        "angle_to_goal": goal_angle(shooter, pitch),
        "n_defenders": int(len(defenders)),
        "defenders_in_cone": defenders_in_cone,
        "nearest_defender_distance": nearest_defender,
    }

    if goalkeeper_xy is not None:
        keeper = np.asarray(goalkeeper_xy, dtype=np.float32)
        features.update(
            {
                "goalkeeper_x": float(keeper[0]),
                "goalkeeper_y": float(keeper[1]),
                "goalkeeper_distance_to_goal_line": abs(float(pitch.length - keeper[0])),
                "goalkeeper_distance_to_shooter": euclidean(keeper, shooter),
            }
        )
    return features


def features_from_projected_positions(
    positions: pd.DataFrame,
    labels: pd.DataFrame | None = None,
    pitch: PitchDimensions = PitchDimensions(),
) -> pd.DataFrame:
    """Build tabular xG features from projected player positions.

    Expected columns in positions: sample_id, role, x, y.
    Roles: shooter, defender, goalkeeper. Extra rows are ignored.
    """
    required = {"sample_id", "role", "x", "y"}
    missing = required.difference(positions.columns)
    if missing:
        raise ValueError(f"Projected positions missing columns: {sorted(missing)}")

    rows = []
    for sample_id, group in positions.groupby("sample_id"):
        shooters = group[group["role"] == "shooter"]
        if shooters.empty:
            continue
        shooter_xy = shooters[["x", "y"]].iloc[0].to_numpy(dtype=np.float32)
        defenders_xy = group[group["role"] == "defender"][["x", "y"]].to_numpy(dtype=np.float32)
        keepers = group[group["role"] == "goalkeeper"]
        goalkeeper_xy = None
        if not keepers.empty:
            goalkeeper_xy = keepers[["x", "y"]].iloc[0].to_numpy(dtype=np.float32)
        rows.append(geometry_features(str(sample_id), shooter_xy, defenders_xy, goalkeeper_xy, pitch))

    features = pd.DataFrame(rows)
    if labels is not None:
        features = features.merge(labels[["sample_id", "label"]], on="sample_id", how="left")
    return features


def load_correspondences(path: str | Path) -> dict[str, np.ndarray]:
    """Load manual calibration points and return homographies by sample_id.

    JSON format:
    {
      "sample_id": {
        "image_points": [[x, y], ...],
        "pitch_points": [[x, y], ...]
      }
    }
    """
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {
        sample_id: compute_homography(entry["image_points"], entry["pitch_points"])
        for sample_id, entry in data.items()
    }


def detect_field_lines(image_path: str | Path, canny_low: int = 50, canny_high: int = 150) -> np.ndarray:
    """Return candidate pitch lines with Hough transform for later calibration work."""
    cv2 = _require_cv2()
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_low, canny_high)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180.0, threshold=70, minLineLength=45, maxLineGap=8)
    if lines is None:
        return np.empty((0, 4), dtype=np.float32)
    return lines.reshape(-1, 4).astype(np.float32)
