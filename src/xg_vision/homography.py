from __future__ import annotations

import json
import math
import argparse
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

    def landmarks(self) -> dict[str, tuple[float, float]]:
        """Canonical pitch points in metres, useful for manual calibration."""
        penalty_depth = 16.5
        penalty_width = 40.32
        goal_area_depth = 5.5
        goal_area_width = 18.32
        center_y = self.width / 2.0
        return {
            "left_touchline_top": (0.0, 0.0),
            "left_touchline_bottom": (0.0, self.width),
            "right_touchline_top": (self.length, 0.0),
            "right_touchline_bottom": (self.length, self.width),
            "center_spot": (self.length / 2.0, center_y),
            "right_penalty_left_top": (self.length - penalty_depth, center_y - penalty_width / 2.0),
            "right_penalty_left_bottom": (self.length - penalty_depth, center_y + penalty_width / 2.0),
            "right_goal_area_left_top": (self.length - goal_area_depth, center_y - goal_area_width / 2.0),
            "right_goal_area_left_bottom": (self.length - goal_area_depth, center_y + goal_area_width / 2.0),
            "right_goal_top": (self.length, center_y - self.goal_width / 2.0),
            "right_goal_bottom": (self.length, center_y + self.goal_width / 2.0),
            "left_penalty_right_top": (penalty_depth, center_y - penalty_width / 2.0),
            "left_penalty_right_bottom": (penalty_depth, center_y + penalty_width / 2.0),
            "left_goal_area_right_top": (goal_area_depth, center_y - goal_area_width / 2.0),
            "left_goal_area_right_bottom": (goal_area_depth, center_y + goal_area_width / 2.0),
            "left_goal_top": (0.0, center_y - self.goal_width / 2.0),
            "left_goal_bottom": (0.0, center_y + self.goal_width / 2.0),
        }


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("Install the homography extras: pip install -e .[homography]") from exc
    return cv2


def compute_homography(
    image_points: Iterable[Iterable[float]],
    pitch_points: Iterable[Iterable[float]],
) -> np.ndarray:
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


def reprojection_error(
    image_points: Iterable[Iterable[float]],
    pitch_points: Iterable[Iterable[float]],
    homography: np.ndarray,
) -> float:
    projected = project_points(image_points, homography)
    target = np.asarray(list(pitch_points), dtype=np.float32)
    if len(projected) == 0:
        return float("nan")
    return float(np.linalg.norm(projected - target, axis=1).mean())


def project_points(points: Iterable[Iterable[float]], homography: np.ndarray) -> np.ndarray:
    points_array = np.asarray(list(points), dtype=np.float32)
    if points_array.size == 0:
        return points_array.reshape(0, 2)
    ones = np.ones((points_array.shape[0], 1), dtype=np.float32)
    homogeneous = np.concatenate([points_array, ones], axis=1)
    projected = homogeneous @ homography.T
    w = projected[:, 2:3]
    w = np.where(np.abs(w) < 1e-8, 1e-8, w)
    projected = projected[:, :2] / w
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


def _cross2d(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _same_side(p1: np.ndarray, p2: np.ndarray, a: np.ndarray, b: np.ndarray) -> bool:
    cp1 = _cross2d(b - a, p1 - a)
    cp2 = _cross2d(b - a, p2 - a)
    return bool(cp1 * cp2 >= 0)


def point_in_triangle(point: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    point = np.asarray(point, dtype=np.float32)
    return _same_side(point, a, b, c) and _same_side(point, b, a, c) and _same_side(point, c, a, b)


def mirror_to_attacking_right(points: np.ndarray, pitch: PitchDimensions = PitchDimensions()) -> np.ndarray:
    mirrored = np.asarray(points, dtype=np.float32).copy()
    if mirrored.size == 0:
        return mirrored.reshape(0, 2)
    mirrored[:, 0] = pitch.length - mirrored[:, 0]
    mirrored[:, 1] = pitch.width - mirrored[:, 1]
    return mirrored


def distance_to_segment(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    start = np.asarray(start, dtype=np.float32)
    end = np.asarray(end, dtype=np.float32)
    segment = end - start
    denom = max(float(np.dot(segment, segment)), 1e-8)
    t = np.clip(((points - start) @ segment) / denom, 0.0, 1.0)
    projection = start + t.reshape(-1, 1) * segment
    return np.linalg.norm(points - projection, axis=1)


def geometry_features(
    sample_id: str,
    shooter_xy: Iterable[float],
    defenders_xy: Iterable[Iterable[float]],
    goalkeeper_xy: Iterable[float] | None = None,
    pitch: PitchDimensions = PitchDimensions(),
    attacking_direction: str = "right",
) -> dict[str, float | int | str]:
    shooter = np.asarray(shooter_xy, dtype=np.float32)
    defenders = np.asarray(list(defenders_xy), dtype=np.float32).reshape(-1, 2)
    goalkeeper = None if goalkeeper_xy is None else np.asarray(goalkeeper_xy, dtype=np.float32)
    if attacking_direction.lower() == "left":
        shooter = mirror_to_attacking_right(shooter.reshape(1, 2), pitch)[0]
        defenders = mirror_to_attacking_right(defenders, pitch)
        if goalkeeper is not None:
            goalkeeper = mirror_to_attacking_right(goalkeeper.reshape(1, 2), pitch)[0]

    goal_center = pitch.goal_center
    left_post, right_post = pitch.goal_posts

    if len(defenders):
        defender_distances = np.linalg.norm(defenders - shooter.reshape(1, 2), axis=1)
        nearest_defender = float(defender_distances.min())
        defenders_3m = int((defender_distances <= 3.0).sum())
        defenders_5m = int((defender_distances <= 5.0).sum())
        defenders_10m = int((defender_distances <= 10.0).sum())
        defenders_in_cone = int(
            sum(point_in_triangle(defender, shooter, left_post, right_post) for defender in defenders)
        )
        lane_distances = distance_to_segment(defenders, shooter, goal_center)
        defenders_in_lane = int(((lane_distances <= 2.0) & (defenders[:, 0] >= shooter[0])).sum())
        mean_defender_distance = float(defender_distances.mean())
    else:
        nearest_defender = float("nan")
        defenders_3m = 0
        defenders_5m = 0
        defenders_10m = 0
        defenders_in_cone = 0
        defenders_in_lane = 0
        mean_defender_distance = float("nan")

    features: dict[str, float | int | str] = {
        "sample_id": sample_id,
        "shot_x": float(shooter[0]),
        "shot_y": float(shooter[1]),
        "shot_x_norm": float(shooter[0] / pitch.length),
        "shot_y_norm": float(shooter[1] / pitch.width),
        "shot_lateral_distance": abs(float(shooter[1] - pitch.width / 2.0)),
        "distance_to_goal": euclidean(shooter, goal_center),
        "angle_to_goal": goal_angle(shooter, pitch),
        "n_defenders": int(len(defenders)),
        "defenders_within_3m": defenders_3m,
        "defenders_within_5m": defenders_5m,
        "defenders_within_10m": defenders_10m,
        "defenders_in_cone": defenders_in_cone,
        "defenders_in_shot_lane": defenders_in_lane,
        "nearest_defender_distance": nearest_defender,
        "mean_defender_distance": mean_defender_distance,
    }

    if goalkeeper is not None:
        keeper = goalkeeper
        features.update(
            {
                "goalkeeper_x": float(keeper[0]),
                "goalkeeper_y": float(keeper[1]),
                "goalkeeper_distance_to_goal_line": abs(float(pitch.length - keeper[0])),
                "goalkeeper_distance_to_shooter": euclidean(keeper, shooter),
                "goalkeeper_lateral_offset": abs(float(keeper[1] - pitch.width / 2.0)),
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
        direction = "right"
        if "attacking_direction" in group.columns and not group["attacking_direction"].isna().all():
            direction = str(group["attacking_direction"].dropna().iloc[0])
        rows.append(geometry_features(str(sample_id), shooter_xy, defenders_xy, goalkeeper_xy, pitch, direction))

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


def load_correspondence_report(path: str | Path) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    homographies: dict[str, np.ndarray] = {}
    rows = []
    for sample_id, entry in data.items():
        matrix = compute_homography(entry["image_points"], entry["pitch_points"])
        homographies[sample_id] = matrix
        rows.append(
            {
                "sample_id": sample_id,
                "n_points": len(entry["image_points"]),
                "reprojection_error_m": reprojection_error(entry["image_points"], entry["pitch_points"], matrix),
            }
        )
    return homographies, pd.DataFrame(rows)


def positions_from_detections(
    detections: pd.DataFrame,
    homographies: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Project detection bottom-centres to pitch coordinates.

    Required columns: sample_id, role, x1, y1, x2, y2.
    Accepted roles: shooter, defender, goalkeeper.
    """
    required = {"sample_id", "role", "x1", "y1", "x2", "y2"}
    missing = required.difference(detections.columns)
    if missing:
        raise ValueError(f"Detections missing columns: {sorted(missing)}")

    rows = []
    passthrough = [column for column in ("confidence", "team", "attacking_direction") if column in detections.columns]
    for sample_id, group in detections.groupby("sample_id"):
        matrix = homographies.get(str(sample_id))
        if matrix is None:
            continue
        projected = project_points(bbox_bottom_centers(group), matrix)
        for (_, detection), xy in zip(group.iterrows(), projected):
            row = {
                "sample_id": str(sample_id),
                "role": str(detection["role"]),
                "x": float(xy[0]),
                "y": float(xy[1]),
            }
            for column in passthrough:
                row[column] = detection[column]
            rows.append(row)
    return pd.DataFrame(rows)


def build_feature_table(
    detections_csv: str | Path,
    correspondences_json: str | Path,
    labels_csv: str | Path | None = None,
    positions_out: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detections = pd.read_csv(detections_csv)
    homographies, report = load_correspondence_report(correspondences_json)
    positions = positions_from_detections(detections, homographies)
    labels = pd.read_csv(labels_csv) if labels_csv is not None else None
    features = features_from_projected_positions(positions, labels=labels)
    if positions_out is not None:
        out_path = Path(positions_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        positions.to_csv(out_path, index=False)
    return features, report


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build homography-based tabular xG features.")
    parser.add_argument("--detections-csv", required=True, help="CSV with sample_id, role, x1, y1, x2, y2.")
    parser.add_argument("--correspondences-json", required=True, help="Manual calibration JSON.")
    parser.add_argument("--labels-csv", default=None, help="Optional manifest/splits CSV with labels.")
    parser.add_argument("--output-csv", required=True, help="Output feature table.")
    parser.add_argument("--positions-csv", default=None, help="Optional projected player positions output.")
    parser.add_argument("--calibration-report", default=None, help="Optional homography QA report output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features, report = build_feature_table(
        args.detections_csv,
        args.correspondences_json,
        labels_csv=args.labels_csv,
        positions_out=args.positions_csv,
    )
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_path, index=False)
    if args.calibration_report:
        report_path = Path(args.calibration_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(report_path, index=False)
    print(f"features={out_path} rows={len(features)}")


if __name__ == "__main__":
    main()
