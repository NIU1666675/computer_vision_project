from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .engine import save_json
from .metrics import binary_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tabular xG baselines from homography features.")
    parser.add_argument("--features-csv", required=True, help="CSV with label and numeric feature columns.")
    parser.add_argument("--output-dir", default="runs/tabular")
    parser.add_argument("--model", default="logreg", choices=["logreg", "xgboost", "both"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def infer_feature_columns(df: pd.DataFrame) -> list[str]:
    blocked = {"sample_id", "label", "split", "shot_dir", "rel_dir", "match_id", "half", "shot_index"}
    return [
        column
        for column in df.columns
        if column not in blocked and pd.api.types.is_numeric_dtype(df[column])
    ]


def split_frame(df: pd.DataFrame, seed: int):
    if "split" in df.columns:
        train = df[df["split"] == "train"].copy()
        test = df[df["split"] == "test"].copy()
        if test.empty and "val" in set(df["split"]):
            test = df[df["split"] == "val"].copy()
        return train, test
    train, test = train_test_split(df, test_size=0.20, random_state=seed, stratify=df["label"])
    return train.copy(), test.copy()


def clean_features(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    frame[features] = frame[features].replace([np.inf, -np.inf], np.nan)
    return frame


def fit_logistic(train: pd.DataFrame, features: list[str]):
    model = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    model.fit(train[features], train["label"])
    return model


def fit_xgboost(train: pd.DataFrame, features: list[str], seed: int):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("Install xgboost with: pip install -e .[homography]") from exc
    positives = int(train["label"].sum())
    negatives = int((train["label"] == 0).sum())
    scale_pos_weight = negatives / max(positives, 1)
    model = XGBClassifier(
        n_estimators=250,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
    )
    model.fit(train[features], train["label"])
    return model


def predict_proba(model, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(frame[features])[:, 1]
    return model.predict(frame[features])


def save_model(model, path: Path) -> None:
    try:
        import joblib
    except ImportError as exc:
        raise ImportError("Install joblib with: pip install joblib") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def evaluate_and_save(name: str, model, test: pd.DataFrame, features: list[str], out_dir: Path, threshold: float):
    prob = predict_proba(model, test, features)
    metrics = binary_metrics(test["label"].to_numpy(), prob, threshold)
    predictions = test[["sample_id", "label"]].copy() if "sample_id" in test else test[["label"]].copy()
    predictions["xg"] = prob
    predictions["prediction"] = (prob >= threshold).astype(int)
    save_json(metrics, out_dir / f"{name}_metrics.json")
    predictions.to_csv(out_dir / f"{name}_predictions.csv", index=False)
    save_model(model, out_dir / f"{name}.joblib")
    return metrics


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.features_csv)
    if "label" not in df.columns:
        raise SystemExit("features CSV must contain a label column")
    features = infer_feature_columns(df)
    if not features:
        raise SystemExit("No numeric feature columns found")

    df = clean_features(df, features)
    train, test = split_frame(df, args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "features.txt").write_text("\n".join(features), encoding="utf-8")

    all_metrics = {}
    if args.model in ("logreg", "both"):
        model = fit_logistic(train, features)
        all_metrics["logreg"] = evaluate_and_save("logreg", model, test, features, out_dir, args.threshold)
    if args.model in ("xgboost", "both"):
        model = fit_xgboost(train, features, args.seed)
        all_metrics["xgboost"] = evaluate_and_save("xgboost", model, test, features, out_dir, args.threshold)

    save_json(all_metrics, out_dir / "metrics.json")
    print(pd.DataFrame(all_metrics).T.to_string())


if __name__ == "__main__":
    main()
