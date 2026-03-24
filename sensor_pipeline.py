"""
Sensor data: load → filter noise → preprocess (missing) → normalize → ML-ready tabular features.
Uses pandas, numpy, and scikit-learn only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

# #region agent log
import importlib.util
import json
import sys
import time


def _agent_debug_log(hypothesis_id: str, message: str, data: dict) -> None:
    try:
        payload = {
            "sessionId": "f856ed",
            "runId": "notebook-debug",
            "timestamp": int(time.time() * 1000),
            "hypothesisId": hypothesis_id,
            "location": "sensor_pipeline.py:pre_numpy",
            "message": message,
            "data": data,
        }
        with (Path(__file__).resolve().parent / "debug-f856ed.log").open(
            "a", encoding="utf-8"
        ) as _f:
            _f.write(json.dumps(payload) + "\n")
    except Exception:
        pass


_umath_spec = importlib.util.find_spec("numpy._core._multiarray_umath")
_agent_debug_log(
    "H4",
    "pre-numpy interpreter and umath extension spec",
    {
        "executable": sys.executable,
        "version": sys.version,
        "version_info": list(sys.version_info[:3]),
        "umath_spec_origin": getattr(_umath_spec, "origin", None),
    },
)
# #endregion

try:
    import numpy as np
except ImportError as _numpy_err:
    # #region agent log
    _agent_debug_log(
        "H1",
        "numpy import failed",
        {
            "error_type": type(_numpy_err).__name__,
            "error_repr": repr(_numpy_err),
            "cause_type": type(_numpy_err.__cause__).__name__
            if _numpy_err.__cause__
            else None,
            "cause_repr": repr(_numpy_err.__cause__)
            if _numpy_err.__cause__
            else None,
        },
    )
    # #endregion
    raise
# #region agent log
_agent_debug_log(
    "H1",
    "numpy import succeeded",
    {"numpy_file": np.__file__},
)
# #endregion
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def make_synthetic_sensor_data(
    n_rows: int = 500,
    n_sensors: int = 3,
    seed: int | None = 42,
    missing_frac: float = 0.02,
    noise_std: float = 0.15,
) -> pd.DataFrame:
    """Wide sensor table with time index, smooth trends, noise, and random missing values."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_rows, dtype=float)
    cols = [f"sensor_{i}" for i in range(n_sensors)]
    base = np.column_stack(
        [np.sin(0.05 * t + i) + 0.1 * (i + 1) for i in range(n_sensors)]
    )
    noise = rng.normal(0.0, noise_std, size=base.shape)
    values = base + noise
    df = pd.DataFrame(values, columns=cols)
    df.insert(0, "timestamp", pd.date_range("2024-01-01", periods=n_rows, freq="s"))

    n_miss = int(n_rows * n_sensors * missing_frac)
    if n_miss > 0:
        ri = rng.integers(0, n_rows, size=n_miss)
        ci = rng.integers(0, n_sensors, size=n_miss)
        df.iloc[ri, ci + 1] = np.nan

    return df


def load_sensor_csv(
    path: str | Path,
    parse_dates: Sequence[str] | None = ("timestamp",),
) -> pd.DataFrame:
    """Load a wide CSV; tries to parse timestamp column(s) if present."""
    path = Path(path)
    df = pd.read_csv(path)
    if parse_dates:
        for c in parse_dates:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def filter_sensor_noise(
    df: pd.DataFrame,
    value_cols: Sequence[str],
    window: int = 5,
    method: Literal["median", "mean"] = "median",
    min_periods: int = 1,
) -> pd.DataFrame:
    """Rolling smooth per sensor column (reduces high-frequency noise)."""
    out = df.copy()
    roll = (
        out[value_cols].rolling(window=window, min_periods=min_periods, center=True)
    )
    if method == "median":
        out[value_cols] = roll.median()
    else:
        out[value_cols] = roll.mean()
    return out


def preprocess_tabular(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    ffill_limit: int | None = None,
) -> pd.DataFrame:
    """
    Select features, coerce numeric, optional forward-fill (time series).
    Remaining NaNs are handled by ``build_feature_pipeline`` (SimpleImputer + scaler).
    """
    X = df.loc[:, list(feature_cols)].apply(pd.to_numeric, errors="coerce")
    if ffill_limit is not None:
        X = X.ffill(limit=ffill_limit)
    return X


def build_feature_pipeline(
    imputer_strategy: Literal["mean", "median", "most_frequent", "constant"] = "median",
    scaler: Literal["standard", "minmax"] = "standard",
) -> Pipeline:
    """Imputation then scaling (fit on train only to avoid leakage)."""
    scaler_cls = StandardScaler if scaler == "standard" else MinMaxScaler
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy=imputer_strategy)),
            ("scaler", scaler_cls()),
        ]
    )


def fit_transform_features(
    pipeline: Pipeline,
    X_train: pd.DataFrame | np.ndarray,
    X_test: pd.DataFrame | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Fit pipeline on train; transform train and optional test."""
    Xtr = np.asarray(pipeline.fit_transform(X_train), dtype=float)
    Xte = None
    if X_test is not None:
        Xte = np.asarray(pipeline.transform(X_test), dtype=float)
    return Xtr, Xte


def make_synthetic_labels(
    X: pd.DataFrame,
    feature_cols: Sequence[str],
    seed: int | None = 42,
) -> np.ndarray:
    """Binary labels from a simple rule on features (demo only)."""
    rng = np.random.default_rng(seed)
    s0 = X[feature_cols[0]].to_numpy(dtype=float)
    y = (s0 > np.median(s0)).astype(np.int64)
    flip = rng.random(len(y)) < 0.05
    y[flip] = 1 - y[flip]
    return y


def _smoke() -> None:
    raw = make_synthetic_sensor_data(n_rows=120, n_sensors=3, seed=0)
    value_cols = [c for c in raw.columns if c != "timestamp"]
    raw = raw.sort_values("timestamp").reset_index(drop=True)
    filtered = filter_sensor_noise(raw, value_cols, window=5, method="median")
    X_df = preprocess_tabular(filtered, value_cols, ffill_limit=5)
    pipe = build_feature_pipeline(scaler="standard")
    n = len(X_df)
    split = int(n * 0.8)
    X_train, X_test = X_df.iloc[:split], X_df.iloc[split:]
    Xtr, Xte = fit_transform_features(pipe, X_train, X_test)
    assert Xtr.shape[0] == split and Xte is not None and Xte.shape[0] == n - split
    y = make_synthetic_labels(X_df, value_cols, seed=1)
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=0)
    clf.fit(Xtr, y[:split])
    score = clf.score(Xte, y[split:])
    assert 0.0 <= score <= 1.0
    print("sensor_pipeline smoke OK, holdout accuracy:", round(score, 4))


if __name__ == "__main__":
    _smoke()
