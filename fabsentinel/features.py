"""FDC summary statistics: turn each wafer's time-series trace into one feature row.

Fabs rarely model raw traces directly; fault detection and classification (FDC)
systems summarise each sensor per recipe step. We use mean, standard deviation
and slope per (step, sensor), plus the main-etch endpoint time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .simulator import SENSORS, Wafer

STEP_NAMES = {1: "stab", 2: "main", 3: "over"}


def _slope(y: np.ndarray) -> float:
    if len(y) < 3:
        return 0.0
    x = np.arange(len(y), dtype=float)
    x -= x.mean()
    return float((x * (y - y.mean())).sum() / (x * x).sum())


def wafer_features(trace: pd.DataFrame) -> dict[str, float]:
    feats: dict[str, float] = {}
    for step, g in trace.groupby("step"):
        tag = STEP_NAMES[int(step)]
        # Skip the first 3 s of each step: transient settling, ignored by real FDC windows too
        g = g.iloc[3:] if len(g) > 6 else g
        for s in SENSORS:
            y = g[s].to_numpy(dtype=float)
            feats[f"{tag}.{s}.mean"] = float(y.mean())
            feats[f"{tag}.{s}.std"] = float(y.std(ddof=1)) if len(y) > 1 else 0.0
            feats[f"{tag}.{s}.slope"] = _slope(y)
    feats["main.endpoint_s"] = float((trace["step"] == 2).sum())
    return feats


def feature_table(wafers: list[Wafer]) -> pd.DataFrame:
    df = pd.DataFrame([wafer_features(w.trace) for w in wafers])
    df.index = [w.wafer_id for w in wafers]
    return df


def sensor_of(feature: str) -> str:
    """'main.rf_refl_w.mean' -> 'rf_refl_w';  'main.endpoint_s' -> 'endpoint_s'."""
    parts = feature.split(".")
    return parts[1]
