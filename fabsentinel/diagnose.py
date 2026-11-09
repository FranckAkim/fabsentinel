"""Root-cause diagnosis for flagged wafers.

Two diagnosers share one interface:

* SignatureDiagnoser - needs **no labelled fault history**. Each failure mode
  has a physics-based signature: which sensors should move, and which way.
  We collapse a wafer's standardised feature deviations into a signed
  per-sensor vector and rank failure modes by cosine similarity.
* LearnedDiagnoser - a random forest trained on labelled past excursions.
  Better once a fab has history; useless on day one.

Each diagnosis carries an OCAP (out-of-control action plan) entry: what the
on-shift technician should check first.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .features import sensor_of

# Signed expected response per sensor. +1 = rises, -1 = falls, magnitude = strength.
# "*.std" style variability is encoded with the _var suffix.
SIGNATURES: dict[str, dict[str, float]] = {
    "cf4_mfc_drift": {"cf4_sccm": -1.0, "oes_f_703nm": -0.8, "endpoint_s": +0.9, "throttle_valve_pct": -0.4},
    "rf_match_degradation": {"rf_refl_w": +1.0, "endpoint_s": +0.4, "oes_f_703nm": -0.3},
    "chamber_leak": {"throttle_valve_pct": -1.0, "pressure_mtorr": +0.6, "endpoint_s": +0.3},
    "he_backside_leak": {"he_backside_torr": -1.0, "esc_temp_c": +0.7, "he_backside_torr_var": +0.4},
    "chiller_fault": {"esc_temp_c": +1.0, "esc_temp_c_trend": +0.8},
    "throttle_valve_sticking": {"throttle_valve_pct_var": +1.0, "pressure_mtorr_var": +0.9},
}

OCAP: dict[str, str] = {
    "cf4_mfc_drift": "Run MFC rate-of-rise verification on CF4; compare to last calibration. Hold tool if flow error > 3%.",
    "rf_match_degradation": "Check match network tune positions and reflected-power trend; inspect RF cable and connectors.",
    "chamber_leak": "Perform chamber leak-up rate test; check lid O-ring and recent PM fittings.",
    "he_backside_leak": "Check He backside flow and ESC chucking voltage; inspect for wafer chipping or particles on ESC.",
    "chiller_fault": "Check chiller setpoint, coolant flow and level; verify ESC temperature against an independent sensor.",
    "throttle_valve_sticking": "Exercise the throttle valve, check controller PID tuning; schedule valve cleaning or replacement.",
}

NOISE_GATE = 2.5
FEATURE_ORDER = sorted({k for s in SIGNATURES.values() for k in s})


def sensor_vector(z: pd.Series) -> pd.Series:
    """Collapse feature z-scores into one signed number per signature dimension."""
    out = {k: 0.0 for k in FEATURE_ORDER}

    def pick(prefix_sensor: str, kind: str) -> float:
        idx = [f for f in z.index if sensor_of(f) == prefix_sensor and f.endswith(kind) and not f.startswith("stab.")]
        if not idx:
            return 0.0
        vals = z[idx]
        return float(vals.iloc[np.argmax(np.abs(vals.to_numpy()))])

    for key in FEATURE_ORDER:
        if key == "endpoint_s":
            out[key] = float(z.get("main.endpoint_s", 0.0))
        elif key.endswith("_var"):
            out[key] = max(pick(key[:-4], ".std"), 0.0)
        elif key.endswith("_trend"):
            out[key] = pick(key[:-6], ".slope")
        else:
            # A level shift can show up as a moved mean or as a within-step ramp
            cands = [pick(key, ".mean"), pick(key, ".slope")]
            out[key] = max(cands, key=abs)
    v = pd.Series(out)
    # Ignore deviations inside normal noise (|z| < 2.5), then damp huge z-scores
    # so one exploding feature does not dominate the direction
    v = v.where(v.abs() >= NOISE_GATE, 0.0)
    return np.sign(v) * np.log1p(np.abs(v))


@dataclass
class Diagnosis:
    ranked: list[tuple[str, float]]  # (fault, confidence 0-1), best first
    top_sensors: list[tuple[str, float]]  # (feature, z)

    @property
    def cause(self) -> str:
        return self.ranked[0][0]

    @property
    def confidence(self) -> float:
        return self.ranked[0][1]

    @property
    def action(self) -> str:
        return OCAP[self.cause]


def _top_features(z: pd.Series, n: int = 4) -> list[tuple[str, float]]:
    s = z.reindex(z.abs().sort_values(ascending=False).index)[:n]
    return [(k, float(v)) for k, v in s.items()]


class SignatureDiagnoser:
    def __init__(self):
        self.S = pd.DataFrame(SIGNATURES).T.reindex(columns=FEATURE_ORDER).fillna(0.0)

    def diagnose(self, z: pd.Series) -> Diagnosis:
        v = sensor_vector(z).reindex(FEATURE_ORDER).to_numpy()
        S = self.S.to_numpy()
        sims = S @ v / (np.linalg.norm(S, axis=1) * (np.linalg.norm(v) + 1e-9))
        sims = np.clip(sims, 0, None)
        p = np.exp(6 * sims)
        p = p / p.sum()
        order = np.argsort(-p)
        return Diagnosis([(self.S.index[i], float(p[i])) for i in order], _top_features(z))


class LearnedDiagnoser:
    def __init__(self, seed: int = 0):
        self.clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=-1)

    def fit(self, Z: pd.DataFrame, labels: list[str]) -> LearnedDiagnoser:
        self.columns_ = list(Z.columns)
        self.clf.fit(Z.to_numpy(), labels)
        return self

    def diagnose(self, z: pd.Series) -> Diagnosis:
        p = self.clf.predict_proba(z.reindex(self.columns_).to_numpy()[None, :])[0]
        order = np.argsort(-p)
        return Diagnosis([(self.clf.classes_[i], float(p[i])) for i in order], _top_features(z))
