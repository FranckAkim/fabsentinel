"""Multivariate statistical process control (MSPC) fault detector.

Fit on known-good ("golden") wafers:
  * standardise every FDC feature against the golden population
  * PCA keeps the components explaining `variance` of golden variation
  * Hotelling's T^2 measures how far a wafer moves *within* that normal pattern
  * SPE (Q statistic) measures how much it breaks the pattern altogether
Control limits come from the F distribution (T^2) and Box's chi-square
approximation (SPE), at the chosen confidence level.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class WaferScore:
    t2: float
    spe: float
    t2_limit: float
    spe_limit: float
    z: pd.Series  # standardised deviation of each feature from golden

    @property
    def flagged(self) -> bool:
        return self.t2 > self.t2_limit or self.spe > self.spe_limit

    @property
    def severity_index(self) -> float:
        """Combined distance in units of control limit (1.0 = on the limit)."""
        return max(self.t2 / self.t2_limit, self.spe / self.spe_limit)


class MSPCDetector:
    def __init__(self, variance: float = 0.80, confidence: float = 0.995):
        self.variance = variance
        self.confidence = confidence

    def fit(self, X: pd.DataFrame) -> MSPCDetector:
        std = X.std(ddof=1)
        keep = std > 1e-9  # drop features that never move (e.g. RF power during stabilise)
        self.columns_ = list(X.columns[keep])
        Xk = X[self.columns_]
        self.mean_ = Xk.mean()
        self.std_ = Xk.std(ddof=1)
        Z = ((Xk - self.mean_) / self.std_).to_numpy()
        n = Z.shape[0]
        U, S, Vt = np.linalg.svd(Z, full_matrices=False)
        eig = S**2 / (n - 1)
        cum = np.cumsum(eig) / eig.sum()
        k = int(np.searchsorted(cum, self.variance) + 1)
        self.k_ = k
        self.P_ = Vt[:k].T
        self.eig_ = eig[:k]
        # T^2 limit (F distribution, Phase II)
        alpha = self.confidence
        self.t2_limit_ = k * (n - 1) * (n + 1) / (n * (n - k)) * stats.f.ppf(alpha, k, n - k)
        # SPE limit: in-sample residuals are optimistically small, so estimate the
        # SPE distribution from 5-fold out-of-fold residuals, then fit Box's approximation.
        spe = self._oof_spe(Z, k)
        m, v = spe.mean(), spe.var(ddof=1)
        g, h = v / (2 * m), 2 * m * m / v
        self.spe_limit_ = g * stats.chi2.ppf(alpha, h)
        return self

    @staticmethod
    def _oof_spe(Z: np.ndarray, k: int, folds: int = 5) -> np.ndarray:
        idx = np.arange(Z.shape[0])
        out = np.empty(Z.shape[0])
        for f in range(folds):
            test = idx % folds == f
            Ztr = Z[~test]
            mu, sd = Ztr.mean(0), Ztr.std(0, ddof=1)
            sd[sd < 1e-9] = 1.0
            Ztr = (Ztr - mu) / sd
            P = np.linalg.svd(Ztr, full_matrices=False)[2][:k].T
            Zte = (Z[test] - mu) / sd
            R = Zte - Zte @ P @ P.T
            out[test] = (R**2).sum(axis=1)
        return out

    def _spe(self, Z: np.ndarray) -> np.ndarray:
        R = Z - Z @ self.P_ @ self.P_.T
        return (R**2).sum(axis=1)

    def standardise(self, X: pd.DataFrame) -> pd.DataFrame:
        return (X[self.columns_] - self.mean_) / self.std_

    def score(self, X: pd.DataFrame) -> list[WaferScore]:
        Zdf = self.standardise(X)
        Z = Zdf.to_numpy()
        T = Z @ self.P_
        t2 = (T**2 / self.eig_).sum(axis=1)
        spe = self._spe(Z)
        return [WaferScore(float(t2[i]), float(spe[i]), float(self.t2_limit_), float(self.spe_limit_), Zdf.iloc[i]) for i in range(len(X))]
