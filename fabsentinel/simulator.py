"""Plasma etch chamber simulator.

Generates 1 Hz sensor traces for a three-step oxide etch recipe, with
wafer-to-wafer noise, chamber seasoning drift between wet cleans, and
injectable equipment faults whose physical side effects couple across
sensors the way they do on a real tool (e.g. a CF4 mass-flow-controller
drift lowers the fluorine emission line and stretches the endpoint time).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SENSORS = [
    "pressure_mtorr",
    "throttle_valve_pct",
    "rf_fwd_w",
    "rf_refl_w",
    "cf4_sccm",
    "o2_sccm",
    "ar_sccm",
    "esc_temp_c",
    "he_backside_torr",
    "oes_f_703nm",
]

# Recipe: step name -> setpoints. Main etch ends on optical endpoint.
RECIPE = {
    1: dict(name="stabilize", seconds=20, pressure=50, rf=0, cf4=100, o2=10, ar=200),
    2: dict(name="main_etch", seconds=None, pressure=30, rf=800, cf4=100, o2=10, ar=200),
    3: dict(name="over_etch", seconds=15, pressure=30, rf=600, cf4=50, o2=20, ar=200),
}
NOMINAL_ENDPOINT_S = 60.0

FAULTS = {
    "cf4_mfc_drift": "CF4 mass flow controller delivering low",
    "rf_match_degradation": "RF matching network drifting out of tune",
    "chamber_leak": "Vacuum leak into the process chamber",
    "he_backside_leak": "Helium backside cooling leak / wafer not clamped",
    "chiller_fault": "ESC chiller losing cooling capacity",
    "throttle_valve_sticking": "Throttle valve sticking / hunting",
}


@dataclass
class Wafer:
    wafer_id: str
    trace: pd.DataFrame  # columns: time_s, step, *SENSORS
    fault: str = "normal"
    severity: float = 0.0
    wafers_since_clean: int = 0
    meta: dict = field(default_factory=dict)


class EtchChamberSimulator:
    """Deterministic for a given seed."""

    def __init__(self, seed: int = 0, clean_interval: int = 150):
        self.rng = np.random.default_rng(seed)
        self.clean_interval = clean_interval
        self._count = 0

    def run_wafer(self, fault: str = "normal", severity: float = 1.0, wafer_id: str | None = None) -> Wafer:
        if fault != "normal" and fault not in FAULTS:
            raise ValueError(f"unknown fault {fault!r}; choose from {sorted(FAULTS)}")
        r = self.rng
        self._count += 1
        since_clean = self._count % self.clean_interval
        sev = severity if fault != "normal" else 0.0

        # Wafer-to-wafer variation (incoming film thickness, small tool noise)
        film = r.normal(1.0, 0.012)
        seasoning = 1.0 - 0.02 * since_clean / self.clean_interval  # polymer build-up slightly lowers OES
        flow_bias = {g: r.normal(0, 0.004) for g in ("cf4", "o2", "ar")}
        temp_base = 60.0 + r.normal(0, 0.15)
        he_base = 10.0 + r.normal(0, 0.05)
        refl_frac = 0.012 + r.normal(0, 0.0015)

        # Fault parameters
        cf4_scale = 1.0 - (0.03 * sev if fault == "cf4_mfc_drift" else 0.0)
        refl_add = 2.5 * sev if fault == "rf_match_degradation" else 0.0
        leak = 0.5 * sev if fault == "chamber_leak" else 0.0  # mTorr-equivalent inflow
        he_drop = 0.35 * sev if fault == "he_backside_leak" else 0.0
        chill_rate = 0.015 * sev if fault == "chiller_fault" else 0.0  # degC per second under RF
        valve_osc = 0.6 * sev if fault == "throttle_valve_sticking" else 0.0

        # Endpoint time depends on etch rate ~ fluorine density ~ CF4 flow, delivered RF, film thickness
        delivered_frac = 1.0 - (refl_frac + refl_add / 800.0)
        rate = cf4_scale**0.8 * delivered_frac / (1.0 - 0.012) * (1.0 - 0.15 * leak / 30.0)
        endpoint = NOMINAL_ENDPOINT_S * film / rate + r.normal(0, 0.6)
        durations = {1: 20, 2: int(round(endpoint)), 3: 15}

        rows = []
        t = 0
        temp = temp_base
        for step in (1, 2, 3):
            sp = RECIPE[step]
            n = durations[step]
            for k in range(n):
                frac = k / max(n - 1, 1)
                rf_on = sp["rf"] > 0
                cf4 = sp["cf4"] * (1 + flow_bias["cf4"]) * cf4_scale + r.normal(0, 0.3)
                o2 = sp["o2"] * (1 + flow_bias["o2"]) + r.normal(0, 0.05)
                ar = sp["ar"] * (1 + flow_bias["ar"]) + r.normal(0, 0.5)
                # Throttle valve opens wider when gas load is lower, closes when a leak adds load
                total_flow = cf4 + o2 + ar
                valve = 22.0 * (total_flow / 310.0) * (50.0 / sp["pressure"]) ** 0.9 - 0.9 * leak
                valve += r.normal(0, 0.15)
                pressure = sp["pressure"] + r.normal(0, 0.12) + 0.25 * leak
                if valve_osc:
                    phase = np.sin(2 * np.pi * t / 7.0)
                    valve += valve_osc * phase
                    pressure += 0.45 * valve_osc * phase
                rf_fwd = sp["rf"] + (r.normal(0, 2.0) if rf_on else 0.0)
                rf_refl = (rf_fwd * refl_frac + (refl_add * (0.7 + 0.6 * frac)) if rf_on else 0.0) + (r.normal(0, 0.25) if rf_on else 0.0)
                he = he_base - he_drop * (0.6 + 0.4 * r.random()) + r.normal(0, 0.04 + 0.08 * (he_drop > 0))
                # ESC temperature: RF heat load, chiller removes it, poor He contact raises wafer temp
                heat = (rf_fwd / 800.0) * (0.03 + chill_rate)
                temp += heat - 0.03 * (temp - temp_base) * (1 if rf_on else 1.5)
                temp_obs = temp + 0.35 * he_drop + r.normal(0, 0.08 + 0.1 * (he_drop > 0))
                # Optical emission (F line): proportional to fluorine density while film remains
                if step == 2:
                    clearing = 1.0 / (1.0 + np.exp((k - (n - 4)) / 1.2))
                    oes = 1000 * cf4_scale * delivered_frac * seasoning * (1 - 0.01 * leak) * (0.55 + 0.45 * clearing)
                elif step == 3:
                    oes = 1000 * cf4_scale * delivered_frac * seasoning * 0.55 * (sp["cf4"] / 100) ** 0.5
                else:
                    oes = 0.0
                if oes:
                    oes += r.normal(0, 4.0)
                rows.append((t, step, pressure, valve, rf_fwd, rf_refl, cf4, o2, ar, temp_obs, he, oes))
                t += 1

        trace = pd.DataFrame(rows, columns=["time_s", "step", *SENSORS])
        return Wafer(
            wafer_id=wafer_id or f"W{self._count:05d}",
            trace=trace,
            fault=fault,
            severity=float(sev),
            wafers_since_clean=since_clean,
            meta={"endpoint_s": durations[2]},
        )

    def run_lot(self, n: int, fault_rate: float = 0.0, severity_range=(0.3, 1.5), faults=None) -> list[Wafer]:
        """Run n wafers; each faults with probability fault_rate."""
        faults = list(faults or FAULTS)
        out = []
        for _ in range(n):
            if self.rng.random() < fault_rate:
                f = faults[self.rng.integers(len(faults))]
                out.append(self.run_wafer(f, float(self.rng.uniform(*severity_range))))
            else:
                out.append(self.run_wafer())
        return out
