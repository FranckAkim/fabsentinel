"""End-to-end benchmark on simulated production.

1. Fit the detector on golden wafers.
2. Train the learned diagnoser on a separate labelled excursion history.
3. Score an unseen production run (mostly good wafers, some faulted ones at random severity).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .diagnose import LearnedDiagnoser, SignatureDiagnoser
from .features import feature_table
from .model import MSPCDetector
from .simulator import FAULTS, EtchChamberSimulator, Wafer

BANDS = [(0.3, 0.6, "low"), (0.6, 1.0, "medium"), (1.0, 1.51, "high")]


@dataclass
class RunResult:
    wafers: list[Wafer]
    table: pd.DataFrame  # one row per test wafer
    metrics: dict
    detector: MSPCDetector
    golden: list[Wafer] = field(repr=False, default_factory=list)


def run_benchmark(
    n_golden: int = 200,
    n_history: int = 300,
    n_test: int = 1000,
    fault_rate: float = 0.2,
    seed: int = 7,
) -> RunResult:
    golden = EtchChamberSimulator(seed=seed).run_lot(n_golden)
    det = MSPCDetector().fit(feature_table(golden))

    hist_sim = EtchChamberSimulator(seed=seed + 1)
    history = hist_sim.run_lot(n_history, fault_rate=1.0)
    learned = LearnedDiagnoser(seed=seed).fit(det.standardise(feature_table(history)), [w.fault for w in history])
    sig = SignatureDiagnoser()

    test = EtchChamberSimulator(seed=seed + 2).run_lot(n_test, fault_rate=fault_rate)
    scores = det.score(feature_table(test))

    rows = []
    for w, s in zip(test, scores, strict=True):
        row = dict(
            wafer_id=w.wafer_id,
            true_fault=w.fault,
            severity=round(w.severity, 2),
            t2=s.t2,
            spe=s.spe,
            index=s.severity_index,
            flagged=s.flagged,
        )
        if s.flagged:
            ds, dl = sig.diagnose(s.z), learned.diagnose(s.z)
            row.update(
                sig_cause=ds.cause,
                sig_conf=ds.confidence,
                sig_top2=[c for c, _ in ds.ranked[:2]],
                learned_cause=dl.cause,
                learned_conf=dl.confidence,
                top_features=ds.top_sensors,
                action=ds.action,
            )
        rows.append(row)
    table = pd.DataFrame(rows)
    return RunResult(test, table, compute_metrics(table), det, golden)


def compute_metrics(t: pd.DataFrame) -> dict:
    normal = t[t.true_fault == "normal"]
    faulty = t[t.true_fault != "normal"]
    detected = faulty[faulty.flagged]
    m = {
        "wafers": len(t),
        "normal_wafers": len(normal),
        "faulty_wafers": len(faulty),
        "false_alarm_rate": float(normal.flagged.mean()),
        "detection_rate": float(faulty.flagged.mean()),
        "precision": float(len(detected) / max(t.flagged.sum(), 1)),
        "detection_by_severity": {},
        "diagnosis_top1_signature": float((detected.sig_cause == detected.true_fault).mean()),
        "diagnosis_top2_signature": float(detected.apply(lambda r: r.true_fault in r.sig_top2, axis=1).mean()),
        "diagnosis_top1_learned": float((detected.learned_cause == detected.true_fault).mean()),
        "per_fault": {},
    }
    for lo, hi, name in BANDS:
        b = faulty[(faulty.severity >= lo) & (faulty.severity < hi)]
        m["detection_by_severity"][name] = {
            "range": f"{lo}-{min(hi, 1.5)}",
            "n": len(b),
            "rate": float(b.flagged.mean()) if len(b) else None,
        }
    for f in FAULTS:
        ff = faulty[faulty.true_fault == f]
        dd = ff[ff.flagged]
        m["per_fault"][f] = {
            "n": len(ff),
            "detected": float(ff.flagged.mean()) if len(ff) else None,
            "signature_top1": float((dd.sig_cause == f).mean()) if len(dd) else None,
            "learned_top1": float((dd.learned_cause == f).mean()) if len(dd) else None,
        }
    m["confusion_signature"] = (
        pd.crosstab(detected.true_fault, detected.sig_cause).reindex(index=list(FAULTS), columns=list(FAULTS), fill_value=0).astype(int).to_dict()
        if len(detected)
        else {}
    )
    return m


def summary_lines(m: dict) -> list[str]:
    sev = m["detection_by_severity"]
    return [
        f"Wafers scored:            {m['wafers']} ({m['faulty_wafers']} with injected faults)",
        f"False alarm rate:         {m['false_alarm_rate']:.1%}",
        f"Fault detection rate:     {m['detection_rate']:.1%}  " + "  ".join(f"[{k} {v['rate']:.0%}]" for k, v in sev.items() if v["rate"] is not None),
        f"Alarm precision:          {m['precision']:.1%}",
        f"Root cause, signatures:   top-1 {m['diagnosis_top1_signature']:.1%}, top-2 {m['diagnosis_top2_signature']:.1%}  (no labelled history needed)",
        f"Root cause, learned (RF): top-1 {m['diagnosis_top1_learned']:.1%}  (trained on past labelled excursions)",
    ]


def severity_summary(m: dict) -> np.ndarray:  # pragma: no cover - convenience
    return np.array([v["rate"] for v in m["detection_by_severity"].values()])
