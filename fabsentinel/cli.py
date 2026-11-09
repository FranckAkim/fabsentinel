"""Command line: `fabsentinel benchmark` and `fabsentinel diagnose`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .diagnose import SignatureDiagnoser
from .evaluate import run_benchmark, summary_lines
from .features import feature_table
from .model import MSPCDetector
from .simulator import FAULTS, EtchChamberSimulator


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="fabsentinel", description="Fault detection and root-cause diagnosis for etch chambers.")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("benchmark", help="simulate production, score it, write metrics and an HTML report")
    b.add_argument("--wafers", type=int, default=1000)
    b.add_argument("--fault-rate", type=float, default=0.2)
    b.add_argument("--seed", type=int, default=7)
    b.add_argument("--out", type=Path, default=Path("out"))

    d = sub.add_parser("diagnose", help="inject one fault and print the diagnosis")
    d.add_argument("fault", choices=sorted(FAULTS))
    d.add_argument("--severity", type=float, default=1.0)
    d.add_argument("--seed", type=int, default=7)

    a = p.parse_args(argv)
    if a.cmd == "benchmark":
        from .report import build_report

        res = run_benchmark(n_test=a.wafers, fault_rate=a.fault_rate, seed=a.seed)
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "metrics.json").write_text(json.dumps(res.metrics, indent=2))
        (a.out / "report.html").write_text(build_report(res))
        print("\n".join(summary_lines(res.metrics)))
        print(f"\nWrote {a.out / 'metrics.json'} and {a.out / 'report.html'}")
        return 0

    sim = EtchChamberSimulator(seed=a.seed)
    det = MSPCDetector().fit(feature_table(sim.run_lot(200)))
    wafer = sim.run_wafer(a.fault, a.severity)
    score = det.score(feature_table([wafer]))[0]
    print(f"Wafer {wafer.wafer_id}: injected {a.fault} at severity {a.severity}")
    print(f"T2 {score.t2:.1f} (limit {score.t2_limit:.1f})   SPE {score.spe:.1f} (limit {score.spe_limit:.1f})")
    if not score.flagged:
        print("Within control limits: no alarm.")
        return 0
    dx = SignatureDiagnoser().diagnose(score.z)
    print("Likely root causes:")
    for cause, conf in dx.ranked[:3]:
        print(f"  {conf:5.0%}  {cause}")
    print("Strongest deviations: " + ", ".join(f"{f} {z:+.1f}σ" for f, z in dx.top_sensors))
    print(f"First action: {dx.action}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
