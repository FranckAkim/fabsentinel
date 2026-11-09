"""Self-contained HTML report: control chart, detection curve, fault traces, alarm table."""

from __future__ import annotations

import html
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .diagnose import SIGNATURES  # noqa: E402
from .evaluate import RunResult, summary_lines  # noqa: E402
from .simulator import FAULTS  # noqa: E402

INK, MUTED, GRID = "#1d2430", "#667085", "#e4e7ec"
COLORS = {
    "normal": "#b8c0cc",
    "cf4_mfc_drift": "#1f77b4",
    "rf_match_degradation": "#d62728",
    "chamber_leak": "#9467bd",
    "he_backside_leak": "#2ca02c",
    "chiller_fault": "#ff7f0e",
    "throttle_valve_sticking": "#8c564b",
}
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _svg(fig) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    s = buf.getvalue()
    return s[s.index("<svg") :]


def control_chart(t: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(10, 3.2))
    x = np.arange(len(t))
    y = t["index"].clip(upper=20).to_numpy()
    for f, c in COLORS.items():
        m = (t.true_fault == f).to_numpy()
        ax.scatter(x[m], y[m], s=9 if f == "normal" else 18, c=c, label=f.replace("_", " "), zorder=3 if f != "normal" else 2, linewidths=0)
    ax.axhline(1.0, color="#b42318", lw=1.2, ls="--")
    ax.text(len(t) * 1.005, 1.0, "control\nlimit", color="#b42318", va="center", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("distance / control limit")
    ax.set_xlabel("wafer (production order)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(ncol=4, fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.22))
    return _svg(fig)


def detection_curve(t: pd.DataFrame) -> str:
    f = t[t.true_fault != "normal"].copy()
    bins = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.51])
    f["bin"] = pd.cut(f.severity, bins, right=False)
    rate = f.groupby("bin", observed=False).flagged.mean()
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    mids = (bins[:-1] + bins[1:]) / 2
    ax.plot(mids, rate.to_numpy() * 100, marker="o", color=INK, lw=1.6)
    ax.set_ylim(0, 105)
    ax.set_xlabel("fault severity (1.0 = reference fault size)")
    ax.set_ylabel("% of faulted wafers caught")
    ax.grid(color=GRID, lw=0.6)
    return _svg(fig)


def fault_traces(res: RunResult) -> str:
    """For each failure mode: the signature's lead sensor, golden 5-95% band vs one caught wafer."""
    gold = pd.concat([w.trace.assign(w=i) for i, w in enumerate(res.golden[:120])])
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.2))
    t = res.table
    for ax, fault in zip(axes.ravel(), FAULTS, strict=True):
        lead = max(SIGNATURES[fault], key=lambda k: abs(SIGNATURES[fault][k]))
        sensor = {"endpoint_s": "oes_f_703nm"}.get(lead, lead.replace("_var", "").replace("_trend", ""))
        caught = t[(t.true_fault == fault) & t.flagged].sort_values("severity")
        if caught.empty:
            ax.set_visible(False)
            continue
        wid = caught.iloc[len(caught) // 2].wafer_id
        wafer = next(w for w in res.wafers if w.wafer_id == wid)
        band = gold.groupby("time_s")[sensor].quantile([0.05, 0.95]).unstack()
        n = min(len(band), len(wafer.trace) + 5)
        ax.fill_between(band.index[:n], band[0.05][:n], band[0.95][:n], color="#d0d5dd", label="golden 5-95%")
        ax.plot(wafer.trace.time_s, wafer.trace[sensor], color=COLORS[fault], lw=1.1, label=f"{wid} (sev {wafer.severity:.2f})")
        ax.set_title(f"{fault.replace('_', ' ')}\n{sensor}", fontsize=8.5, color=INK)
        lo, hi = np.nanpercentile(wafer.trace[sensor][wafer.trace.step >= 2], [1, 99])
        glo, ghi = np.nanpercentile(band[0.05][20:n], 1), np.nanpercentile(band[0.95][20:n], 99)
        pad = (max(hi, ghi) - min(lo, glo)) * 0.25 + 1e-6
        ax.set_ylim(min(lo, glo) - pad, max(hi, ghi) + pad)
        ax.set_xlim(15, n)
        ax.legend(fontsize=6.5, frameon=False, loc="best")
        ax.grid(color=GRID, lw=0.5)
    fig.tight_layout()
    return _svg(fig)


def alarm_table(t: pd.DataFrame, limit: int = 30) -> str:
    a = t[t.flagged].sort_values("index", ascending=False).head(limit)
    rows = []
    for _, r in a.iterrows():
        feats = ", ".join(f"{f} ({z:+.1f}σ)" for f, z in (r.top_features or [])[:3])
        ok = "✓" if r.sig_cause == r.true_fault else "✗"
        rows.append(
            f"<tr><td>{html.escape(r.wafer_id)}</td><td class=n>{r['index']:.1f}×</td>"
            f"<td><b>{html.escape(r.sig_cause.replace('_', ' '))}</b> <span class=m>{r.sig_conf:.0%}</span></td>"
            f"<td class=m>{html.escape(feats)}</td><td>{html.escape(r.action)}</td>"
            f"<td class=m>{html.escape(r.true_fault.replace('_', ' '))} {ok}</td></tr>"
        )
    return (
        "<table><thead><tr><th>Wafer</th><th>Distance</th><th>Likely root cause</th><th>Strongest deviations</th>"
        "<th>First action (OCAP)</th><th>Injected fault</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def build_report(res: RunResult) -> str:
    m = res.metrics
    t = res.table
    stats = "".join(f"<li>{html.escape(line)}</li>" for line in summary_lines(m))
    per_fault = "".join(
        f"<tr><td>{f.replace('_', ' ')}</td><td>{html.escape(FAULTS[f])}</td><td class=n>{v['n']}</td>"
        f"<td class=n>{v['detected']:.0%}</td><td class=n>{(v['signature_top1'] or 0):.0%}</td><td class=n>{(v['learned_top1'] or 0):.0%}</td></tr>"
        for f, v in m["per_fault"].items()
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FabSentinel Report</title>
<style>
:root{{color-scheme:light}} body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;color:{INK};background:#fff;margin:0}}
.w{{max-width:1080px;margin:0 auto;padding:28px max(16px,env(safe-area-inset-left,0px)) 48px max(16px,env(safe-area-inset-right,0px))}}
h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:17px;margin:28px 0 8px}} p.lead{{color:{MUTED};margin:0 0 16px;max-width:75ch}}
ul.k{{font-family:ui-monospace,Menlo,monospace;font-size:13px;background:#f6f7f9;border:1px solid {GRID};border-radius:6px;padding:12px 16px 12px 32px;line-height:1.7}}
.row{{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,2fr);gap:24px;align-items:start}}
@media(max-width:760px){{.row{{grid-template-columns:1fr}}}}
.fig{{overflow-x:auto}} .fig svg{{max-width:100%;height:auto}}
.tw{{overflow-x:auto}} table{{border-collapse:collapse;font-size:12.5px;width:100%}}
th,td{{text-align:left;padding:6px 8px;border-bottom:1px solid {GRID};vertical-align:top}} th{{color:{MUTED};font-weight:600}}
td.n{{text-align:right;font-variant-numeric:tabular-nums}} .m{{color:{MUTED}}}
footer{{color:{MUTED};font-size:12px;margin-top:28px}}
</style></head><body><div class="w">
<h1>FabSentinel: etch chamber excursion report</h1>
<p class="lead">Multivariate SPC (PCA Hotelling T² + SPE) trained on {len(res.golden)} golden wafers, scoring {m["wafers"]} production wafers.
Every alarm gets a ranked root cause and the first out-of-control action. Data is simulated: faults are injected at random severity so detection and diagnosis can be scored against ground truth.</p>
<ul class="k">{stats}</ul>
<h2>Control chart</h2><div class="fig">{control_chart(t)}</div>
<div class="row"><div><h2>Detection vs fault size</h2><div class="fig">{detection_curve(t)}</div></div>
<div><h2>By failure mode</h2><div class="tw"><table><thead><tr><th>Failure mode</th><th>Physical cause</th><th>n</th><th>Caught</th><th>Signature dx</th><th>Learned dx</th></tr></thead><tbody>{per_fault}</tbody></table></div></div></div>
<h2>What each fault looks like</h2><div class="fig">{fault_traces(res)}</div>
<h2>Top alarms</h2><div class="tw">{alarm_table(t)}</div>
<footer>Detector: {res.detector.k_} principal components over {len(res.detector.columns_)} FDC features, {res.detector.confidence:.1%} control limits. Generated by FabSentinel.</footer>
</div></body></html>"""
