"""Riccati coherence histogram with clean Romanian labels for poster."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = Path("mpc_pqr_run_logspace_norm")
OUT = RUN / "report_pdf_ro"
OUT.mkdir(exist_ok=True)
ctrl_csv = RUN / "eval_control" / "per_sample_control.csv"

ric_before, ric_after = [], []
with open(ctrl_csv) as f:
    for r in csv.DictReader(f):
        try:
            b = float(r["ric_before_mint"]); a = float(r["ric_after_mint"])
            if np.isfinite(b): ric_before.append(b)
            if np.isfinite(a): ric_after.append(a)
        except (KeyError, ValueError):
            continue
ric_before = np.array(ric_before); ric_after = np.array(ric_after)

fig, ax = plt.subplots(figsize=(9, 5.5))
bins = np.logspace(np.log10(max(ric_after.min(), 1e-3)),
                   np.log10(max(ric_before.max(), 1e-2)), 40)
ax.hist(ric_before, bins=bins, alpha=0.65, label="înainte de MinT",
        color="steelblue", edgecolor="white")
ax.hist(ric_after, bins=bins, alpha=0.7, label="după MinT",
        color="coral", edgecolor="white")
ax.set_xlabel("Reziduul Riccati", fontsize=13)
ax.set_ylabel("Număr de scenarii", fontsize=13)
ax.set_title("Coerență fizică — înainte vs. după MinT",
             fontsize=14, fontweight="bold")
ax.set_xscale("log")
ax.legend(fontsize=12, loc="best", framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.tick_params(axis="both", labelsize=11)

fig.tight_layout()
fig.savefig(OUT / "riccati_coherence.pdf", bbox_inches="tight")
fig.savefig(OUT / "riccati_coherence.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"[ok] {OUT / 'riccati_coherence.pdf'}")
