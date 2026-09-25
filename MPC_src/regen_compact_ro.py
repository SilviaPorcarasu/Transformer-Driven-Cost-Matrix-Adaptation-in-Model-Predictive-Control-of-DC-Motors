"""Compact combined figure for the poster:
   Top: 1 trajectory + control input (most representative scenario)
   Bottom: boxplots IAE / final error / quality across all val samples
   Plus: combined table image
"""
import os, sys, json, csv
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_eval import HybridControllerModel, MPCDataset, log_diag_to_matrix, denormalize_log, mint_projection
from run_control_eval import ParametricMPC, simulate_closed_loop

RUN = Path("mpc_pqr_run_logspace_norm")
OUT = RUN / "report_pdf_ro"
OUT.mkdir(exist_ok=True)
ckpt = torch.load(RUN / "best.pt", map_location="cpu", weights_only=False)
cfg = ckpt["args"]
stats_in = ckpt.get("input_stats", {})
stats_log = ckpt.get("stats", {})

d_seq = int(cfg["d_seq"]); d_static = int(cfg["d_static"])
n = int(cfg["n"]); m = int(cfg["m"])
device = torch.device("cpu")

model = HybridControllerModel(
    d_seq=d_seq, d_static=d_static,
    d_model=int(cfg["d_model"]), n=n, m=m,
    tx_layers=int(cfg["tx_layers"]), tx_heads=int(cfg["tx_heads"]),
    dropout=float(cfg["dropout"])).to(device)
model.load_state_dict(ckpt["model_state"]); model.eval()

mean_P = torch.tensor(stats_log["mean_P"], dtype=torch.float32, device=device)
std_P  = torch.tensor(stats_log["std_P"],  dtype=torch.float32, device=device)
mean_Q = torch.tensor(stats_log["mean_Q"], dtype=torch.float32, device=device)
std_Q  = torch.tensor(stats_log["std_Q"],  dtype=torch.float32, device=device)
mean_R = torch.tensor(stats_log["mean_R"], dtype=torch.float32, device=device)
std_R  = torch.tensor(stats_log["std_R"],  dtype=torch.float32, device=device)

ds = MPCDataset(root_dir="mpc_pqr_dataset_streaming_debug_500_samples",
                drop_duplicate_seq_feature=(d_seq == 4))
ds.set_normalization_stats(stats_in.get("seq_mean"), stats_in.get("seq_std"),
                            stats_in.get("static_mean"), stats_in.get("static_std"))

# Load CSV with all per-sample metrics (for boxplots)
rows_csv = []
with open(RUN / "eval_control" / "per_sample_control.csv") as f:
    for r in csv.DictReader(f):
        rows_csv.append(r)


def f_or_nan(s):
    try:
        v = float(s); return v if np.isfinite(v) else np.nan
    except: return np.nan


# Pick ONE representative scenario (medium difficulty, multi-step → most dramatic)
import json as _j
chosen = None
for si, sd in enumerate(ds.sample_dirs):
    meta = _j.load(open(sd / "meta.json"))
    if (meta["scenario"]["difficulty"] == "medium"
            and meta["scenario"]["reference_type"] == "multi_step"):
        chosen = si; break
if chosen is None:
    chosen = 23  # fallback

print(f"[traj] chosen sample_idx={chosen}")

item = ds[chosen]
meta = _j.load(open(ds.sample_dirs[chosen] / "meta.json"))
sig = np.load(ds.sample_dirs[chosen] / "signals.npz")
A_d = np.array(meta["model"]["A_d"]); B_d = np.array(meta["model"]["B_d"])
C_d = np.array(meta["model"]["C_d"]); E_d = np.array(meta["model"]["E_d"])
Ts = float(meta["model"]["Ts"])
horizon = int(meta["constraints"]["horizon"])
sim_steps = int(meta["constraints"]["sim_steps"])
u_min = float(meta["constraints"]["u_min"]); u_max = float(meta["constraints"]["u_max"])
du_min = float(meta["constraints"]["du_min"]); du_max = float(meta["constraints"]["du_max"])
reference = np.array(sig["reference"]); disturbance = np.array(sig["disturbance"])
x_ref_full = np.array(sig["x_ref"])
x0 = np.array(meta["scenario"]["x0"])
noise_std = float(meta["scenario"]["measurement_noise_std"])

p_gt = np.array(meta["target"]["P_diag"])
q_gt = np.array(meta["target"]["Q_diag"])
r_gt = np.array(meta["target"]["R_diag"])

with torch.no_grad():
    ts = item["tokens_seq"].unsqueeze(0); tst = item["tokens_static"].unsqueeze(0)
    p_log_n, q_log_n, r_log_n, _ = model(ts, tst)
    p_log = denormalize_log(p_log_n, mean_P, std_P)
    q_log = denormalize_log(q_log_n, mean_Q, std_Q)
    r_log = denormalize_log(r_log_n, mean_R, std_R)
    P_pred = log_diag_to_matrix(p_log)
    Q_pred = log_diag_to_matrix(q_log)
    R_pred = log_diag_to_matrix(r_log)
p_pred = torch.pow(10.0, p_log).squeeze(0).cpu().numpy()
q_pred = torch.pow(10.0, q_log).squeeze(0).cpu().numpy()
r_pred = torch.pow(10.0, r_log).squeeze(0).cpu().numpy()

A_t = torch.tensor(A_d[None]).float(); B_t = torch.tensor(B_d[None]).float()
P_proj, Q_proj, R_proj = mint_projection(P_pred.cpu(), Q_pred.cpu(), R_pred.cpu(),
                                          A_t, B_t, n_iter=3)
p_mint = np.maximum(np.diag(P_proj.squeeze(0).numpy()), 1e-8)
q_mint = np.maximum(np.diag(Q_proj.squeeze(0).numpy()), 1e-8)
r_mint = np.maximum(np.diag(R_proj.squeeze(0).numpy()), 1e-8)

mpc = ParametricMPC(A_d, B_d, C_d, E_d, u_min, u_max, du_min, du_max, horizon)
sims = {}
for name, (pd_, qd_, rd_) in [
    ("baseline", (p_gt, q_gt, r_gt)),
    ("tfmr", (p_pred, q_pred, r_pred)),
    ("mint", (p_mint, q_mint, r_mint))
]:
    rng_sim = np.random.default_rng(100000 + chosen)
    sims[name] = simulate_closed_loop(mpc, A_d, B_d, C_d, E_d, pd_, qd_, rd_,
                                      x_ref_full, reference, disturbance, x0,
                                      sim_steps, horizon, noise_std, rng_sim)

# ==================== Compact combined figure ====================
fig = plt.figure(figsize=(13, 9))
gs = GridSpec(2, 3, figure=fig, height_ratios=[1.2, 1.0],
              hspace=0.35, wspace=0.25)

# Top-left: trajectory ref vs y
ax_y = fig.add_subplot(gs[0, 0:2])
t = np.arange(len(reference)) * Ts
ax_y.plot(t, reference, "k-", linewidth=2.5, alpha=0.7, label="referință r(t)")
for name, color, lab in [("baseline", "tab:green", "Baseline"),
                          ("tfmr", "tab:blue", "Transformer"),
                          ("mint", "tab:red", "Transformer+MinT")]:
    s = sims[name]
    if s["feasible"]:
        ax_y.plot(np.arange(len(s["y"])) * Ts, s["y"], color=color,
                  linewidth=2.0, alpha=0.85, label=lab)
ax_y.set_xlabel("Timp [s]", fontsize=11)
ax_y.set_ylabel("Ieșire y(t)", fontsize=11)
ax_y.set_title("(a) Urmărire referință — scenariu reprezentativ (mediu, multi-treaptă)",
                fontsize=12, fontweight="bold")
ax_y.legend(loc="best", fontsize=9, framealpha=0.9)
ax_y.grid(True, alpha=0.3)

# Top-right: control input u(t)
ax_u = fig.add_subplot(gs[0, 2])
for name, color, lab in [("baseline", "tab:green", "Baseline"),
                          ("tfmr", "tab:blue", "Transformer"),
                          ("mint", "tab:red", "Tfmr+MinT")]:
    s = sims[name]
    if s["feasible"]:
        ax_u.plot(np.arange(len(s["u"])) * Ts, s["u"], color=color,
                  linewidth=1.4, alpha=0.85, label=lab)
ax_u.axhline(u_max, color="gray", linestyle=":", alpha=0.6)
ax_u.axhline(u_min, color="gray", linestyle=":", alpha=0.6)
ax_u.set_xlabel("Timp [s]", fontsize=11)
ax_u.set_ylabel("Comandă u(t)", fontsize=11)
ax_u.set_title("(b) Comandă", fontsize=12, fontweight="bold")
ax_u.legend(loc="best", fontsize=8, framealpha=0.9)
ax_u.grid(True, alpha=0.3)

# Bottom row: 3 bar charts (median with 25-75 percentile error bars)
variants = [("baseline", "Baseline"), ("tfmr", "Transformer"), ("mint", "Tfmr+MinT")]
colors_v = {"baseline": "tab:green", "tfmr": "tab:blue", "mint": "tab:red"}

for col, (mk, title, logy) in enumerate([
    ("iae_norm", "Integrala erorii absolute", False),
    ("final_error_norm", "Eroarea finală", False),
    ("quality_score", "Scor calitate", False),
]):
    ax = fig.add_subplot(gs[1, col])
    medians, p25s, p75s, labels, cs = [], [], [], [], []
    for v, lab in variants:
        vals = [f_or_nan(r.get(f"{v}_{mk}")) for r in rows_csv
                if r.get(f"{v}_feasible") == "True"]
        vals = np.array([x for x in vals if np.isfinite(x)])
        med = float(np.median(vals)); p25 = float(np.percentile(vals, 25))
        p75 = float(np.percentile(vals, 75))
        medians.append(med); p25s.append(p25); p75s.append(p75)
        labels.append(lab); cs.append(colors_v[v])

    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, medians, color=cs, alpha=0.85, edgecolor="black",
                   linewidth=1.2, width=0.65)

    # Annotate values on top of bars
    for x, m in zip(x_pos, medians):
        ax.text(x, m, f"{m:.3f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=10)
    ax.tick_params(axis="x", rotation=10)

    sub = "(c)" if col == 0 else ("(d)" if col == 1 else "(e)")
    ax.set_title(f"{sub} {title}", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    if mk == "quality_score":
        ax.set_ylim(0, 1.1)
    else:
        ymax = max(medians) * 1.30
        ax.set_ylim(0, ymax)

fig.suptitle("Performanța în buclă închisă — traiectorii (sus) și valori mediane (jos)",
              fontsize=13, fontweight="bold", y=0.995)

fig.savefig(OUT / "rezultate_compact.pdf", bbox_inches="tight")
fig.savefig(OUT / "rezultate_compact.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"[ok] {OUT / 'rezultate_compact.pdf'}")

# ==================== Combined table ====================
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.axis("off")
fig.suptitle("Tabel — Performanța în buclă închisă",
             fontsize=13, fontweight="bold", y=0.99)

headers = ["Metrică", "Baseline", "Transformer", "Transformer + MinT"]
data = [
    ["Integrala erorii absolute",      "0.258",  "0.246",  "0.312"],
    ["Eroarea finală",                  "0.028",  "0.028",  "0.062"],
    ["Suprareglare",                    "0.000",  "0.020",  "0.000"],
    ["Saturare",                        "0.025",  "0.050",  "0.000"],
    ["Energia de control",              "153.5",  "174.6",  "88.2"],
    ["Variația de control",             "2.59",   "4.36",   "2.41"],
    ["Reziduul Riccati",                "—",      "2011.7", "212.0  (×9.5 ↓)"],
    ["Fezabilitate",                    "100 %",  "100 %",  "100 %"],
]
tbl = ax.table(cellText=data, colLabels=headers, loc="center",
               cellLoc="center", colLoc="center",
               colWidths=[0.30, 0.22, 0.22, 0.26])
tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1.0, 1.7)

best_cells = {
    (1, 2),
    (2, 1), (2, 2),
    (3, 3),
    (4, 3),
    (5, 3),
    (6, 3),
    (7, 3),
}
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#2E5A88")
        cell.set_text_props(color="white", weight="bold")
        cell.set_height(0.10)
    else:
        cell.set_height(0.075)
        if r % 2 == 0:
            cell.set_facecolor("#F0F4F8")
        if (r, c) in best_cells:
            cell.set_text_props(weight="bold")

fig.tight_layout()
fig.savefig(OUT / "tabel_combinat.pdf", bbox_inches="tight")
fig.savefig(OUT / "tabel_combinat.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"[ok] {OUT / 'tabel_combinat.pdf'}")
