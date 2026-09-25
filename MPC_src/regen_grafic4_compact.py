"""Compact version of grafic4 — 3 scenarios (one per difficulty), y vs ref + u(t)."""
import os, sys, json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
model = HybridControllerModel(
    d_seq=d_seq, d_static=d_static,
    d_model=int(cfg["d_model"]), n=n, m=m,
    tx_layers=int(cfg["tx_layers"]), tx_heads=int(cfg["tx_heads"]),
    dropout=float(cfg["dropout"]))
model.load_state_dict(ckpt["model_state"]); model.eval()

mean_P = torch.tensor(stats_log["mean_P"], dtype=torch.float32)
std_P  = torch.tensor(stats_log["std_P"],  dtype=torch.float32)
mean_Q = torch.tensor(stats_log["mean_Q"], dtype=torch.float32)
std_Q  = torch.tensor(stats_log["std_Q"],  dtype=torch.float32)
mean_R = torch.tensor(stats_log["mean_R"], dtype=torch.float32)
std_R  = torch.tensor(stats_log["std_R"],  dtype=torch.float32)

ds = MPCDataset(root_dir="mpc_pqr_dataset_streaming_debug_500_samples",
                drop_duplicate_seq_feature=(d_seq == 4))
ds.set_normalization_stats(stats_in.get("seq_mean"), stats_in.get("seq_std"),
                            stats_in.get("static_mean"), stats_in.get("static_std"))

# Pick 3 scenarios with biggest reference variation, one per difficulty
target_samples = {"easy": None, "medium": None, "hard": None}
candidates = {"easy": [], "medium": [], "hard": []}
for si, sd in enumerate(ds.sample_dirs):
    meta = json.load(open(sd / "meta.json"))
    diff = meta["scenario"]["difficulty"]
    if diff not in target_samples: continue
    sig = np.load(sd / "signals.npz")
    ref_range = float(sig["reference"].max() - sig["reference"].min())
    candidates[diff].append((si, ref_range))
for diff in target_samples:
    candidates[diff].sort(key=lambda x: -x[1])
    if candidates[diff]:
        target_samples[diff] = candidates[diff][0][0]

print(f"[scenarios] {target_samples}")

mpc_cache = {}
trajs = []

for diff_label in ["easy", "medium", "hard"]:
    si = target_samples[diff_label]
    if si is None: continue
    item = ds[si]
    meta = json.load(open(ds.sample_dirs[si] / "meta.json"))
    sig = np.load(ds.sample_dirs[si] / "signals.npz")
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
    ref_type = meta["scenario"]["reference_type"]

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

    key = (tuple(A_d.flatten().round(6)), horizon)
    if key not in mpc_cache:
        mpc_cache[key] = ParametricMPC(A_d, B_d, C_d, E_d, u_min, u_max, du_min, du_max, horizon)
    mpc = mpc_cache[key]
    sims = {}
    for name, (pd_, qd_, rd_) in [("baseline", (p_gt, q_gt, r_gt)),
                                    ("tfmr", (p_pred, q_pred, r_pred)),
                                    ("mint", (p_mint, q_mint, r_mint))]:
        rng_sim = np.random.default_rng(100000 + si)
        sims[name] = simulate_closed_loop(mpc, A_d, B_d, C_d, E_d, pd_, qd_, rd_,
                                          x_ref_full, reference, disturbance, x0,
                                          sim_steps, horizon, noise_std, rng_sim)
    trajs.append({"diff": diff_label, "ref_type": ref_type, "ref": reference,
                   "Ts": Ts, "u_min": u_min, "u_max": u_max, "sims": sims})
    print(f"  ok {diff_label}/{ref_type}")

# ============ Compact figure: 3 rows × 2 cols ============
diff_ro = {"easy": "ușor", "medium": "mediu", "hard": "dificil"}
ref_ro = {"step": "treaptă", "multi_step": "multi-treaptă", "sine": "sinus"}

fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=False)

for i, tr in enumerate(trajs):
    Ts = tr["Ts"]; ref = tr["ref"]; t = np.arange(len(ref)) * Ts
    diff = diff_ro.get(tr["diff"], tr["diff"])
    rt = ref_ro.get(tr["ref_type"], tr["ref_type"])

    # Output
    ax = axes[i, 0]
    ax.plot(t, ref, "k-", linewidth=2.0, alpha=0.7, label="referință r(t)")
    for name, color, lab in [("baseline", "tab:green", "Baseline"),
                              ("tfmr", "tab:blue", "Transformer"),
                              ("mint", "tab:red", "Transformer+MinT")]:
        s = tr["sims"][name]
        if s["feasible"]:
            ax.plot(np.arange(len(s["y"])) * Ts, s["y"], color=color,
                    linewidth=1.6, alpha=0.85, label=lab)
    ax.set_ylabel(f"{diff} / {rt}\n\nIeșire y(t)", fontsize=11, fontweight="bold")
    if i == 0:
        ax.legend(loc="best", fontsize=8, framealpha=0.9, ncol=2)
    if i == len(trajs) - 1:
        ax.set_xlabel("Timp [s]", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Control
    ax = axes[i, 1]
    for name, color in [("baseline", "tab:green"),
                         ("tfmr", "tab:blue"),
                         ("mint", "tab:red")]:
        s = tr["sims"][name]
        if s["feasible"]:
            ax.plot(np.arange(len(s["u"])) * Ts, s["u"], color=color,
                    linewidth=1.3, alpha=0.85)
    ax.axhline(tr["u_max"], color="gray", linestyle=":", alpha=0.6)
    ax.axhline(tr["u_min"], color="gray", linestyle=":", alpha=0.6)
    ax.set_ylabel("Comandă u(t)", fontsize=11)
    if i == len(trajs) - 1:
        ax.set_xlabel("Timp [s]", fontsize=11)
    ax.grid(True, alpha=0.3)

fig.suptitle("Performanță în buclă închisă — 3 scenarii (ușor / mediu / dificil)",
             fontsize=13, fontweight="bold", y=0.995)
fig.tight_layout()
fig.savefig(OUT / "traiectorii_3scenarii.pdf", bbox_inches="tight")
fig.savefig(OUT / "traiectorii_3scenarii.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"\n[ok] {OUT / 'traiectorii_3scenarii.pdf'}")
