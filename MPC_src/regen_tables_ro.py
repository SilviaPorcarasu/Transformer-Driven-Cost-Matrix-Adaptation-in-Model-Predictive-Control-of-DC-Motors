"""Generate Romanian tables as PDF/PNG images for the poster."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("mpc_pqr_run_logspace_norm/report_pdf_ro")
OUT.mkdir(exist_ok=True)


def make_table(title, headers, rows, filename, col_widths=None,
               figsize=(10, None), highlight_cells=None):
    nrows = len(rows) + 1
    if figsize[1] is None:
        figsize = (figsize[0], 0.6 * nrows + 1.0)
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.99)

    tbl = ax.table(
        cellText=rows, colLabels=headers, loc="center",
        cellLoc="center", colLoc="center",
        colWidths=col_widths,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.0, 1.7)

    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2E5A88")
            cell.set_text_props(color="white", weight="bold")
            cell.set_height(0.10)
        else:
            cell.set_height(0.075)
            if r % 2 == 0:
                cell.set_facecolor("#F0F4F8")
            if highlight_cells and (r, c) in highlight_cells:
                cell.set_facecolor("#FFE0B2")
                cell.set_text_props(weight="bold")

    fig.tight_layout()
    pdf = OUT / f"{filename}.pdf"
    png = OUT / f"{filename}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {pdf.name}")


# ─────────────────────────────────────────────────────────────
# TABEL 1 — Acuratețea modelului ML
# ─────────────────────────────────────────────────────────────
make_table(
    title="Tabel I — Acuratețea predicției matricelor P, Q, R (461 scenarii val.)",
    headers=["Matrice", "FRE mean", "FRE med", "FRE p95",
             "R² mean", "R² med", "R² p95"],
    rows=[
        ["P (2×2)", "1.53×10⁴", "11.27", "2.00×10⁴", "−3.62×10¹⁰", "−168.65", "0.68"],
        ["Q (2×2)", "100.74", "1.39", "321.90", "−5.04×10⁵", "−1.88", "0.64"],
        ["R (1×1)", "1.07", "0.99", "1.00", "−1.98×10⁸", "−1.87×10⁵", "−5.73"],
    ],
    filename="tabel1_ml_accuracy",
    figsize=(12, None),
)

# ─────────────────────────────────────────────────────────────
# TABEL 2 — Coerență Riccati
# ─────────────────────────────────────────────────────────────
make_table(
    title="Tabel II — Reziduul Riccati (norma Frobenius)",
    headers=["", "Mean", "Median", "p95"],
    rows=[
        ["Înainte de MinT", "6309.81", "2011.68", "26259.39"],
        ["După MinT", "908.22", "211.98", "4431.47"],
        ["Reducere", "×6.95", "×9.49", "×5.93"],
    ],
    filename="tabel2_riccati",
    figsize=(9, None),
    highlight_cells={(3, 1), (3, 2), (3, 3)},
)

# ─────────────────────────────────────────────────────────────
# TABEL 3 — Performanță în buclă închisă
# ─────────────────────────────────────────────────────────────
make_table(
    title="Tabel III — Performanță în buclă închisă (mediană / mean, 80 scenarii)",
    headers=["Metric", "Baseline (oracle)", "Transformer", "Transformer+MinT"],
    rows=[
        ["IAE (normalizat)",          "0.258 / 0.264", "0.246 / 0.237", "0.312 / 0.407"],
        ["ISE (normalizat)",          "0.189 / 0.205", "0.181 / 0.195", "0.259 / 0.307"],
        ["RMSE (normalizat)",         "0.397 / 0.389", "0.388 / 0.379", "0.464 / 0.469"],
        ["Eroare finală (norm.)",     "0.028 / 0.050", "0.028 / 0.033", "0.062 / 0.195"],
        ["Overshoot",                 "0.000 / 0.024", "0.020 / 0.046", "0.000 / 0.020"],
        ["Saturare",                  "0.025 / 0.043", "0.050 / 0.064", "0.000 / 0.020"],
        ["Timp stabilizare [s]",      "1.200 / 1.006", "1.200 / 1.030", "1.200 / 1.133"],
        ["Energie comandă",           "153.5 / 163.1", "174.7 / 184.3", "88.2 / 107.4"],
        ["Variație comandă",          "2.59 / 3.19",   "4.36 / 5.65",   "2.41 / 2.41"],
        ["Meta-cost",                 "3.42 / 3.42",   "3.62 / 3.59",   "3.83 / 3.92"],
        ["Scor calitate",             "0.962 / 0.957", "0.955 / 0.955", "0.946 / 0.892"],
        ["Feasibility",               "100 %",         "100 %",         "100 %"],
    ],
    filename="tabel3_control",
    figsize=(13, None),
)

# ─────────────────────────────────────────────────────────────
# TABEL 4 — Defalcare pe dificultate
# ─────────────────────────────────────────────────────────────
make_table(
    title="Tabel IV — Defalcare pe dificultate scenariu (mediană IAE_norm)",
    headers=["Dificultate", "n", "Riccati înainte", "Riccati după",
             "IAE Baseline", "IAE Transformer", "IAE MinT", "MinT ajută"],
    rows=[
        ["Ușor",   "28", "5153.64", "450.22", "0.233", "0.239", "0.262", "28.6 %"],
        ["Mediu",  "36", "793.01",  "128.02", "0.239", "0.244", "0.328", "25.0 %"],
        ["Dificil","16", "759.99",  "149.06", "0.328", "0.262", "0.766", "0.0 %"],
    ],
    filename="tabel4_difficulty",
    figsize=(13, None),
)

print(f"\n[done] all tables in {OUT}")
for f in sorted(OUT.glob("tabel*")):
    print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")
