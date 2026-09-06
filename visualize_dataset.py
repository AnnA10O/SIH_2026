"""
Dataset Visualisation Dashboard — PS 26077
Generates 6 plots so you can SEE what the labeled IMD dataset looks like.
Run: python visualize_dataset.py
Saves: outputs/dataset_dashboard.png
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm

sys.path.insert(0, str(Path(__file__).resolve().parent))

IMD_DIR = Path("data/raw/imd_rain")
OUT_DIR  = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

LABEL_COLORS = {
    "CONFIRMED_CLOUDBURST":  "#e63946",   # red
    "CANDIDATE_CLOUDBURST":  "#f4a261",   # orange
    "WIDESPREAD_HEAVY_RAIN": "#457b9d",   # steel blue
    "HEAVY_RAIN":            "#a8dadc",   # light cyan
    "MODERATE_RAIN":         "#b7e4c7",   # light green
    "NORMAL":                "#e9ecef",   # light grey
}

LABEL_ORDER = [
    "CONFIRMED_CLOUDBURST", "CANDIDATE_CLOUDBURST",
    "WIDESPREAD_HEAVY_RAIN", "HEAVY_RAIN", "MODERATE_RAIN", "NORMAL"
]

# ── Load all regions ───────────────────────────────────────────────────────────
print("Loading parquet files...")
dfs = []
for fp in sorted(IMD_DIR.glob("*_2000_2023_JJAS.parquet")):
    df = pd.read_parquet(fp)
    region_name = fp.stem.replace("_2000_2023_JJAS", "").capitalize()
    df["region_short"] = region_name
    dfs.append(df)
    print(f"  {fp.name}: {len(df):,} rows")

full = pd.concat(dfs, ignore_index=True)
full["date"] = pd.to_datetime(full["date"])
full["year"] = full["date"].dt.year
print(f"\nTotal: {len(full):,} rows\n")

# ── Figure setup ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor("#0d1117")
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38)

ax_pie   = fig.add_subplot(gs[0, 0])          # Plot 1: overall label pie
ax_reg   = fig.add_subplot(gs[0, 1:])         # Plot 2: per-region stacked bar
ax_trend = fig.add_subplot(gs[1, :2])         # Plot 3: yearly cloudburst trend
ax_lscore= fig.add_subplot(gs[1, 2])          # Plot 4: L-score histogram
ax_rain  = fig.add_subplot(gs[2, :2])         # Plot 5: rainfall distribution by label
ax_hmap  = fig.add_subplot(gs[2, 2])          # Plot 6: spatial heatmap

DARK_BG  = "#0d1117"
PANEL_BG = "#161b22"
TEXT_COL = "#c9d1d9"
GRID_COL = "#30363d"

for ax in [ax_pie, ax_reg, ax_trend, ax_lscore, ax_rain, ax_hmap]:
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COL, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)

fig.suptitle(
    "PS 26077 — IMD 24-Year Labeled Dataset Dashboard (2000–2023 JJAS)",
    color=TEXT_COL, fontsize=14, fontweight="bold", y=0.98
)

# ── Plot 1: Overall label distribution pie ─────────────────────────────────────
counts = full["rain_label"].value_counts()
pie_labels = [l for l in LABEL_ORDER if l in counts.index]
pie_vals   = [counts[l] for l in pie_labels]
pie_colors = [LABEL_COLORS[l] for l in pie_labels]
short_names = {
    "CONFIRMED_CLOUDBURST":  "Confirmed CB",
    "CANDIDATE_CLOUDBURST":  "Candidate CB",
    "WIDESPREAD_HEAVY_RAIN": "Widespread Heavy",
    "HEAVY_RAIN":            "Heavy Rain",
    "MODERATE_RAIN":         "Moderate Rain",
    "NORMAL":                "Normal",
}
wedges, texts, autotexts = ax_pie.pie(
    pie_vals,
    labels=[short_names[l] for l in pie_labels],
    colors=pie_colors,
    autopct=lambda p: f"{p:.1f}%" if p > 1.5 else "",
    startangle=140,
    textprops={"color": TEXT_COL, "fontsize": 7.5},
    wedgeprops={"linewidth": 0.5, "edgecolor": DARK_BG}
)
for at in autotexts:
    at.set_color(DARK_BG)
    at.set_fontsize(7)
ax_pie.set_title("Overall Label Distribution\n(913,536 records)", color=TEXT_COL, fontsize=9, pad=6)

# ── Plot 2: Per-region stacked bar ─────────────────────────────────────────────
regions = full["region_short"].unique()
region_label_counts = full.groupby(["region_short", "rain_label"]).size().unstack(fill_value=0)
# Normalise to percentages
region_pct = region_label_counts.div(region_label_counts.sum(axis=1), axis=0) * 100

bottoms = np.zeros(len(region_pct))
for lbl in LABEL_ORDER:
    if lbl in region_pct.columns:
        vals = region_pct[lbl].values
        ax_reg.bar(region_pct.index, vals, bottom=bottoms,
                   color=LABEL_COLORS[lbl], label=short_names[lbl],
                   edgecolor=DARK_BG, linewidth=0.3)
        bottoms += vals

ax_reg.set_title("Label Mix by Region (%)", color=TEXT_COL, fontsize=9)
ax_reg.set_ylabel("% of region records", color=TEXT_COL, fontsize=8)
ax_reg.tick_params(axis="x", rotation=20)
ax_reg.set_ylim(0, 105)
ax_reg.yaxis.grid(True, color=GRID_COL, linewidth=0.4, linestyle="--")
ax_reg.legend(loc="upper right", fontsize=6.5, framealpha=0.2,
              labelcolor=TEXT_COL, facecolor=PANEL_BG, edgecolor=GRID_COL)

# ── Plot 3: Yearly cloudburst trend (confirmed + candidate) ────────────────────
cb_yearly = full[full["rain_label"].isin(["CONFIRMED_CLOUDBURST", "CANDIDATE_CLOUDBURST"])]
conf_yr   = cb_yearly[cb_yearly["rain_label"] == "CONFIRMED_CLOUDBURST"].groupby("year").size()
cand_yr   = cb_yearly[cb_yearly["rain_label"] == "CANDIDATE_CLOUDBURST"].groupby("year").size()

years = list(range(2000, 2024))
conf_vals = [conf_yr.get(y, 0) for y in years]
cand_vals = [cand_yr.get(y, 0) for y in years]

ax_trend.bar(years, conf_vals, color=LABEL_COLORS["CONFIRMED_CLOUDBURST"],
             label="Confirmed Cloudburst", alpha=0.9, edgecolor=DARK_BG, linewidth=0.3)
ax_trend.bar(years, cand_vals, bottom=conf_vals, color=LABEL_COLORS["CANDIDATE_CLOUDBURST"],
             label="Candidate Cloudburst", alpha=0.9, edgecolor=DARK_BG, linewidth=0.3)
ax_trend.set_title("Yearly Cloudburst Cell Count (all regions, JJAS)", color=TEXT_COL, fontsize=9)
ax_trend.set_ylabel("Grid cell-day events", color=TEXT_COL, fontsize=8)
ax_trend.set_xlabel("Year", color=TEXT_COL, fontsize=8)
ax_trend.yaxis.grid(True, color=GRID_COL, linewidth=0.4, linestyle="--")
ax_trend.legend(fontsize=7.5, framealpha=0.2, labelcolor=TEXT_COL,
                facecolor=PANEL_BG, edgecolor=GRID_COL)
# Mark Kedarnath 2013
ax_trend.axvline(2013, color="#e63946", linewidth=1.2, linestyle="--", alpha=0.7)
ax_trend.text(2013.15, ax_trend.get_ylim()[1] * 0.88, "Kedarnath\n2013",
              color="#e63946", fontsize=7, va="top")

# ── Plot 4: L-score histogram split by label ───────────────────────────────────
for lbl, col in [("CONFIRMED_CLOUDBURST", "#e63946"),
                 ("CANDIDATE_CLOUDBURST",  "#f4a261"),
                 ("WIDESPREAD_HEAVY_RAIN", "#457b9d")]:
    subset = full[full["rain_label"] == lbl]["L_score"]
    if len(subset) > 0:
        ax_lscore.hist(subset, bins=40, range=(0, 1), color=col, alpha=0.72,
                       label=short_names[lbl], density=True, edgecolor=DARK_BG, linewidth=0.2)

ax_lscore.axvline(0.70, color="#e63946", linewidth=1.2, linestyle="--")
ax_lscore.axvline(0.40, color="#f4a261", linewidth=1.2, linestyle="--")
ax_lscore.text(0.71, ax_lscore.get_ylim()[1] * 0.95 if ax_lscore.get_ylim()[1] > 0 else 5,
               "L=0.70", color="#e63946", fontsize=7, va="top")
ax_lscore.text(0.41, ax_lscore.get_ylim()[1] * 0.75 if ax_lscore.get_ylim()[1] > 0 else 4,
               "L=0.40", color="#f4a261", fontsize=7, va="top")
ax_lscore.set_title("L-score Distribution\nby Label Class", color=TEXT_COL, fontsize=9)
ax_lscore.set_xlabel("L-score (spatial localization)", color=TEXT_COL, fontsize=8)
ax_lscore.set_ylabel("Density", color=TEXT_COL, fontsize=8)
ax_lscore.legend(fontsize=6.5, framealpha=0.2, labelcolor=TEXT_COL,
                 facecolor=PANEL_BG, edgecolor=GRID_COL)
ax_lscore.yaxis.grid(True, color=GRID_COL, linewidth=0.4, linestyle="--")

# ── Plot 5: Rainfall distribution by label (box plot) ─────────────────────────
plot_labels = ["CONFIRMED_CLOUDBURST", "CANDIDATE_CLOUDBURST", "WIDESPREAD_HEAVY_RAIN"]
data_for_box = [
    full[full["rain_label"] == lbl]["rain_mm_day"].clip(upper=300).values
    for lbl in plot_labels
]
bp = ax_rain.boxplot(data_for_box, patch_artist=True, vert=True,
                     medianprops={"color": DARK_BG, "linewidth": 1.5},
                     whiskerprops={"color": TEXT_COL},
                     capprops={"color": TEXT_COL},
                     flierprops={"marker": ".", "markersize": 1.5,
                                 "markerfacecolor": "#666", "alpha": 0.3})
for patch, lbl in zip(bp["boxes"], plot_labels):
    patch.set_facecolor(LABEL_COLORS[lbl])
    patch.set_alpha(0.85)

ax_rain.set_xticks([1, 2, 3])
ax_rain.set_xticklabels(["Confirmed CB", "Candidate CB", "Widespread Heavy"], color=TEXT_COL, fontsize=8)
ax_rain.set_ylabel("Daily Rainfall (mm, clipped at 300)", color=TEXT_COL, fontsize=8)
ax_rain.set_title("Rainfall Intensity Distribution by Class", color=TEXT_COL, fontsize=9)
ax_rain.yaxis.grid(True, color=GRID_COL, linewidth=0.4, linestyle="--")

# ── Plot 6: Spatial heatmap of confirmed cloudbursts ──────────────────────────
cb_cells = full[full["rain_label"] == "CONFIRMED_CLOUDBURST"]
if len(cb_cells) > 0:
    sc = ax_hmap.scatter(
        cb_cells["lon"], cb_cells["lat"],
        c=cb_cells["rain_mm_day"],
        cmap="inferno", s=6, alpha=0.6,
        norm=LogNorm(vmin=max(cb_cells["rain_mm_day"].min(), 1),
                     vmax=cb_cells["rain_mm_day"].max())
    )
    cbar = plt.colorbar(sc, ax=ax_hmap, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=TEXT_COL, labelsize=7)
    cbar.set_label("Rain mm/day (log scale)", color=TEXT_COL, fontsize=7)
    cbar.outline.set_edgecolor(GRID_COL)

ax_hmap.set_title("Confirmed Cloudburst Cells\n(24-year spatial distribution)", color=TEXT_COL, fontsize=9)
ax_hmap.set_xlabel("Longitude", color=TEXT_COL, fontsize=8)
ax_hmap.set_ylabel("Latitude", color=TEXT_COL, fontsize=8)
ax_hmap.yaxis.grid(True, color=GRID_COL, linewidth=0.3, linestyle="--")
ax_hmap.xaxis.grid(True, color=GRID_COL, linewidth=0.3, linestyle="--")

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = OUT_DIR / "dataset_dashboard.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
print(f"\nDashboard saved -> {out_path}")

# ── Also print a quick text summary ───────────────────────────────────────────
print("\n=== LABEL COUNTS ===")
for lbl in LABEL_ORDER:
    n = counts.get(lbl, 0)
    pct = n / len(full) * 100
    print(f"  {lbl:<30} {n:>8,}  ({pct:.3f}%)")

print("\n=== OVERRIDE STATS ===")
overrides = full[full["override_applied"] == True]
print(f"  Total registry overrides applied: {len(overrides):,}")
if len(overrides) > 0:
    print(f"  Events covered:")
    for ev in overrides["override_event"].unique():
        n = (overrides["override_event"] == ev).sum()
        print(f"    {ev}: {n} cells overridden")
