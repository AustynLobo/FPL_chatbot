"""
FPL Prediction Visualiser
==========================
Reads prediction CSVs output by fpl_predictor.py and generates
position-specific outlier scatter plots.

Usage:
    python fpl_visualise.py                          # all positions, latest GW
    python fpl_visualise.py --pos MID                # midfielders only
    python fpl_visualise.py --pos DEF                # defenders only
    python fpl_visualise.py --pos FWD                # forwards only
    python fpl_visualise.py --pos GK                 # goalkeepers only
    python fpl_visualise.py --pos ALL                # all four on one figure
    python fpl_visualise.py --pos MID --gw 31        # specific GW
    python fpl_visualise.py --pos MID --top 40       # top N by predicted points
    python fpl_visualise.py --pos ALL --save         # save PNG to data/predictions/

Positions accepted (case-insensitive):
    GK  / GOALKEEPER
    DEF / DEFENDER
    MID / MIDFIELDER
    FWD / FORWARD / ATT / ATTACKER
    ALL  — renders a 2x2 grid of all four positions
"""

import argparse
import glob
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Position config
# ─────────────────────────────────────────────────────────────────────────────
POS_CONFIG = {
    "GK": {
        "label"      : "Goalkeeper",
        "pos_code"   : "GK",
        "accent"     : "#58A6FF",   # blue
        "default_top": 20,
    },
    "DEF": {
        "label"      : "Defender",
        "pos_code"   : "DEF",
        "accent"     : "#3FB950",   # green
        "default_top": 50,
    },
    "MID": {
        "label"      : "Midfielder",
        "pos_code"   : "MID",
        "accent"     : "#E3B341",   # gold
        "default_top": 60,
    },
    "FWD": {
        "label"      : "Forward",
        "pos_code"   : "FWD",
        "accent"     : "#F85149",   # red
        "default_top": 30,
    },
}

POS_ALIASES = {
    "GK"         : "GK",
    "GOALKEEPER" : "GK",
    "DEF"        : "DEF",
    "DEFENDER"   : "DEF",
    "MID"        : "MID",
    "MIDFIELDER" : "MID",
    "FWD"        : "FWD",
    "FORWARD"    : "FWD",
    "ATT"        : "FWD",
    "ATTACKER"   : "FWD",
    "ALL"        : "ALL",
}

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="FPL position outlier scatter plot")
parser.add_argument(
    "--pos",
    default="ALL",
    help="Position to plot: GK / DEF / MID / FWD / ALL  (default: ALL)"
)
parser.add_argument("--gw",   type=int, default=None, help="Gameweek number (default: latest)")
parser.add_argument("--top",  type=int, default=None, help="Top N players to include (default: position-specific)")
parser.add_argument("--save", action="store_true",    help="Save PNG to data/predictions/ instead of showing")
args = parser.parse_args()

pos_key = POS_ALIASES.get(args.pos.upper())
if pos_key is None:
    raise SystemExit(
        f"  Unknown position '{args.pos}'\n"
        f"  Valid options: GK, DEF, MID, FWD, ALL"
    )

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load predictions CSV
# ─────────────────────────────────────────────────────────────────────────────
PRED_DIR = os.path.join("data", "predictions")

if args.gw:
    csv_path = os.path.join(PRED_DIR, f"fpl_predictions_gw{args.gw}.csv")
else:
    files = sorted(glob.glob(os.path.join(PRED_DIR, "fpl_predictions_gw*.csv")))
    if not files:
        raise SystemExit(
            "  No prediction files found in data/predictions/\n"
            "  Run fpl_predictor.py first to generate predictions."
        )
    csv_path = files[-1]

if not os.path.exists(csv_path):
    raise SystemExit(f"  File not found: {csv_path}")

df_all = pd.read_csv(csv_path)
gw_num = int(os.path.basename(csv_path).replace("fpl_predictions_gw","").replace(".csv",""))
print(f"  Loaded GW{gw_num} predictions — {len(df_all)} players total")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Core plotting function
# ─────────────────────────────────────────────────────────────────────────────
def plot_position(ax, df_all, pos_key, top_n, gw_num, compact=False):
    cfg      = POS_CONFIG[pos_key]
    pos_code = cfg["pos_code"]
    label    = cfg["label"]
    accent   = cfg["accent"]

    # Filter and clean
    pos_df = df_all[df_all["Pos"] == pos_code].copy()
    for col in ["Price(£m)", "PredPts", "RawPts", "AvgPts(L5)", "Sel%"]:
        if col in pos_df.columns:
            pos_df[col] = pd.to_numeric(pos_df[col], errors="coerce")

    pos_df = pos_df.dropna(subset=["Price(£m)", "PredPts"])
    pos_df = pos_df[pos_df["PredPts"] > 0]
    pos_df = pos_df.sort_values("PredPts", ascending=False).head(top_n).reset_index(drop=True)

    if pos_df.empty:
        ax.set_facecolor("#0D1117")
        ax.text(0.5, 0.5, f"No data for {label}",
                ha="center", va="center", color="#8B949E",
                transform=ax.transAxes, fontfamily="monospace")
        return

    n_players = len(pos_df)
    print(f"  {label:>12}s : {n_players} players")

    price = pos_df["Price(£m)"].values
    pts   = pos_df["PredPts"].values
    names = pos_df["Player"].values

    # Bubble size from AvgPts(L5)
    avg_pts     = pd.to_numeric(pos_df["AvgPts(L5)"], errors="coerce").fillna(2)
    size_raw    = np.clip(avg_pts, 0.5, 12)
    max_size    = size_raw.max() if size_raw.max() > 0 else 1
    bubble_size = (size_raw / max_size * (300 if compact else 600)) + (30 if compact else 60)

    # Bubble colour from Sel%
    sel      = pd.to_numeric(pos_df["Sel%"], errors="coerce").fillna(5)
    sel_norm = (sel - sel.min()) / (sel.max() - sel.min() + 1e-9)
    cmap     = matplotlib.colors.LinearSegmentedColormap.from_list(
        f"cmap_{pos_key}",
        ["#F5C518", "#E8834A", "#4A9FE8", "#2C5F8A"]
    )
    colours = cmap(sel_norm)

    # Outlier detection via residuals from trend line
    coeffs     = np.polyfit(price, pts, 1)
    trend_fn   = np.poly1d(coeffs)
    residuals  = pts - trend_fn(price)
    res_std    = residuals.std()
    label_mask = (
        (np.abs(residuals) > 0.9 * res_std) |
        (pts >= np.percentile(pts, 88))
    )

    # Axis setup
    ax.set_facecolor("#0D1117")
    for spine in ax.spines.values():
        spine.set_color("#30363D")
    ax.tick_params(colors="#8B949E", labelsize=7 if compact else 9)

    x_pad = max((price.max() - price.min()) * 0.08, 0.3)
    y_pad = max((pts.max()   - pts.min())   * 0.08, 0.3)
    x_min, x_max = price.min() - x_pad, price.max() + x_pad
    y_min, y_max = pts.min()   - y_pad, pts.max()   + y_pad
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    price_mid = np.median(price)
    pts_mid   = np.median(pts)

    # Quadrant shading
    ax.fill_betweenx([pts_mid, y_max], x_min, price_mid,
                     color="#238636", alpha=0.04, zorder=0)
    ax.fill_betweenx([y_min, pts_mid], price_mid, x_max,
                     color="#DA3633", alpha=0.04, zorder=0)

    # Quadrant dividers
    ax.axvline(price_mid, color="#30363D", linewidth=0.8, linestyle="--", alpha=0.6, zorder=1)
    ax.axhline(pts_mid,   color="#30363D", linewidth=0.8, linestyle="--", alpha=0.6, zorder=1)

    # Quadrant labels
    qs = 6 if compact else 7
    ax.text(x_min + 0.05, y_max - 0.05, "HIGH VALUE\nDifferentials",
            fontsize=qs, color="#3FB950", alpha=0.6, fontfamily="monospace", va="top")
    ax.text(x_max - 0.05, y_min + 0.05, "OVERPRICED\nAvoid / Sell",
            fontsize=qs, color="#F85149", alpha=0.6, fontfamily="monospace",
            va="bottom", ha="right")
    ax.text(x_max - 0.05, y_max - 0.05, "PREMIUM\nTemplates",
            fontsize=qs, color="#E3B341", alpha=0.6, fontfamily="monospace",
            va="top", ha="right")
    ax.text(x_min + 0.05, y_min + 0.05, "CHEAP & WEAK\nBench fodder",
            fontsize=qs, color="#8B949E", alpha=0.6, fontfamily="monospace", va="bottom")

    # Trend line
    x_trend = np.linspace(x_min, x_max, 200)
    ax.plot(x_trend, trend_fn(x_trend),
            color=accent, linewidth=1, linestyle=":", alpha=0.4, zorder=2)

    # Scatter
    ax.scatter(price, pts,
               s=bubble_size, c=colours,
               alpha=0.85, linewidths=0.6,
               edgecolors="#21262D", zorder=3)

    # Player labels on outliers only
    fs = 6.5 if compact else 7.8
    for i, (px, py, name) in enumerate(zip(price, pts, names)):
        if not label_mask[i]:
            continue
        dx = 0.10 if px < price_mid else -0.10
        dy = 0.12 if py > pts_mid   else -0.18
        txt = ax.annotate(
            name,
            xy=(px, py),
            xytext=(px + dx, py + dy),
            fontsize=fs,
            color="#E6EDF3",
            fontfamily="monospace",
            arrowprops=dict(
                arrowstyle="-", color="#444C56", lw=0.6,
                connectionstyle="arc3,rad=0.1"
            ),
            bbox=dict(
                boxstyle="round,pad=0.2", facecolor="#161B22",
                edgecolor="#30363D", alpha=0.85, linewidth=0.5
            ),
            zorder=5
        )
        txt.set_path_effects([pe.withStroke(linewidth=1.2, foreground="#0D1117")])

    # Titles and axis labels
    title_fs = 10 if compact else 13
    label_fs = 8  if compact else 10
    sub_fs   = 6  if compact else 7.5

    ax.set_title(
        f"GW{gw_num}  {label} Outlier Map  ·  top {n_players}",
        color="#E6EDF3", fontsize=title_fs,
        fontfamily="monospace", fontweight="bold", pad=10
    )
    ax.set_xlabel("Price  (£m)", color="#8B949E", fontsize=label_fs,
                  fontfamily="monospace", labelpad=6)
    ax.set_ylabel("Predicted Points", color="#8B949E", fontsize=label_fs,
                  fontfamily="monospace", labelpad=6)
    ax.text(
        0.5, 1.01,
        "Bubble size = recent avg pts  ·  Colour = ownership %",
        transform=ax.transAxes, ha="center",
        fontsize=sub_fs, color="#8B949E", fontfamily="monospace"
    )

    # Colour bar
    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=sel.min(), vmax=sel.max())
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.01,
                        fraction=0.015 if compact else 0.018,
                        aspect=25 if compact else 30)
    cbar.set_label("Sel%", color="#8B949E",
                   fontsize=6 if compact else 8, fontfamily="monospace")
    cbar.ax.yaxis.set_tick_params(color="#8B949E",
                                  labelsize=6 if compact else 7.5)
    cbar.outline.set_edgecolor("#30363D")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#8B949E")

    # Bubble size legend
    sz_pairs = [(4, "Low"), (7, "Mid"), (10, "High")] if compact \
               else [(5, "Low avg pts"), (9, "Mid avg pts"), (13, "High avg pts")]
    legend_elements = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor="#8B949E", markeredgecolor="#444C56",
               markersize=sz, label=lbl)
        for sz, lbl in sz_pairs
    ]
    legend = ax.legend(
        handles=legend_elements,
        title="Size = AvgPts(L5)" if compact else "Bubble size = AvgPts(L5)",
        loc="lower right",
        framealpha=0.15, facecolor="#161B22", edgecolor="#30363D",
        labelcolor="#8B949E",
        title_fontsize=6 if compact else 7.5,
        fontsize=6 if compact else 7.5,
    )
    legend.get_title().set_color("#8B949E")
    legend.get_title().set_fontfamily("monospace")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Render
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"      : "monospace",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
})

if pos_key == "ALL":
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.patch.set_facecolor("#0D1117")
    fig.suptitle(
        f"GW{gw_num}  FPL Outlier Maps  —  All Positions",
        color="#E6EDF3", fontsize=16, fontfamily="monospace",
        fontweight="bold", y=1.005
    )
    for ax, pk in zip(axes.flat, ["GK", "DEF", "MID", "FWD"]):
        top_n = args.top or POS_CONFIG[pk]["default_top"]
        plot_position(ax, df_all, pk, top_n, gw_num, compact=True)

    plt.tight_layout(h_pad=3.5, w_pad=2.5)
    out_suffix = "all_positions"

else:
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor("#0D1117")
    top_n = args.top or POS_CONFIG[pos_key]["default_top"]
    plot_position(ax, df_all, pos_key, top_n, gw_num, compact=False)
    plt.tight_layout()
    out_suffix = pos_key.lower()

# ─────────────────────────────────────────────────────────────────────────────
# 4. Save or show
# ─────────────────────────────────────────────────────────────────────────────
if args.save:
    out_path = os.path.join(PRED_DIR, f"fpl_outliers_{out_suffix}_gw{gw_num}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\n  Chart saved -> {out_path}")
else:
    print("\nShowing chart — close the window to exit.")
    plt.show()