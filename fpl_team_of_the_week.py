"""
FPL Team of the Week
=====================
Two modes:

  --mode actual   (default)
      Reads exact GW points from the saved history parquet.
      Builds the best XI from players who ACTUALLY played in that GW.
      Answers: "Who was the team of the week for GW30?"

  --mode predict
      Reads PredPts from the predictions CSV.
      Builds the best XI for the UPCOMING gameweek.
      Answers: "Who should I pick for GW31?"

Rules enforced:
  - Exactly 1 GK
  - 3–5 DEF, 3–5 MID, 1–3 FWD (formation dependent)
  - Total cost <= budget (default £100m)
  - Max 3 players from the same club (standard FPL rule)

Valid formations: 3-4-3, 3-5-2, 4-3-3, 4-4-2, 4-5-1, 5-3-2, 5-4-1, 5-2-3

Usage:
    python fpl_team_of_week.py                          # actual TOTW, latest GW
    python fpl_team_of_week.py --mode actual            # same as above
    python fpl_team_of_week.py --mode predict           # predicted XI for next GW
    python fpl_team_of_week.py --mode actual  --gw 30   # specific GW
    python fpl_team_of_week.py --mode predict --gw 31   # specific upcoming GW
    python fpl_team_of_week.py --formation 4-3-3        # force a formation
    python fpl_team_of_week.py --budget 95              # custom budget (default 100)
    python fpl_team_of_week.py --save                   # save PNG to data/predictions/

Requirements:
    pip install pulp matplotlib pandas numpy pyarrow
"""

import argparse
import glob
import json
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
FORMATIONS = {
    "3-4-3": (3, 4, 3),
    "3-5-2": (3, 5, 2),
    "4-3-3": (4, 3, 3),
    "4-4-2": (4, 4, 2),
    "4-5-1": (4, 5, 1),
    "5-3-2": (5, 3, 2),
    "5-4-1": (5, 4, 1),
    "5-2-3": (5, 2, 3),
}

MAX_PER_CLUB   = 3

PRED_DIR       = os.path.join("data", "predictions")
CACHE_DIR      = os.path.join("data", "cache")
HISTORY_PATH   = os.path.join(CACHE_DIR, "player_history.parquet")
BOOTSTRAP_PATH = os.path.join(CACHE_DIR, "bootstrap.json")

POS_MAP     = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
POS_COLOURS = {
    "GK" : "#F5C518",
    "DEF": "#58A6FF",
    "MID": "#3FB950",
    "FWD": "#F85149",
}
ROW_Y = {"GK": 0.10, "DEF": 0.30, "MID": 0.55, "FWD": 0.80}

PITCH_GREEN    = "#2D5A1B"
PITCH_LINE     = "#FFFFFF"
CARD_BG        = "#1A3A10"
TEXT_PRIMARY   = "#FFFFFF"
TEXT_SECONDARY = "#B8D4A8"
ACCENT_GOLD    = "#F5C518"

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="FPL Team of the Week")
parser.add_argument(
    "--mode",
    choices=["actual", "predict"],
    default="actual",
    help="'actual' = TOTW from real GW points | 'predict' = best XI for next GW (default: actual)"
)
parser.add_argument("--gw",        type=int,   default=None,  help="Gameweek number (default: auto-detect)")
parser.add_argument("--formation", default=None,              help="Force formation e.g. 4-3-3")
parser.add_argument("--budget",    type=float, default=100.0, help="Budget in £m (default: 100)")
parser.add_argument("--save",      action="store_true",       help="Save PNG instead of showing")
args = parser.parse_args()

if args.formation and args.formation not in FORMATIONS:
    raise SystemExit(
        f"  Unknown formation '{args.formation}'\n"
        f"  Valid: {', '.join(FORMATIONS.keys())}"
    )

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load data based on mode
# ─────────────────────────────────────────────────────────────────────────────

def load_bootstrap_teams():
    """Returns a dict of team_id -> team_short_name from bootstrap cache."""
    if not os.path.exists(BOOTSTRAP_PATH):
        raise SystemExit(
            "  data/cache/bootstrap.json not found.\n"
            "  Run fpl_predictor.py first."
        )
    with open(BOOTSTRAP_PATH) as f:
        bootstrap = json.load(f)

    teams_df = pd.DataFrame(bootstrap["teams"])
    teams_df["id"] = pd.to_numeric(teams_df["id"], errors="coerce")
    # Use short_name (e.g. "ARS", "CHE") — falls back to name if not available
    name_col = "short_name" if "short_name" in teams_df.columns else "name"
    return dict(zip(teams_df["id"], teams_df[name_col]))


def load_actual(gw_override):
    """
    Reads exact total_points for a specific finished GW from the history
    parquet. Merges player name, position, price and club from bootstrap.

    Returns (df, gw_num) where df has columns:
        Player, Club, Pos, Price(£m), score, team_id
    """
    if not os.path.exists(HISTORY_PATH):
        raise SystemExit(
            "  data/cache/player_history.parquet not found.\n"
            "  Run fpl_predictor.py first to build the local cache."
        )

    history = pd.read_parquet(HISTORY_PATH)

    # Normalise event column
    if "round" in history.columns and "event" not in history.columns:
        history = history.rename(columns={"round": "event"})
    history["event"]       = pd.to_numeric(history.get("event", np.nan), errors="coerce")
    history["player_id"]   = pd.to_numeric(history["player_id"],         errors="coerce")
    history["total_points"]= pd.to_numeric(history.get("total_points",0),errors="coerce").fillna(0)
    history["minutes"]     = pd.to_numeric(history.get("minutes", 0),    errors="coerce").fillna(0)

    finished_gws = sorted(history["event"].dropna().unique().astype(int))
    if not finished_gws:
        raise SystemExit("  No finished gameweeks found in history cache.")

    gw_num = gw_override if gw_override else int(finished_gws[-1])
    if gw_num not in finished_gws:
        raise SystemExit(
            f"  GW{gw_num} not found in history.\n"
            f"  Available: GW{finished_gws[0]}–GW{finished_gws[-1]}"
        )

    gw_data = history[history["event"] == gw_num].copy()
    gw_data = gw_data[gw_data["minutes"] > 0]   # played at least 1 minute

    if gw_data.empty:
        raise SystemExit(f"  No players with minutes > 0 found in GW{gw_num}.")

    # Bootstrap metadata
    with open(BOOTSTRAP_PATH) as f:
        bootstrap = json.load(f)

    players = pd.DataFrame(bootstrap["elements"])
    players["id"]           = pd.to_numeric(players["id"],           errors="coerce")
    players["now_cost"]     = pd.to_numeric(players["now_cost"],     errors="coerce")
    players["element_type"] = pd.to_numeric(players["element_type"], errors="coerce")
    players["team"]         = pd.to_numeric(players["team"],         errors="coerce")

    team_names = load_bootstrap_teams()

    merged = gw_data.merge(
        players[["id", "web_name", "element_type", "now_cost", "team"]],
        left_on="player_id", right_on="id", how="left"
    )
    merged = merged.dropna(subset=["web_name", "element_type", "now_cost", "team"])

    merged["Pos"]      = merged["element_type"].map(POS_MAP)
    merged["Price(£m)"]= merged["now_cost"] / 10
    merged["Player"]   = merged["web_name"]
    merged["team_id"]  = merged["team"].astype(int)
    merged["Club"]     = merged["team_id"].map(team_names).fillna("UNK")
    merged["score"]    = merged["total_points"]

    result = merged[["Player", "Club", "Pos", "Price(£m)", "score", "team_id"]].copy()
    result = result.dropna(subset=["Pos"])
    result = result[result["score"] >= 0].reset_index(drop=True)

    return result, gw_num


def load_predict(gw_override):
    """
    Reads PredPts from the predictions CSV. Merges club name from bootstrap.

    Returns (df, gw_num) where df has columns:
        Player, Club, Pos, Price(£m), score, team_id
    """
    if gw_override:
        csv_path = os.path.join(PRED_DIR, f"fpl_predictions_gw{gw_override}.csv")
    else:
        files = sorted(glob.glob(os.path.join(PRED_DIR, "fpl_predictions_gw*.csv")))
        if not files:
            raise SystemExit(
                "  No prediction files found in data/predictions/\n"
                "  Run fpl_predictor.py first."
            )
        csv_path = files[-1]

    if not os.path.exists(csv_path):
        raise SystemExit(f"  File not found: {csv_path}")

    df = pd.read_csv(csv_path)
    gw_num = int(
        os.path.basename(csv_path)
        .replace("fpl_predictions_gw", "")
        .replace(".csv", "")
    )

    for col in ["Price(£m)", "PredPts"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Price(£m)", "PredPts", "Pos"])
    df = df[df["PredPts"] > 0].reset_index(drop=True)

    # Merge club name from bootstrap
    with open(BOOTSTRAP_PATH) as f:
        bootstrap = json.load(f)

    players = pd.DataFrame(bootstrap["elements"])
    players["id"]   = pd.to_numeric(players["id"],   errors="coerce")
    players["team"] = pd.to_numeric(players["team"], errors="coerce")
    team_names = load_bootstrap_teams()
    players["Club"] = players["team"].map(team_names).fillna("UNK")

    # Match on player name — predictions CSV uses web_name as Player
    name_to_club    = dict(zip(players["web_name"], players["Club"]))
    name_to_team_id = dict(zip(players["web_name"], players["team"]))

    df["Club"]    = df["Player"].map(name_to_club).fillna("UNK")
    df["team_id"] = df["Player"].map(name_to_team_id).fillna(-1).astype(int)

    result = df[["Player", "Club", "Pos", "Price(£m)", "PredPts", "team_id"]].copy()
    result = result.rename(columns={"PredPts": "score"})

    return result, gw_num


# ── Load ──────────────────────────────────────────────────────────────────────
print(f"\n  Mode : {args.mode.upper()}")

if args.mode == "actual":
    df, gw_num    = load_actual(args.gw)
    score_label   = "Actual pts"
    title_prefix  = "TEAM OF THE WEEK"
    subtitle_gw   = f"GW{gw_num} — actual points scored"
    file_suffix   = "actual"
else:
    df, gw_num    = load_predict(args.gw)
    score_label   = "Predicted pts"
    title_prefix  = "PREDICTED XI"
    subtitle_gw   = f"GW{gw_num} — model predictions"
    file_suffix   = "predicted"

print(f"  GW              : {gw_num}")
print(f"  Players loaded  : {len(df)}")
print(f"  Budget          : £{args.budget}m")
print(f"  Max per club    : {MAX_PER_CLUB}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. LP optimiser — max 3 per club enforced
# ─────────────────────────────────────────────────────────────────────────────
def optimise_team(df, formation_str, budget):
    try:
        import pulp
    except ImportError:
        raise SystemExit("  PuLP not installed. Run: pip install pulp")

    n_def, n_mid, n_fwd = FORMATIONS[formation_str]
    n = len(df)

    prob = pulp.LpProblem(f"TOTW_{formation_str}", pulp.LpMaximize)
    x    = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]

    # Objective: maximise total score
    prob += pulp.lpSum(df["score"].iloc[i] * x[i] for i in range(n))

    # Budget constraint
    prob += pulp.lpSum(df["Price(£m)"].iloc[i] * x[i] for i in range(n)) <= budget

    # Position constraints
    gk_idx  = [i for i in range(n) if df["Pos"].iloc[i] == "GK"]
    def_idx = [i for i in range(n) if df["Pos"].iloc[i] == "DEF"]
    mid_idx = [i for i in range(n) if df["Pos"].iloc[i] == "MID"]
    fwd_idx = [i for i in range(n) if df["Pos"].iloc[i] == "FWD"]

    if not gk_idx or not def_idx or not mid_idx or not fwd_idx:
        return None

    prob += pulp.lpSum(x[i] for i in gk_idx)  == 1
    prob += pulp.lpSum(x[i] for i in def_idx) == n_def
    prob += pulp.lpSum(x[i] for i in mid_idx) == n_mid
    prob += pulp.lpSum(x[i] for i in fwd_idx) == n_fwd

    # ── Max 3 players per club ────────────────────────────────────────────────
    for club_id in df["team_id"].unique():
        club_idx = [i for i in range(n) if df["team_id"].iloc[i] == club_id]
        if len(club_idx) > MAX_PER_CLUB:
            prob += pulp.lpSum(x[i] for i in club_idx) <= MAX_PER_CLUB

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob.status] != "Optimal":
        return None

    selected_idx = [i for i in range(n) if pulp.value(x[i]) == 1]
    selected_df  = df.iloc[selected_idx].copy()
    total_pts    = selected_df["score"].sum()
    total_cost   = selected_df["Price(£m)"].sum()

    return selected_df, total_pts, total_cost


# ─────────────────────────────────────────────────────────────────────────────
# 3. Try all formations
# ─────────────────────────────────────────────────────────────────────────────
formations_to_try = [args.formation] if args.formation else list(FORMATIONS.keys())

best_result    = None
best_pts       = -1
best_formation = None

print(f"\n  Optimising formations...")
for formation_str in formations_to_try:
    result = optimise_team(df, formation_str, args.budget)
    if result is None:
        print(f"    {formation_str}  →  infeasible")
        continue
    selected_df, total_pts, total_cost = result
    print(f"    {formation_str}  →  {total_pts:.1f} pts  |  £{total_cost:.1f}m")
    if total_pts > best_pts:
        best_pts       = total_pts
        best_result    = result
        best_formation = formation_str

if best_result is None:
    raise SystemExit(
        "  Could not find a valid team within budget and club constraints.\n"
        "  Try increasing --budget."
    )

selected_df, total_pts, total_cost = best_result
n_def, n_mid, n_fwd = FORMATIONS[best_formation]

# ── Console output ────────────────────────────────────────────────────────────
print(f"\n  Best formation  : {best_formation}")
print(f"  Total score     : {total_pts:.1f} pts")
print(f"  Total cost      : £{total_cost:.1f}m  (budget: £{args.budget}m)")

# Check club counts in selected team
club_counts = selected_df["Club"].value_counts()
clubs_at_max = club_counts[club_counts == MAX_PER_CLUB].index.tolist()

print(f"\n  {'Pos':<5} {'Player':<22} {'Club':<6} {'Price':>8}  {'Score':>8}")
print(f"  {'─'*5} {'─'*22} {'─'*6} {'─'*8}  {'─'*8}")

for _, row in selected_df.sort_values(
    ["Pos", "score"], ascending=[True, False]
).iterrows():
    club_flag = " ◀ 3 players" if row["Club"] in clubs_at_max else ""
    print(
        f"  {row['Pos']:<5} {row['Player']:<22} {row['Club']:<6} "
        f"£{row['Price(£m)']:>6.1f}m  {row['score']:>7.1f} pts{club_flag}"
    )

print(f"\n  Club breakdown:")
for club, count in club_counts.sort_values(ascending=False).items():
    bar = "█" * count
    print(f"    {club:<6}  {bar}  ({count})")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Visualise
# ─────────────────────────────────────────────────────────────────────────────
def draw_pitch(ax):
    ax.set_facecolor(PITCH_GREEN)

    # Alternating pitch stripes
    for i in range(10):
        colour = "#2D5A1B" if i % 2 == 0 else "#325F1F"
        ax.axhspan(i / 10, (i + 1) / 10, facecolor=colour, zorder=0)

    lw = 1.5
    ax.plot([0,1,1,0,0], [0,0,1,1,0],
            color=PITCH_LINE, lw=lw, transform=ax.transAxes, zorder=2)
    ax.axhline(0.5, color=PITCH_LINE, lw=lw, alpha=0.5, zorder=2)

    ax.add_patch(mpatches.Circle(
        (0.5, 0.5), 0.08, fill=False,
        edgecolor=PITCH_LINE, lw=lw, alpha=0.5,
        transform=ax.transAxes, zorder=2
    ))
    ax.plot(0.5, 0.5, "o", color=PITCH_LINE,
            markersize=3, transform=ax.transAxes, alpha=0.6, zorder=2)

    for y_start, height in [(0.0, 0.15), (0.85, 0.15)]:
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.22, y_start), 0.56, height,
            boxstyle="square,pad=0", fill=False,
            edgecolor=PITCH_LINE, lw=lw, alpha=0.5,
            transform=ax.transAxes, zorder=2
        ))
    for y_start, height in [(0.0, 0.06), (0.94, 0.06)]:
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.36, y_start), 0.28, height,
            boxstyle="square,pad=0", fill=False,
            edgecolor=PITCH_LINE, lw=lw, alpha=0.5,
            transform=ax.transAxes, zorder=2
        ))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def draw_player_card(ax, x, y, player, club, pos, price, pts):
    pos_colour = POS_COLOURS.get(pos, "#FFFFFF")

    # Truncate name
    name = player if len(player) <= 11 else player[:10] + "."

    # Glow ring
    ax.add_patch(mpatches.Circle(
        (x, y), 0.055, color=pos_colour, alpha=0.20,
        transform=ax.transAxes, zorder=4
    ))
    # Card body
    ax.add_patch(mpatches.Circle(
        (x, y), 0.042, color=CARD_BG,
        transform=ax.transAxes, zorder=5
    ))
    # Border
    ax.add_patch(mpatches.Circle(
        (x, y), 0.043, fill=False,
        edgecolor=pos_colour, lw=1.8,
        transform=ax.transAxes, zorder=6
    ))

    # Position badge — top right of card
    ax.add_patch(mpatches.Circle(
        (x + 0.030, y + 0.030), 0.013,
        color=pos_colour,
        transform=ax.transAxes, zorder=7
    ))
    ax.text(
        x + 0.030, y + 0.030, pos,
        ha="center", va="center",
        fontsize=4.5, color="#000000",
        fontweight="bold", fontfamily="monospace",
        transform=ax.transAxes, zorder=8
    )

    # Player name
    txt_name = ax.text(
        x, y + 0.012, name,
        ha="center", va="center",
        fontsize=7.0, color=TEXT_PRIMARY,
        fontweight="bold", fontfamily="monospace",
        transform=ax.transAxes, zorder=7
    )
    txt_name.set_path_effects([pe.withStroke(linewidth=1.5, foreground="#000000")])

    # Club name — shown below player name in accent colour
    txt_club = ax.text(
        x, y - 0.007, club,
        ha="center", va="center",
        fontsize=5.5, color=ACCENT_GOLD,
        fontfamily="monospace",
        transform=ax.transAxes, zorder=7
    )
    txt_club.set_path_effects([pe.withStroke(linewidth=1.0, foreground="#000000")])

    # Price and score at bottom of card
    txt_stats = ax.text(
        x, y - 0.023,
        f"£{price:.1f}  {pts:.1f}pts",
        ha="center", va="center",
        fontsize=5.5, color=TEXT_SECONDARY,
        fontfamily="monospace",
        transform=ax.transAxes, zorder=7
    )
    txt_stats.set_path_effects([pe.withStroke(linewidth=1.0, foreground="#000000")])


def get_row_positions(n, y):
    xs = [0.5] if n == 1 else np.linspace(0.12, 0.88, n).tolist()
    return [(x, y) for x in xs]


# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 11))
fig.patch.set_facecolor("#0D1117")

# Header
ax_header = fig.add_axes([0.0, 0.88, 1.0, 0.12])
ax_header.set_facecolor("#0D1117")
ax_header.axis("off")

# Mode badge
badge_colour = "#F85149" if args.mode == "actual" else "#3FB950"
badge_text   = "ACTUAL"    if args.mode == "actual" else "PREDICTED"
ax_header.add_patch(mpatches.FancyBboxPatch(
    (0.01, 0.50), 0.11, 0.38,
    boxstyle="round,pad=0.02",
    facecolor=badge_colour, edgecolor="none",
    transform=ax_header.transAxes, zorder=2
))
ax_header.text(
    0.065, 0.69, badge_text,
    ha="center", va="center",
    fontsize=8, color="#FFFFFF",
    fontweight="bold", fontfamily="monospace",
    transform=ax_header.transAxes, zorder=3
)

# Title
ax_header.text(
    0.5, 0.78, f"GW{gw_num}  {title_prefix}",
    ha="center", va="center",
    fontsize=20, color=TEXT_PRIMARY,
    fontweight="bold", fontfamily="monospace",
    transform=ax_header.transAxes
)
ax_header.text(
    0.5, 0.28,
    f"Formation: {best_formation}   ·   "
    f"Cost: £{total_cost:.1f}m / £{args.budget:.0f}m   ·   "
    f"{total_pts:.1f} {score_label}   ·   max {MAX_PER_CLUB} per club",
    ha="center", va="center",
    fontsize=10, color=ACCENT_GOLD,
    fontfamily="monospace",
    transform=ax_header.transAxes
)

# Pitch
ax_pitch = fig.add_axes([0.05, 0.02, 0.90, 0.86])
draw_pitch(ax_pitch)

# Draw player cards
by_pos = {
    pos: selected_df[selected_df["Pos"] == pos].sort_values("score", ascending=False)
    for pos in ["GK", "DEF", "MID", "FWD"]
}

for pos, players_in_row in by_pos.items():
    if players_in_row.empty:
        continue
    coords = get_row_positions(len(players_in_row), ROW_Y[pos])
    for (x, y_coord), (_, row) in zip(coords, players_in_row.iterrows()):
        draw_player_card(
            ax_pitch,
            x, y_coord,
            player = str(row["Player"]),
            club   = str(row["Club"]),
            pos    = pos,
            price  = float(row["Price(£m)"]),
            pts    = float(row["score"]),
        )

# Position legend
ax_pitch.legend(
    handles=[
        mpatches.Patch(facecolor=POS_COLOURS["GK"],  label="Goalkeeper"),
        mpatches.Patch(facecolor=POS_COLOURS["DEF"], label="Defender"),
        mpatches.Patch(facecolor=POS_COLOURS["MID"], label="Midfielder"),
        mpatches.Patch(facecolor=POS_COLOURS["FWD"], label="Forward"),
    ],
    loc="lower right",
    bbox_to_anchor=(0.99, 0.01),
    framealpha=0.25, facecolor="#0D1117", edgecolor="#30363D",
    labelcolor=TEXT_PRIMARY, fontsize=8, ncol=4
)

# Club breakdown box — bottom left of pitch
club_lines = [f"{club:<5} {'█' * cnt} ({cnt})"
              for club, cnt in club_counts.sort_values(ascending=False).items()]
club_text  = "Club breakdown\n" + "\n".join(club_lines)
ax_pitch.text(
    0.01, 0.01, club_text,
    ha="left", va="bottom",
    fontsize=6, color=TEXT_SECONDARY,
    fontfamily="monospace",
    transform=ax_pitch.transAxes,
    bbox=dict(
        boxstyle="round,pad=0.4",
        facecolor="#0D1117", edgecolor="#30363D",
        alpha=0.75, linewidth=0.8
    ),
    zorder=10
)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Save or show
# ─────────────────────────────────────────────────────────────────────────────
if args.save:
    out_path = os.path.join(PRED_DIR, f"fpl_totw_{file_suffix}_gw{gw_num}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\n  Saved -> {out_path}")
else:
    print("\nShowing team — close the window to exit.")
    plt.show()