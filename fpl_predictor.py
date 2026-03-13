"""
FPL Gameweek Score Predictor
=============================
Data sources (all live from FPL API):
  bootstrap      : /api/bootstrap-static/
  fixtures       : /api/fixtures/
  player history : /api/element-summary/{id}/  ← exact per-GW points + minutes

Key improvements over previous version:
  1. Exact FPL points from element-summary API (no reconstruction from fixture stats)
  2. Exact minutes per GW (fixes availability score accuracy)
  3. Separate XGBoost model per position (GK / DEF / MID / FWD)
  4. Position-specific features (e.g. clean sheets for GK/DEF, goals for FWD)
  5. Best players ranked per position in output
  6. Injury/availability intelligence using chance_of_playing_next_round + status
  7. Transfer momentum feature (crowd wisdom signal)
  8. Data-driven opponent defensive strength (replaces human-rated FDR)
  9. Double gameweek multiplier per position
  10. Price change signal output
  11. Overprediction fixes: target clipping, min_child_weight, max_depth, prediction cap

Usage:
    python fpl_predictor.py [--predict-gw 31] [--debug "Salah"] [--force-refresh]
    python fpl_predictor.py --export --s3-bucket my-fpl-predictions

Output:
    data/predictions/fpl_predictions_gw<N>.csv        — all players ranked by PredPts
    data/predictions/fpl_best_by_position_gw<N>.csv   — top 10 per position
    data/predictions/fpl_validation_summary_gw<N>.csv — per-player MAE on val GWs
    data/predictions/fpl_price_signals_gw<N>.csv      — price change predictions
    data/predictions/fpl_training_summary.txt          — split info and per-position MAE
"""

import argparse
import io
import json
import os
import time
import warnings
import requests
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")
os.makedirs("data", exist_ok=True)
os.makedirs(os.path.join("data", "predictions"), exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--predict-gw",    type=int, default=None)
parser.add_argument("--debug",         default="", help="Player web_name for detailed trace")
parser.add_argument("--force-refresh", action="store_true",
                    help="Ignore cache and re-fetch all data from FPL API")
parser.add_argument("--export",    action="store_true",
                    help="Upload predictions + cache to S3 after running")
parser.add_argument("--s3-bucket", default="",
                    help="S3 bucket name for --export  e.g. my-fpl-bucket")
args = parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# Overprediction fix constants
# ─────────────────────────────────────────────────────────────────────────────
# Hard cap on raw predicted points per position (before availability multiplier)
MAX_POINTS  = {1: 15, 2: 15, 3: 18, 4: 18}

# Clip training target to reduce influence of outlier hauls
CLIP_TARGET = {1: 16, 2: 16, 3: 20, 4: 20}

# ─────────────────────────────────────────────────────────────────────────────
# S3 upload helpers  (only used when --export is passed)
# ─────────────────────────────────────────────────────────────────────────────
def s3_client():
    try:
        import boto3
        return boto3.client("s3")
    except ImportError:
        raise SystemExit("❌ boto3 not installed. Run: pip install boto3")

def upload_file_to_s3(local_path, s3_key, bucket):
    s3 = s3_client()
    s3.upload_file(local_path, bucket, s3_key)
    print(f"  ☁️  s3://{bucket}/{s3_key}")

def upload_df_to_s3(df, s3_key, bucket, fmt="csv"):
    s3 = s3_client()
    buf = io.BytesIO()
    if fmt == "parquet":
        df.to_parquet(buf, index=False)
    else:
        df.to_csv(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=s3_key, Body=buf.getvalue())
    print(f"  ☁️  s3://{bucket}/{s3_key}")

def upload_json_to_s3(obj, s3_key, bucket):
    s3 = s3_client()
    s3.put_object(
        Bucket=bucket, Key=s3_key,
        Body=json.dumps(obj).encode()
    )
    print(f"  ☁️  s3://{bucket}/{s3_key}")

# ─────────────────────────────────────────────────────────────────────────────
# Cache paths
# ─────────────────────────────────────────────────────────────────────────────
CACHE_DIR       = os.path.join("data", "cache")
CACHE_META      = os.path.join(CACHE_DIR, "meta.json")
CACHE_BOOTSTRAP = os.path.join(CACHE_DIR, "bootstrap.json")
CACHE_FIXTURES  = os.path.join(CACHE_DIR, "fixtures.json")
CACHE_HISTORY   = os.path.join(CACHE_DIR, "player_history.parquet")
os.makedirs(CACHE_DIR, exist_ok=True)


def _live_last_gw():
    bs = requests.get(
        "https://fantasy.premierleague.com/api/bootstrap-static/", timeout=15
    ).json()
    finished = [e["id"] for e in bs.get("events", []) if e.get("finished")]
    return max(finished) if finished else 0, bs


def cache_valid():
    if not all(os.path.exists(p) for p in
               [CACHE_META, CACHE_BOOTSTRAP, CACHE_FIXTURES, CACHE_HISTORY]):
        return False, None
    try:
        live_gw, live_bs = _live_last_gw()
        with open(CACHE_META) as f:
            cached_gw = json.load(f).get("last_finished_gw", -1)
        if live_gw == cached_gw:
            print(f"  Cache hit  — last finished GW = {live_gw}, no re-fetch needed")
            return True, live_bs
        print(f"  Cache miss — cached GW {cached_gw} -> live GW {live_gw}, re-fetching")
        return False, live_bs
    except Exception as ex:
        print(f"  Cache check failed ({ex}), re-fetching")
        return False, None


def save_cache(bs, fx, hist_df, gw):
    with open(CACHE_BOOTSTRAP, "w") as f:
        json.dump(bs, f)
    with open(CACHE_FIXTURES, "w") as f:
        json.dump(fx, f)
    hist_df.to_parquet(CACHE_HISTORY, index=False)
    with open(CACHE_META, "w") as f:
        json.dump({"last_finished_gw": int(gw)}, f)
    print(f"  Cache saved for GW {gw}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load bootstrap + fixtures  (from cache or FPL API)
# ─────────────────────────────────────────────────────────────────────────────
print("Checking cache...")
is_cached, prefetched_bs = (False, None) if args.force_refresh else cache_valid()

if is_cached:
    print("Loading bootstrap and fixtures from cache...")
    with open(CACHE_BOOTSTRAP) as f:
        bootstrap = json.load(f)
    with open(CACHE_FIXTURES) as f:
        fixtures_raw = json.load(f)
else:
    print("Fetching bootstrap and fixtures from FPL API...")
    try:
        bootstrap    = prefetched_bs or requests.get(
            "https://fantasy.premierleague.com/api/bootstrap-static/", timeout=15).json()
        fixtures_raw = requests.get(
            "https://fantasy.premierleague.com/api/fixtures/", timeout=15).json()
    except requests.exceptions.RequestException as e:
        raise SystemExit(f"  ❌ API error: {e}")

players  = pd.DataFrame(bootstrap["elements"])
teams_df = pd.DataFrame(bootstrap["teams"])
fixtures = pd.DataFrame(fixtures_raw)

players["id"] = pd.to_numeric(players["id"], errors="coerce")
fixtures["event"] = pd.to_numeric(fixtures["event"], errors="coerce")

print(f"  Players : {len(players)}  |  Fixtures : {len(fixtures)}  |  Teams : {len(teams_df)}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Identify GWs and split
# ─────────────────────────────────────────────────────────────────────────────
finished_mask       = fixtures["finished"].astype(str).str.lower() == "true"
finished_gws_sorted = sorted(fixtures[finished_mask]["event"].dropna().unique().astype(int))
last_finished_gw    = finished_gws_sorted[-1] if finished_gws_sorted else 0
predict_gw          = args.predict_gw if args.predict_gw else last_finished_gw + 1

n_gws       = len(finished_gws_sorted)
n_train_gws = max(1, int(round(n_gws * 0.80)))
train_gws   = finished_gws_sorted[:n_train_gws]
val_gws     = finished_gws_sorted[n_train_gws:]

print(f"\n  Finished GWs    : GW{finished_gws_sorted[0]}–GW{last_finished_gw}  ({n_gws} total)")
print(f"  Train  (80%)    : GW{train_gws[0]}–GW{train_gws[-1]}  ({len(train_gws)} GWs)")
print(f"  Val    (20%)    : " + (f"GW{val_gws[0]}–GW{val_gws[-1]}  ({len(val_gws)} GWs)" if val_gws else "none yet"))
print(f"  Predicting      : GW{predict_gw}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Load exact per-GW history  (from cache or element-summary API)
# ─────────────────────────────────────────────────────────────────────────────
all_player_ids = players["id"].dropna().astype(int).tolist()

if is_cached:
    history = pd.read_parquet(CACHE_HISTORY)
    print(f"  Loaded {len(history)} player-GW records from cache")
else:
    print(f"\nFetching per-GW history for {len(players)} players (~2 min)...")
    all_histories = []

    for i, pid in enumerate(all_player_ids):
        try:
            url  = f"https://fantasy.premierleague.com/api/element-summary/{pid}/"
            data = requests.get(url, timeout=10).json()
            hist = pd.DataFrame(data.get("history", []))
            if not hist.empty:
                hist["player_id"] = pid
                all_histories.append(hist)
        except Exception:
            pass
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(all_player_ids)} players fetched...")
        time.sleep(0.05)

    history = pd.concat(all_histories, ignore_index=True) if all_histories else pd.DataFrame()
    print(f"  Fetched {len(history)} records across {history['player_id'].nunique()} players")
    save_cache(bootstrap, fixtures_raw, history, last_finished_gw)

# Standardise column names and types
history["player_id"] = pd.to_numeric(history["player_id"], errors="coerce")
history["round"]     = pd.to_numeric(history.get("round", history.get("event", np.nan)), errors="coerce")
history             = history.rename(columns={"round": "event"})
history["event"]    = pd.to_numeric(history["event"], errors="coerce")

EXACT_STATS = [
    "total_points", "minutes", "goals_scored", "assists", "clean_sheets",
    "goals_conceded", "own_goals", "penalties_saved", "penalties_missed",
    "yellow_cards", "red_cards", "saves", "bonus", "bps",
    "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded"
]
for col in EXACT_STATS:
    if col in history.columns:
        history[col] = pd.to_numeric(history[col], errors="coerce").fillna(0)
    else:
        history[col] = 0

# Merge fixture context (FDR, home/away) from fixtures table
fix_context = []
for _, fix in fixtures[finished_mask].iterrows():
    gw      = fix["event"]
    diff_h  = float(fix.get("team_h_difficulty") or 3)
    diff_a  = float(fix.get("team_a_difficulty") or 3)
    score_h = float(fix.get("team_h_score") or 0)
    score_a = float(fix.get("team_a_score") or 0)
    fix_context.append({"team": fix["team_h"], "event": gw, "fdr": diff_h,
                         "is_home": 1, "score_for": score_h, "score_ag": score_a})
    fix_context.append({"team": fix["team_a"], "event": gw, "fdr": diff_a,
                         "is_home": 0, "score_for": score_a, "score_ag": score_h})

fix_ctx_df = pd.DataFrame(fix_context)

player_team = players[["id","team"]].rename(columns={"id":"player_id"})
history = history.merge(player_team, on="player_id", how="left")
history = history.merge(fix_ctx_df, on=["team","event"], how="left")
history["fdr"]       = history["fdr"].fillna(3)
history["is_home"]   = history["is_home"].fillna(0)
history["score_for"] = history["score_for"].fillna(0)
history["score_ag"]  = history["score_ag"].fillna(0)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Merge player metadata onto history
# ─────────────────────────────────────────────────────────────────────────────
META_COLS = [c for c in [
    "id", "element_type", "now_cost", "selected_by_percent",
    "expected_goals_per_90", "expected_assists_per_90",
    "expected_goal_involvements_per_90", "expected_goals_conceded_per_90",
    "goals_conceded_per_90", "saves_per_90", "clean_sheets_per_90",
    "web_name", "first_name", "second_name", "team",
    "minutes", "starts", "chance_of_playing_next_round", "status", "news"
] if c in players.columns]

history = history.merge(
    players[META_COLS], left_on="player_id", right_on="id", how="left",
    suffixes=("", "_season")
)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Build complete player × GW grid (zeros for missed GWs)
# ─────────────────────────────────────────────────────────────────────────────
print("Building player × GW grid with exact data...")

grid = pd.MultiIndex.from_product(
    [all_player_ids, finished_gws_sorted],
    names=["player_id", "event"]
).to_frame(index=False)

history["player_id"] = history["player_id"].astype(int)
history["event"]     = history["event"].astype(int)

GRID_STATS = [
    "total_points", "minutes", "goals_scored", "assists", "clean_sheets",
    "goals_conceded", "own_goals", "penalties_saved", "penalties_missed",
    "yellow_cards", "red_cards", "saves", "bonus", "bps",
    "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded", "score_for", "score_ag", "fdr", "is_home"
]
GRID_STATS = [c for c in GRID_STATS if c in history.columns]

grid = grid.merge(
    history[["player_id","event"] + GRID_STATS].drop_duplicates(["player_id","event"]),
    on=["player_id","event"], how="left"
)

FILL_ZERO = [c for c in GRID_STATS if c not in ("fdr","is_home")]
for col in FILL_ZERO:
    grid[col] = grid[col].fillna(0)
grid["fdr"]     = grid["fdr"].fillna(3)
grid["is_home"] = grid["is_home"].fillna(0)
grid = grid.sort_values(["player_id","event"]).reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# 5b. Opponent defensive strength  (data-driven replacement for raw FDR)
#
#     For each team, compute how many goals they conceded in their last 5
#     finished GWs. High value = weak defence = easier opponent.
# ─────────────────────────────────────────────────────────────────────────────
print("Computing opponent defensive strength...")

team_goals_conceded = (
    grid.groupby(["team", "event"])["score_ag"]
    .mean()
    .reset_index()
    .rename(columns={"score_ag": "goals_conceded_tgw"})
    if "team" in grid.columns
    else pd.DataFrame(columns=["team","event","goals_conceded_tgw"])
)

# merge team onto grid for this computation
grid_with_team = grid.merge(
    players[["id","team"]].rename(columns={"id":"player_id"}),
    on="player_id", how="left"
)

team_gc = (
    grid_with_team.groupby(["team","event"])["score_ag"]
    .mean()
    .reset_index()
    .rename(columns={"score_ag": "goals_conceded_tgw"})
    .sort_values(["team","event"])
)
team_gc["opp_def_weakness"] = (
    team_gc.groupby("team")["goals_conceded_tgw"]
    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
)

# map opponent weakness onto each fixture in predict_gw
opp_weakness_rows = []
gw_fix_upcoming = fixtures[fixtures["event"] == predict_gw].copy()
for _, fix in gw_fix_upcoming.iterrows():
    home_team = fix["team_h"]
    away_team = fix["team_a"]

    away_weakness_rows = team_gc[team_gc["team"] == away_team]["opp_def_weakness"]
    home_weakness_rows = team_gc[team_gc["team"] == home_team]["opp_def_weakness"]

    away_weakness = float(away_weakness_rows.iloc[-1]) if len(away_weakness_rows) > 0 else 1.0
    home_weakness = float(home_weakness_rows.iloc[-1]) if len(home_weakness_rows) > 0 else 1.0

    # home players face away team's defence, away players face home team's defence
    opp_weakness_rows.append({"team": home_team, "opp_def_weakness": away_weakness})
    opp_weakness_rows.append({"team": away_team, "opp_def_weakness": home_weakness})

opp_weakness_df = pd.DataFrame(opp_weakness_rows)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Rolling features  (shift(1) prevents leakage)
# ─────────────────────────────────────────────────────────────────────────────
print("Computing rolling features...")

ROLL_STATS = [
    "total_points", "minutes", "goals_scored", "assists", "clean_sheets",
    "goals_conceded", "saves", "bonus", "bps",
    "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded", "score_for", "score_ag"
]
ROLL_STATS = [c for c in ROLL_STATS if c in grid.columns]

for stat in ROLL_STATS:
    grp = grid.groupby("player_id")[stat]

    # 5-GW rolling window
    grid[f"{stat}_mean5"] = grp.transform(lambda x: x.shift(1).rolling(5,  min_periods=1).mean())
    grid[f"{stat}_sum5"]  = grp.transform(lambda x: x.shift(1).rolling(5,  min_periods=1).sum())
    grid[f"{stat}_std5"]  = grp.transform(lambda x: x.shift(1).rolling(5,  min_periods=1).std().fillna(0))

    # EWA: all history, recent games weighted more heavily
    grid[f"{stat}_ewm"]   = grp.transform(lambda x: x.shift(1).ewm(span=5, min_periods=1).mean())

    # Long-term baseline (19 GWs ≈ half a season)
    grid[f"{stat}_mean19"] = grp.transform(lambda x: x.shift(1).rolling(19, min_periods=1).mean())

    # Trend: positive = improving, negative = declining
    grid[f"{stat}_trend"]  = grid[f"{stat}_mean5"] - grid[f"{stat}_mean19"]

# Overall points trend
grid["pts_trend"] = grid["total_points_mean5"] - grid["total_points_mean19"]

# xG last GW — expose previous GW raw xG directly so XGBoost sees
# "how many expected goals did this player generate last week"
for xg_col in ["expected_goals", "expected_assists", "expected_goal_involvements",
                "expected_goals_conceded"]:
    if xg_col in grid.columns:
        grid[f"{xg_col}_last"] = grid.groupby("player_id")[xg_col].transform(
            lambda x: x.shift(1)
        )

# ─────────────────────────────────────────────────────────────────────────────
# 7. Merge player metadata + transfer momentum + position-specific features
# ─────────────────────────────────────────────────────────────────────────────
PER90_FEATURES = [c for c in [
    "expected_goals_per_90", "expected_assists_per_90",
    "expected_goal_involvements_per_90", "expected_goals_conceded_per_90",
    "goals_conceded_per_90", "saves_per_90", "clean_sheets_per_90"
] if c in players.columns]

grid = grid.merge(
    players[["id","element_type","now_cost","selected_by_percent"] + PER90_FEATURES],
    left_on="player_id", right_on="id", how="left"
)

# Interaction features
grid["pts_x_fdr"]   = grid["total_points_mean5"] * (6 - grid["fdr"])
grid["home_x_form"] = grid["is_home"] * grid["total_points_mean5"]

# Position-specific features
grid["cs_form"]    = grid["clean_sheets_mean5"] * grid["element_type"].isin([1,2]).astype(int)
grid["gi_form"]    = grid["expected_goal_involvements_mean5"] * grid["element_type"].isin([3,4]).astype(int)
grid["saves_form"] = grid["saves_mean5"] * (grid["element_type"] == 1).astype(int)

# Transfer momentum — crowd wisdom signal
# Normalise to -1 → +1 so it doesn't dwarf other features
transfer_momentum = players[["id", "transfers_in_event", "transfers_out_event"]].copy() \
    if "transfers_in_event" in players.columns else None

if transfer_momentum is not None:
    transfer_momentum["id"] = pd.to_numeric(transfer_momentum["id"], errors="coerce")
    transfer_momentum["transfers_in_event"]  = pd.to_numeric(transfer_momentum["transfers_in_event"],  errors="coerce").fillna(0)
    transfer_momentum["transfers_out_event"] = pd.to_numeric(transfer_momentum["transfers_out_event"], errors="coerce").fillna(0)
    transfer_momentum["transfer_momentum"]   = (
        transfer_momentum["transfers_in_event"] - transfer_momentum["transfers_out_event"]
    )
    max_momentum = transfer_momentum["transfer_momentum"].abs().max()
    transfer_momentum["transfer_momentum_norm"] = (
        transfer_momentum["transfer_momentum"] / max_momentum
        if max_momentum > 0 else 0.0
    )
    grid = grid.merge(
        transfer_momentum[["id","transfer_momentum_norm"]],
        left_on="player_id", right_on="id", how="left"
    )
    grid["transfer_momentum_norm"] = grid["transfer_momentum_norm"].fillna(0.0)
else:
    grid["transfer_momentum_norm"] = 0.0

# ─────────────────────────────────────────────────────────────────────────────
# 8. Define feature sets
# ─────────────────────────────────────────────────────────────────────────────
COMMON_ROLL = (
    [f"{s}_mean5"  for s in ROLL_STATS] +
    [f"{s}_sum5"   for s in ROLL_STATS] +
    [f"{s}_std5"   for s in ROLL_STATS] +
    [f"{s}_ewm"    for s in ROLL_STATS] +
    [f"{s}_mean19" for s in ROLL_STATS] +
    [f"{s}_trend"  for s in ROLL_STATS] +
    [f"{c}_last" for c in [
        "expected_goals", "expected_assists",
        "expected_goal_involvements", "expected_goals_conceded"
    ]]
)
COMMON_META = [
    "fdr", "is_home", "now_cost", "pts_x_fdr", "home_x_form", "pts_trend",
    "transfer_momentum_norm"
] + PER90_FEATURES

# Position-specific extra features
POS_EXTRA = {
    1: [                                            # GK
        "cs_form", "saves_form",
        "clean_sheets_per_90", "saves_per_90",
        "expected_goals_conceded_last",
        "expected_goals_conceded_ewm",
        "opp_def_weakness",
    ],
    2: [                                            # DEF
        "cs_form",
        "clean_sheets_per_90", "goals_conceded_per_90",
        "expected_goals_conceded_last",
        "expected_goals_conceded_ewm",
        "expected_goals_last",
        "expected_assists_last",
        "opp_def_weakness",
    ],
    3: [                                            # MID
        "gi_form",
        "expected_goals_per_90", "expected_assists_per_90",
        "expected_goals_last",
        "expected_assists_last",
        "expected_goal_involvements_last",
        "expected_goals_ewm",
        "expected_goal_involvements_ewm",
        "opp_def_weakness",
    ],
    4: [                                            # FWD
        "gi_form",
        "expected_goals_per_90", "expected_assists_per_90",
        "expected_goals_last",
        "expected_assists_last",
        "expected_goal_involvements_last",
        "expected_goals_ewm",
        "expected_goal_involvements_ewm",
        "opp_def_weakness",
    ],
}

BASE_FEATURES = COMMON_ROLL + COMMON_META
TARGET        = "total_points"
MIN_MINUTES   = 1

# ─────────────────────────────────────────────────────────────────────────────
# 9. Train one model per position  +  validate on 20% GWs
# ─────────────────────────────────────────────────────────────────────────────
print("\nTraining per-position XGBoost models...")

model_df = grid[grid["event"] > finished_gws_sorted[0]].copy()
model_df = model_df[model_df["minutes_mean5"] > MIN_MINUTES]
# opp_def_weakness is prediction-time only — exclude from training dropna
train_base = [c for c in BASE_FEATURES + [TARGET] if c in model_df.columns]
model_df = model_df.dropna(subset=train_base)
model_df["opp_def_weakness"] = 1.0   # neutral placeholder during training

POS_LABELS = {1:"GK", 2:"DEF", 3:"MID", 4:"FWD"}
models     = {}
pos_maes   = {}
val_frames = []

overall_train_rows = 0
overall_val_rows   = 0

for pos, pos_label in POS_LABELS.items():
    pos_df = model_df[model_df["element_type"] == pos].copy()
    if pos_df.empty:
        print(f"  {pos_label}: no data — skipping")
        continue

    # Clip training target to reduce outlier haul influence
    pos_df[TARGET] = pos_df[TARGET].clip(upper=CLIP_TARGET[pos])

    extra = [c for c in POS_EXTRA.get(pos, []) if c in pos_df.columns]
    feats = list(dict.fromkeys(
        c for c in BASE_FEATURES + extra if c in pos_df.columns
    ))

    train_p = pos_df[pos_df["event"].isin(train_gws)].dropna(subset=feats + [TARGET])
    val_p   = pos_df[pos_df["event"].isin(val_gws)].dropna(subset=feats + [TARGET])

    X_tr = train_p[feats].astype(float)
    y_tr = train_p[TARGET].astype(float)
    X_v  = val_p[feats].astype(float)
    y_v  = val_p[TARGET].astype(float)

    overall_train_rows += len(train_p)
    overall_val_rows   += len(val_p)

    # Overprediction fixes: max_depth=3, min_child_weight=10
    m = XGBRegressor(
        n_estimators=800, learning_rate=0.03, max_depth=3,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        objective="reg:squarederror", random_state=42, verbosity=0
    )

    if len(X_v) > 0:
        m.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)
        val_preds     = m.predict(X_v).clip(min=0)
        mae           = mean_absolute_error(y_v, val_preds)
        pos_maes[pos] = mae

        vf = val_p[["player_id","event",TARGET]].copy()
        vf["predicted"]    = val_preds
        vf["error"]        = (vf["predicted"] - vf[TARGET]).abs()
        vf["element_type"] = pos
        val_frames.append(vf)

        print(f"  {pos_label:<3}  train={len(train_p):,}  val={len(val_p):,}  MAE={mae:.3f} pts")
    else:
        m.fit(X_tr, y_tr, verbose=False)
        print(f"  {pos_label:<3}  train={len(train_p):,}  val=0  (no val GWs yet)")

    # Retrain on ALL data for final predictions
    all_p = pos_df.dropna(subset=feats + [TARGET])
    m.fit(all_p[feats].astype(float), all_p[TARGET].astype(float), verbose=False)

    models[pos] = {"model": m, "features": feats, "label": pos_label}

n_train_rows = overall_train_rows
n_val_rows   = overall_val_rows
total_rows   = n_train_rows + n_val_rows
train_pct    = 100 * n_train_rows / total_rows if total_rows else 0
val_pct      = 100 * n_val_rows   / total_rows if total_rows else 0

print(f"\n  ── Overall split ────────────────────────────────────")
print(f"  Train rows : {n_train_rows:,}  ({train_pct:.1f}%)")
print(f"  Val rows   : {n_val_rows:,}  ({val_pct:.1f}%)")
if pos_maes:
    overall_mae = np.mean(list(pos_maes.values()))
    print(f"  Overall MAE (avg across positions): {overall_mae:.3f} pts")
print(f"  ────────────────────────────────────────────────────")

# ─────────────────────────────────────────────────────────────────────────────
# 10. Per-player validation summary
# ─────────────────────────────────────────────────────────────────────────────
val_summary_out   = None
val_breakdown_out = None

if val_frames:
    val_all      = pd.concat(val_frames, ignore_index=True)
    player_names = players[["id","web_name"]].copy()
    player_names["id"] = pd.to_numeric(player_names["id"], errors="coerce")

    per_player_val = (
        val_all.groupby("player_id")
        .agg(
            actual_avg    = (TARGET,        "mean"),
            predicted_avg = ("predicted",   "mean"),
            mae           = ("error",       "mean"),
            gws_eval      = ("event",       "count"),
            element_type  = ("element_type","first"),
        )
        .reset_index()
        .merge(player_names, left_on="player_id", right_on="id", how="left")
        .rename(columns={
            "web_name"    : "Player",
            "element_type": "Pos",
            "actual_avg"  : "AvgActualPts",
            "predicted_avg":"AvgPredPts",
            "mae"         : "MAE",
            "gws_eval"    : "GWsEval",
        })
    )
    per_player_val["Pos"]          = per_player_val["Pos"].map(POS_LABELS).fillna("?")
    per_player_val["AvgActualPts"] = per_player_val["AvgActualPts"].round(2)
    per_player_val["AvgPredPts"]   = per_player_val["AvgPredPts"].round(2)
    per_player_val["MAE"]          = per_player_val["MAE"].round(3)

    val_summary_out = os.path.join("data", "predictions", f"fpl_validation_summary_gw{predict_gw}.csv")
    per_player_val[["Player","Pos","GWsEval","AvgActualPts","AvgPredPts","MAE"]].to_csv(
        val_summary_out, index=False
    )
    print(f"\n  Per-player validation saved → {val_summary_out}")

# ─────────────────────────────────────────────────────────────────────────────
# 11. Build GW fixture context for predict_gw
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nBuilding predictions for GW {predict_gw}...")

gw_fix  = fixtures[fixtures["event"] == predict_gw].copy()
home_gw = gw_fix[["team_h","team_h_difficulty"]].rename(columns={"team_h":"team","team_h_difficulty":"fdr"})
home_gw["is_home"] = 1
away_gw = gw_fix[["team_a","team_a_difficulty"]].rename(columns={"team_a":"team","team_a_difficulty":"fdr"})
away_gw["is_home"] = 0

gw_teams = pd.concat([home_gw, away_gw], ignore_index=True)
gw_teams["num_fixtures"] = gw_teams.groupby("team")["fdr"].transform("count")
gw_teams = gw_teams.sort_values("fdr").drop_duplicates("team")

# ─────────────────────────────────────────────────────────────────────────────
# 12. Rolling state snapshot at last_finished_gw
# ─────────────────────────────────────────────────────────────────────────────
roll_cols = (
    [f"{s}_mean5"  for s in ROLL_STATS] +
    [f"{s}_sum5"   for s in ROLL_STATS] +
    [f"{s}_std5"   for s in ROLL_STATS] +
    [f"{s}_ewm"    for s in ROLL_STATS] +
    [f"{s}_mean19" for s in ROLL_STATS] +
    [f"{s}_trend"  for s in ROLL_STATS] +
    [f"{c}_last" for c in [
        "expected_goals", "expected_assists",
        "expected_goal_involvements", "expected_goals_conceded"
    ]] +
    ["pts_trend", "cs_form", "gi_form", "saves_form", "transfer_momentum_norm"]
)
roll_cols = [c for c in roll_cols if c in grid.columns]

latest = grid[grid["event"] == last_finished_gw][["player_id"] + roll_cols].copy()

# Fallback for players whose team had a blank GW
missing = set(all_player_ids) - set(latest["player_id"].unique())
if missing:
    fallback = (
        grid[grid["player_id"].isin(missing)]
        .sort_values("event").groupby("player_id").last().reset_index()
        [["player_id"] + roll_cols]
    )
    latest = pd.concat([latest, fallback], ignore_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# 13. Availability score — injury-aware
#
#     Priority order:
#       1. status = 'i' / 'u' / 's'  → 0.0  (hard unavailable)
#       2. chance_of_playing_next_round is set → use it directly
#       3. Fallback to minutes-based score
# ─────────────────────────────────────────────────────────────────────────────
latest["avg_minutes_last5"] = latest["minutes_sum5"] / 5

# Merge injury fields from players onto latest snapshot
player_availability = players[["id", "chance_of_playing_next_round", "status"]].copy()
player_availability["id"] = pd.to_numeric(player_availability["id"], errors="coerce")
player_availability["chance_of_playing_next_round"] = pd.to_numeric(
    player_availability["chance_of_playing_next_round"], errors="coerce"
)

latest = latest.merge(
    player_availability,
    left_on="player_id", right_on="id", how="left"
)

def compute_availability(row):
    status  = row.get("status", "a")
    chance  = row.get("chance_of_playing_next_round", None)
    avg_min = row.get("avg_minutes_last5", 0)

    # Hard unavailable — injured, suspended, or unknown status
    if status in ("i", "u", "s"):
        return 0.0

    # FPL has given an explicit chance of playing — use it directly
    if pd.notna(chance) and chance is not None:
        return float(chance) / 100.0

    # No injury news — fall back to minutes-based score
    if avg_min >= 75: return 1.00
    if avg_min >= 60: return 0.90
    if avg_min >= 45: return 0.75
    if avg_min >= 30: return 0.55
    if avg_min >= 15: return 0.35
    return 0.10

latest["availability_score"] = latest.apply(compute_availability, axis=1)

# ─────────────────────────────────────────────────────────────────────────────
# 14. Assemble prediction dataframe and predict per position
# ─────────────────────────────────────────────────────────────────────────────
pred_df = players[META_COLS].copy()
pred_df = pred_df.merge(latest, left_on="id", right_on="player_id", how="left")
pred_df = pred_df.merge(gw_teams[["team","fdr","is_home","num_fixtures"]], on="team", how="left")

# Merge opponent defensive strength
pred_df = pred_df.merge(opp_weakness_df, on="team", how="left")
pred_df["opp_def_weakness"] = pred_df["opp_def_weakness"].fillna(1.0)

pred_df["fdr"]               = pd.to_numeric(pred_df["fdr"],          errors="coerce").fillna(3)
pred_df["is_home"]            = pd.to_numeric(pred_df["is_home"],      errors="coerce").fillna(0)
pred_df["num_fixtures"]       = pd.to_numeric(pred_df["num_fixtures"], errors="coerce").fillna(0)
pred_df["avg_minutes_last5"]  = pred_df["avg_minutes_last5"].fillna(0)
pred_df["availability_score"] = pred_df["availability_score"].fillna(0.10)

extra_cols = roll_cols + PER90_FEATURES + [
    "now_cost","pts_x_fdr","home_x_form","pts_trend",
    "cs_form","gi_form","saves_form","transfer_momentum_norm","opp_def_weakness"
]
for col in extra_cols:
    if col not in pred_df.columns:
        pred_df[col] = 0.0
    pred_df[col] = pd.to_numeric(pred_df[col], errors="coerce").fillna(0)

pred_df["pts_x_fdr"]   = pred_df["total_points_mean5"] * (6 - pred_df["fdr"])
pred_df["home_x_form"] = pred_df["is_home"] * pred_df["total_points_mean5"]

# Predict using each position's model
pred_df["raw_predicted_points"] = 0.0

for pos, info in models.items():
    mask  = pred_df["element_type"] == pos
    feats = info["features"]

    for c in feats:
        if c not in pred_df.columns:
            pred_df[c] = 0.0
        pred_df[c] = pd.to_numeric(pred_df[c], errors="coerce").fillna(0)

    feats = list(dict.fromkeys(feats))
    X_p   = pred_df.loc[mask, feats].astype(float)
    if len(X_p) > 0:
        raw = info["model"].predict(X_p).clip(min=0)
        # Cap predictions per position to prevent unrealistic haul predictions
        raw = np.clip(raw, 0, MAX_POINTS[pos])
        pred_df.loc[mask, "raw_predicted_points"] = raw

pred_df["raw_predicted_points"] = pred_df["raw_predicted_points"].round(2)
pred_df.loc[pred_df["num_fixtures"] == 0, "raw_predicted_points"] = 0.0

# Double Gameweek multiplier — position-specific boost for two fixtures
# GK/DEF benefit less (clean sheet still binary per game)
# MID/FWD benefit more (double chance to score/assist)
DGW_MULTIPLIER = {1: 1.6, 2: 1.7, 3: 1.8, 4: 1.9}
for pos, multiplier in DGW_MULTIPLIER.items():
    dgw_mask = (pred_df["num_fixtures"] >= 2) & (pred_df["element_type"] == pos)
    pred_df.loc[dgw_mask, "raw_predicted_points"] = (
        pred_df.loc[dgw_mask, "raw_predicted_points"] * multiplier
    ).clip(upper=20)

pred_df["predicted_points"] = (
    pred_df["raw_predicted_points"] * pred_df["availability_score"]
).round(2)

pred_df["price_m"] = pred_df["now_cost"] / 10
pred_df["value"]   = (pred_df["predicted_points"] / pred_df["price_m"].clip(lower=0.1)).round(3)

# ─────────────────────────────────────────────────────────────────────────────
# 14b. Price change signal
#
#      Simple signal based on transfer momentum and recent form.
#      ↑ Rise likely  : high positive momentum + decent form
#      ↓ Fall likely  : heavy net transfers out or very poor form
#      → Stable       : everything else
# ─────────────────────────────────────────────────────────────────────────────
def price_change_signal(row):
    momentum = row.get("transfer_momentum_norm", 0)
    pts      = row.get("total_points_mean5", 0)
    if momentum > 0.3 and pts > 4:
        return "↑ Rise likely"
    elif momentum < -0.3 or pts < 1:
        return "↓ Fall likely"
    else:
        return "→ Stable"

pred_df["price_signal"] = pred_df.apply(price_change_signal, axis=1)

# ─────────────────────────────────────────────────────────────────────────────
# 15. Debug trace
# ─────────────────────────────────────────────────────────────────────────────
if args.debug:
    mask = pred_df["web_name"].str.contains(args.debug, case=False, na=False)
    if mask.any():
        r = pred_df[mask].iloc[0]
        pos_label = POS_LABELS.get(int(r.get("element_type", 3)), "?")
        print(f"\n{'─'*60}")
        print(f"  PLAYER TRACE : {r['web_name']}  ({pos_label})")
        print(f"{'─'*60}")
        print(f"  Rolling window            : GWs {finished_gws_sorted[-5:]}")
        print(f"  avg_minutes_last5         : {r['avg_minutes_last5']:.1f} min/GW  (EXACT)")
        print(f"  status                    : {r.get('status','?')}")
        print(f"  chance_of_playing         : {r.get('chance_of_playing_next_round','?')}")
        print(f"  availability_score        : {r['availability_score']}")
        print(f"  total_points_mean5        : {r['total_points_mean5']:.2f}  (EXACT FPL pts)")
        print(f"  goals_scored_mean5        : {r['goals_scored_mean5']:.2f}")
        print(f"  assists_mean5             : {r['assists_mean5']:.2f}")
        print(f"  clean_sheets_mean5        : {r['clean_sheets_mean5']:.2f}")
        print(f"  transfer_momentum_norm    : {r.get('transfer_momentum_norm', 0):.3f}")
        print(f"  opp_def_weakness          : {r.get('opp_def_weakness', 1.0):.3f}")
        print(f"  fdr                       : {r['fdr']}")
        print(f"  is_home                   : {r['is_home']}")
        print(f"  num_fixtures              : {r['num_fixtures']}")
        print(f"  price_signal              : {r.get('price_signal','?')}")
        print(f"  raw_predicted_points      : {r['raw_predicted_points']}")
        print(f"  × availability_score      : {r['availability_score']}")
        print(f"  = predicted_points        : {r['predicted_points']}")
        print(f"{'─'*60}")
        pid   = int(r["id"])
        last5 = grid[
            (grid["player_id"] == pid) &
            (grid["event"].isin(finished_gws_sorted[-5:]))
        ][["event","total_points","minutes","goals_scored","assists","clean_sheets","bonus"]]
        last5.columns = ["GW","Pts","Min","GS","Ast","CS","Bon"]
        print(f"\n  Last 5 GW breakdown (EXACT from API):")
        print(last5.to_string(index=False))
        print()
    else:
        print(f"\n  Debug: '{args.debug}' not found")

# ─────────────────────────────────────────────────────────────────────────────
# 16. Output — all players + best per position + price signals
# ─────────────────────────────────────────────────────────────────────────────
DISPLAY_COLS = [c for c in [
    "web_name","element_type","price_m","fdr","is_home","num_fixtures",
    "avg_minutes_last5","availability_score",
    "raw_predicted_points","predicted_points","value",
    "total_points_mean5","selected_by_percent","price_signal"
] if c in pred_df.columns]

results = (
    pred_df[DISPLAY_COLS]
    .sort_values("predicted_points", ascending=False)
    .rename(columns={
        "web_name"             : "Player",
        "element_type"         : "Pos",
        "price_m"              : "Price(£m)",
        "fdr"                  : "FDR",
        "is_home"              : "Home",
        "num_fixtures"         : "Fixtures",
        "avg_minutes_last5"    : "AvgMin(L5)",
        "availability_score"   : "AvailScore",
        "raw_predicted_points" : "RawPts",
        "predicted_points"     : "PredPts",
        "value"                : "Value",
        "total_points_mean5"   : "AvgPts(L5)",
        "selected_by_percent"  : "Sel%",
        "price_signal"         : "PriceSignal",
    })
)
results["Pos"] = results["Pos"].map(POS_LABELS).fillna("?")

# Best players per position
TOP_N = 10
best_by_pos = []
for pos_label in ["GK","DEF","MID","FWD"]:
    top = results[results["Pos"] == pos_label].head(TOP_N).copy()
    top.insert(0, "Rank", range(1, len(top)+1))
    best_by_pos.append(top)

print(f"\n{'='*80}")
print(f"  BEST PLAYERS BY POSITION — GW {predict_gw}")
print(f"{'='*80}")
for pos_label, top in zip(["GK","DEF","MID","FWD"], best_by_pos):
    pos_mae_str = f"  (pos MAE: {pos_maes[{'GK':1,'DEF':2,'MID':3,'FWD':4}[pos_label]]:.3f} pts)" \
                  if {'GK':1,'DEF':2,'MID':3,'FWD':4}[pos_label] in pos_maes else ""
    print(f"\n  ── {pos_label}{pos_mae_str} {'─'*50}")
    print(top[["Rank","Player","Price(£m)","FDR","Home","Fixtures",
               "AvgMin(L5)","AvailScore","RawPts","PredPts","Value","PriceSignal"]].to_string(index=False))

# Save all predictions
pred_out = os.path.join("data", "predictions", f"fpl_predictions_gw{predict_gw}.csv")
results.to_csv(pred_out, index=False)

# Save best by position
best_out = os.path.join("data", "predictions", f"fpl_best_by_position_gw{predict_gw}.csv")
pd.concat(best_by_pos).to_csv(best_out, index=False)

# Save price signals — sorted by momentum descending
price_signal_cols = [c for c in [
    "web_name","element_type","price_m","transfer_momentum_norm",
    "total_points_mean5","price_signal"
] if c in pred_df.columns]

price_out = os.path.join("data", "predictions", f"fpl_price_signals_gw{predict_gw}.csv")
(
    pred_df[price_signal_cols]
    .rename(columns={
        "web_name"                : "Player",
        "element_type"            : "Pos",
        "price_m"                 : "Price(£m)",
        "transfer_momentum_norm"  : "Momentum",
        "total_points_mean5"      : "AvgPts(L5)",
        "price_signal"            : "PriceSignal",
    })
    .assign(Pos=lambda df: df["Pos"].map(POS_LABELS).fillna("?"))
    .sort_values("Momentum", ascending=False)
    .to_csv(price_out, index=False)
)

# ─────────────────────────────────────────────────────────────────────────────
# 17. Training summary
# ─────────────────────────────────────────────────────────────────────────────
summary_path = os.path.join("data", "predictions", "fpl_training_summary.txt")
with open(summary_path, "w") as f:
    f.write("FPL Model Training Summary\n==========================\n\n")
    f.write(f"Predicting GW     : {predict_gw}\n")
    f.write(f"Data source       : FPL API — exact per-GW history from element-summary\n\n")
    f.write(f"Finished GWs      : GW{finished_gws_sorted[0]}–GW{last_finished_gw} ({n_gws} total)\n")
    f.write(f"Train GWs (80%)   : GW{train_gws[0]}–GW{train_gws[-1]} ({len(train_gws)} GWs)\n")
    if val_gws:
        f.write(f"Val GWs   (20%)   : GW{val_gws[0]}–GW{val_gws[-1]} ({len(val_gws)} GWs)\n\n")
    f.write(f"Active filter     : avg minutes > {MIN_MINUTES} min/GW in rolling window\n")
    f.write(f"Train rows        : {n_train_rows:,} ({train_pct:.1f}%)\n")
    f.write(f"Val rows          : {n_val_rows:,} ({val_pct:.1f}%)\n\n")
    f.write(f"Overprediction controls:\n")
    f.write(f"  max_depth=3, min_child_weight=10\n")
    f.write(f"  Prediction cap  : GK/DEF={MAX_POINTS[1]}, MID/FWD={MAX_POINTS[3]}\n")
    f.write(f"  Target clip     : GK/DEF={CLIP_TARGET[1]}, MID/FWD={CLIP_TARGET[3]}\n\n")
    f.write(f"Per-position MAE on validation GWs:\n")
    for pos, label in POS_LABELS.items():
        mae_str = f"{pos_maes[pos]:.3f} pts" if pos in pos_maes else "n/a"
        f.write(f"  {label:<4}: {mae_str}\n")
    if pos_maes:
        f.write(f"  Overall avg: {np.mean(list(pos_maes.values())):.3f} pts\n")

print(f"\n✅  All players      → {pred_out}")
print(f"✅  Best by position → {best_out}")
print(f"✅  Price signals    → {price_out}")
if val_summary_out:
    print(f"✅  Validation       → {val_summary_out}")
print(f"✅  Summary          → {summary_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 18. Export to S3
# ─────────────────────────────────────────────────────────────────────────────
if args.export:
    if not args.s3_bucket:
        print("\n⚠️  --export requires --s3-bucket <bucket-name>")
    else:
        bucket = args.s3_bucket
        gw_tag = f"gw{predict_gw}"
        print(f"\nUploading to s3://{bucket} ...")

        upload_file_to_s3(pred_out,     f"predictions/fpl_predictions_{gw_tag}.csv",       bucket)
        upload_file_to_s3(best_out,     f"predictions/fpl_best_by_position_{gw_tag}.csv",  bucket)
        upload_file_to_s3(summary_path, f"predictions/fpl_training_summary.txt",           bucket)
        upload_file_to_s3(price_out,    f"predictions/fpl_price_signals_{gw_tag}.csv",     bucket)
        if val_summary_out:
            upload_file_to_s3(val_summary_out,
                              f"predictions/fpl_validation_{gw_tag}.csv", bucket)

        with open(CACHE_BOOTSTRAP) as f:
            bs_obj = json.load(f)
        with open(CACHE_FIXTURES) as f:
            fx_obj = json.load(f)
        upload_json_to_s3(bs_obj,  "cache/bootstrap.json",           bucket)
        upload_json_to_s3(fx_obj,  "cache/fixtures.json",            bucket)
        upload_df_to_s3(history,   "cache/player_history.parquet",   bucket, fmt="parquet")
        upload_json_to_s3({"last_finished_gw": int(last_finished_gw)},
                          "cache/meta.json", bucket)

        print(f"\n✅  All files uploaded to s3://{bucket}")
        print(f"   Lambda chatbot reads: predictions/fpl_best_by_position_{gw_tag}.csv")

print(f"\nRun again next GW — API data updates automatically each week.")
print(f"To trace a player : python fpl_predictor.py --debug \"Salah\"")
print(f"To upload to S3   : python fpl_predictor.py --export --s3-bucket your-bucket-name")