"""Build hitter wOBA projections with lagged features and XGBoost models."""

from pathlib import Path
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

pd.set_option("display.max_columns", None)


# Run settings
PROJECT_SEASON = 2025
TRAINING_START = 2021
EVALUATE_AFTER_BUILD = True

HISTORICAL_STATS_FILE = ROOT_DIR / "data" / "raw" / "fangraphs" / "batting_actuals_2018_2025_qual1.csv"
DEPTH_CHARTS_FILE = ROOT_DIR / "projection_systems" / "hitters" / "2026" / "Depth Charts 2026.csv"
OUTPUT_FILE = ROOT_DIR / "projection_systems" / "hitters" / str(PROJECT_SEASON) / (
    f"toms_hitter_projections_{PROJECT_SEASON}.csv"
)
EVALUATION_CODE = ROOT_DIR / "model_building_and_testing" / "evaluation" / "reviewing_systems.py"

SAMPLE_PLAYERS = [
    ("Aaron Judge", 2023),
    ("Aaron Judge", 2024),
    ("Aaron Judge", 2025),
    ("Junior Caminero", 2025),
    ("Mike Trout", 2025),
]


# Model columns and parameters
labels = ["IDfg", "Season", "Name", "Team", "Age"]

target_variables = ["HR", "1B", "2B", "3B", "BB", "IBB", "SF", "SH", "HBP", "SO"]

features = [
    "AVG", "BB%", "K%", "BB/K",
    "OBP", "SLG", "OPS", "ISO", "BABIP",
    "GB/FB", "LD%", "GB%", "FB%", "IFFB%",
    "HR/FB", "IFH%", "wOBA",
    "Spd", "O-Swing%", "Z-Swing%", "Swing%",
    "O-Contact%", "Z-Contact%", "Contact%", "Zone%",
    "F-Strike%", "Pull%", "Cent%", "Oppo%",
    "Soft%", "Med%", "TTO%",
    "EV", "LA", "Barrel%", "HardHit%",
    "CStr%", "CSW%", "xBA", "xSLG", "xwOBA",
]

BEST_PARAMS = {
    "HR": dict(colsample_bytree=0.7, learning_rate=0.01, max_depth=3, n_estimators=500, subsample=0.7),
    "1B": dict(colsample_bytree=0.7, learning_rate=0.01, max_depth=3, n_estimators=500, subsample=0.7),
    "2B": dict(colsample_bytree=0.7, learning_rate=0.01, max_depth=3, n_estimators=500, subsample=0.7),
    "3B": dict(colsample_bytree=0.7, learning_rate=0.01, max_depth=3, n_estimators=500, subsample=0.7),
    "BB": dict(colsample_bytree=0.7, learning_rate=0.01, max_depth=3, n_estimators=500, subsample=1.0),
    "IBB": dict(colsample_bytree=1.0, learning_rate=0.05, max_depth=5, n_estimators=100, subsample=0.7),
    "SF": dict(colsample_bytree=1.0, learning_rate=0.05, max_depth=5, n_estimators=100, subsample=0.7),
    "SH": dict(colsample_bytree=1.0, learning_rate=0.05, max_depth=5, n_estimators=100, subsample=0.7),
    "HBP": dict(colsample_bytree=0.7, learning_rate=0.01, max_depth=3, n_estimators=500, subsample=0.7),
    "SO": dict(colsample_bytree=0.7, learning_rate=0.01, max_depth=3, n_estimators=500, subsample=0.7),
}

WOBA_WEIGHTS = {"BB": 0.691, "HBP": 0.722, "1B": 0.882, "2B": 1.252, "3B": 1.584, "HR": 2.037}

W1, W2, W3 = 5 / 12, 4 / 12, 3 / 12
REGRESSION_PA = 100
PEAK_AGE = 29
DECLINE_RATE = 0.003
IMPROVE_RATE = 0.006
PA_SPLIT = 950


# Small helpers for checking the work
def get_sample(df):
    rows = []
    for name, season in SAMPLE_PLAYERS:
        match = df[(df["Name"] == name) & (df["Season"] == season)]
        if not match.empty:
            rows.append(match.iloc[[0]])
        else:
            print(f"  [not found: {name} {season}]")

    if not rows:
        print("  No sample players found. Showing head() instead.")
        return df.head()

    return pd.concat(rows)


def show(df, label=""):
    if label:
        print()
        print("=" * 60)
        print(label)
        print("=" * 60)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print()
    print("Sample rows:")
    print(get_sample(df).to_string(index=False))


def require_file(path, label):
    if not Path(path).exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


# Load the batting data
def load_historical_batting_stats():
    require_file(HISTORICAL_STATS_FILE, "Historical stats file")
    if HISTORICAL_STATS_FILE.suffix.lower() == ".parquet":
        fg = pd.read_parquet(HISTORICAL_STATS_FILE)
    else:
        fg = pd.read_csv(HISTORICAL_STATS_FILE)

    fg["IDfg"] = fg["IDfg"].astype("string")
    return fg


def load_depth_charts_batting():
    require_file(DEPTH_CHARTS_FILE, "Depth Charts file")
    depth_charts = pd.read_csv(DEPTH_CHARTS_FILE)
    depth_charts = depth_charts.rename(
        columns={
            "PlayerId": "IDfg",
            "playerid": "IDfg",
            "PlayerName": "Name",
        }
    )

    required = {"IDfg", "Name", "Team", "PA"}
    missing = sorted(required - set(depth_charts.columns))
    if missing:
        raise ValueError(f"Depth Charts CSV is missing required columns: {missing}")

    depth_charts = depth_charts.rename(columns={"PA": "DC_pa"})
    depth_charts["IDfg"] = depth_charts["IDfg"].astype("string")
    return depth_charts


def add_forward_projection_rows(fg):
    max_actual_season = int(fg["Season"].max())
    if PROJECT_SEASON <= max_actual_season:
        return fg

    if PROJECT_SEASON != max_actual_season + 1:
        raise ValueError(
            f"Only the next forward season is supported. Requested {PROJECT_SEASON}, "
            f"but latest actual season is {max_actual_season}."
        )

    depth_charts = load_depth_charts_batting()
    forward_rows = depth_charts.assign(Season=PROJECT_SEASON)[["IDfg", "Season", "Name", "Team", "DC_pa"]].copy()

    cols = list(fg.columns)
    if "DC_pa" not in cols:
        insert_at = cols.index("Age") + 1 if "Age" in cols else len(cols)
        cols.insert(insert_at, "DC_pa")

    forward_rows = forward_rows.reindex(columns=cols)
    forward_rows = forward_rows.astype(fg.dtypes.to_dict(), errors="ignore")
    fg = pd.concat([fg, forward_rows], ignore_index=True)

    prior_age = fg[fg["Season"] == max_actual_season].set_index("IDfg")["Age"]
    is_projection_season = fg["Season"] == PROJECT_SEASON
    fg.loc[is_projection_season, "Age"] = fg.loc[is_projection_season, "IDfg"].map(prior_age) + 1
    return fg


FETCH_START = min(TRAINING_START - 3, PROJECT_SEASON - 7)
ACTUAL_END = PROJECT_SEASON if PROJECT_SEASON < pd.Timestamp.today().year else PROJECT_SEASON - 1


# Build the model frame
def make_lag(df, cols, lag):
    prior = df[["IDfg", "Season"] + cols].copy()
    prior["Season"] = prior["Season"] + lag
    return prior.rename(columns={col: f"{lag}Prev_{col}" for col in cols})


def marcel_proj_rate(df, stat):
    r1 = df[f"1Prev_{stat}"].fillna(0)
    r2 = df[f"2Prev_{stat}"].fillna(0)
    r3 = df[f"3Prev_{stat}"].fillna(0)
    pa1 = df["1Prev_PA"].fillna(0)
    pa2 = df["2Prev_PA"].fillna(0)
    pa3 = df["3Prev_PA"].fillna(0)

    player_events = W1 * r1 * pa1 + W2 * r2 * pa2 + W3 * r3 * pa3
    player_weighted_pa = W1 * pa1 + W2 * pa2 + W3 * pa3
    denom = player_weighted_pa.replace(0, np.nan)
    lg_rate = (
        df[f"_Lg1_{stat}"] * pa1 * W1
        + df[f"_Lg2_{stat}"] * pa2 * W2
        + df[f"_Lg3_{stat}"] * pa3 * W3
    ) / denom
    regressed_rate = (player_events + lg_rate * REGRESSION_PA) / (player_weighted_pa + REGRESSION_PA)
    age_adj = np.where(
        df["Age"] > PEAK_AGE,
        1 / (1 + DECLINE_RATE * (df["Age"] - PEAK_AGE)),
        np.where(df["Age"] < PEAK_AGE, 1 + IMPROVE_RATE * (PEAK_AGE - df["Age"]), 1),
    )
    return regressed_rate * age_adj


def build_model_frame():
    fg = load_historical_batting_stats()
    fg = fg[(fg["Season"] >= FETCH_START) & (fg["Season"] <= ACTUAL_END)].copy()
    fg = add_forward_projection_rows(fg)

    prev1 = make_lag(fg, ["PA"] + target_variables + features, 1)
    prev2 = make_lag(fg, ["PA"] + target_variables, 2)
    prev3 = make_lag(fg, ["PA"] + target_variables, 3)

    base_cols = labels + ["wOBA", "PA"] + target_variables
    if "DC_pa" in fg.columns:
        base_cols.insert(labels.index("Age") + 1, "DC_pa")

    df = fg[base_cols].copy()
    df = df[df["Season"] >= TRAINING_START].reset_index(drop=True)
    df = df.merge(prev1, on=["IDfg", "Season"], how="left")
    df = df.merge(prev2, on=["IDfg", "Season"], how="left")
    df = df.merge(prev3, on=["IDfg", "Season"], how="left")

    df[target_variables] = df[target_variables].astype(float)
    for stat in target_variables:
        mask = df[stat].notna() & df["PA"].notna()
        df.loc[mask, stat] = df.loc[mask, stat] / df.loc[mask, "PA"]

    for lag in [1, 2, 3]:
        pa_col = f"{lag}Prev_PA"
        for stat in target_variables:
            df[f"{lag}Prev_{stat}"] = df[f"{lag}Prev_{stat}"] / df[pa_col]

    lg_totals = fg[fg["PA"].notna()].groupby("Season")[target_variables + ["PA"]].sum()
    for lag in [1, 2, 3]:
        shifted = lg_totals.copy()
        shifted.index = shifted.index + lag
        shifted = shifted.rename(columns={col: f"_Lg{lag}_{col}" for col in shifted.columns})
        df = df.merge(shifted.reset_index(), on="Season", how="left")

    for lag in [1, 2, 3]:
        for stat in target_variables:
            df[f"_Lg{lag}_{stat}"] = df[f"_Lg{lag}_{stat}"] / df[f"_Lg{lag}_PA"]

    for stat in target_variables:
        df[f"marcel_{stat}_proj_rate"] = marcel_proj_rate(df, stat)

    df = df.drop(columns=[col for col in df.columns if col.startswith("_Lg")])

    pa_cols = ["1Prev_PA", "2Prev_PA", "3Prev_PA"]
    df = df[df[pa_cols].fillna(0).sum(axis=1) > 100].reset_index(drop=True)
    if "DC_pa" in df.columns:
        df = df[(df["PA"] > 75) | ((df["Season"] == PROJECT_SEASON) & df["DC_pa"].notna())].copy()
    else:
        df = df[df["PA"] > 75].copy()
    df["hist_pa"] = df[pa_cols].fillna(0).sum(axis=1)

    return fg, df


# Train one XGBoost model per component
def train_and_project(df):
    rf_features = [f"1Prev_{feature}" for feature in features] + ["Age"]
    train = df[df["Season"] < PROJECT_SEASON].copy()
    rows = df[df["Season"] == PROJECT_SEASON].copy()

    for stat in target_variables:
        print(f"Training {stat}...")
        valid = train.dropna(subset=[stat]).copy()

        lo_train = valid[valid["hist_pa"] < PA_SPLIT]
        model_lo = XGBRegressor(**BEST_PARAMS[stat], random_state=42, verbosity=0)
        model_lo.fit(lo_train[rf_features].fillna(0).infer_objects(copy=False), lo_train[stat])

        hi_features = rf_features + [f"marcel_{stat}_proj_rate"]
        hi_train = valid[valid["hist_pa"] >= PA_SPLIT]
        model_hi = XGBRegressor(**BEST_PARAMS[stat], random_state=42, verbosity=0)
        model_hi.fit(hi_train[hi_features].fillna(0).infer_objects(copy=False), hi_train[stat])

        lo_idx = rows[rows["hist_pa"] < PA_SPLIT].index
        hi_idx = rows[rows["hist_pa"] >= PA_SPLIT].index

        if len(lo_idx):
            rows.loc[lo_idx, f"xgb_proj_{stat}"] = model_lo.predict(
                rows.loc[lo_idx, rf_features].fillna(0).infer_objects(copy=False)
            )
        if len(hi_idx):
            rows.loc[hi_idx, f"xgb_proj_{stat}"] = model_hi.predict(
                rows.loc[hi_idx, hi_features].fillna(0).infer_objects(copy=False)
            )

        rows[f"xgb_proj_{stat}"] = rows[f"xgb_proj_{stat}"].clip(lower=0)

    return rows


# Turn component rates into a projection file
def add_projected_woba(rows):
    rows = rows.copy()
    rows["xgb_wOBA"] = (
        WOBA_WEIGHTS["BB"] * (rows["xgb_proj_BB"] - rows["xgb_proj_IBB"])
        + WOBA_WEIGHTS["HBP"] * rows["xgb_proj_HBP"]
        + WOBA_WEIGHTS["1B"] * rows["xgb_proj_1B"]
        + WOBA_WEIGHTS["2B"] * rows["xgb_proj_2B"]
        + WOBA_WEIGHTS["3B"] * rows["xgb_proj_3B"]
        + WOBA_WEIGHTS["HR"] * rows["xgb_proj_HR"]
    ) / (1 - rows["xgb_proj_SH"] - rows["xgb_proj_IBB"])
    return rows


def build_projection_file(rows):
    pa_col = "DC_pa" if "DC_pa" in rows.columns and rows["DC_pa"].notna().any() else "PA"
    to_save = rows[["IDfg", "Name", "Team", "Season", pa_col, "xgb_wOBA"]].dropna(
        subset=["xgb_wOBA"]
    ).copy()
    to_save = to_save.rename(columns={pa_col: "projected_PA"})

    for stat in target_variables:
        rate_col = f"xgb_proj_{stat}"
        to_save[f"xgb_{stat}_per_PA"] = rows[rate_col]
        to_save[f"xgb_{stat}"] = rows[rate_col] * rows[pa_col]

    return to_save


def save_projection_file(to_save):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    to_save.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(to_save)} projections to {OUTPUT_FILE}")


def maybe_evaluate():
    if not EVALUATE_AFTER_BUILD or PROJECT_SEASON > 2025:
        print("Skipping evaluation because this is a future projection season.")
        return

    require_file(EVALUATION_CODE, "Evaluation code")
    env = os.environ.copy()
    env["XGB_FILE"] = str(OUTPUT_FILE)

    result = subprocess.run(
        [sys.executable, str(EVALUATION_CODE)],
        cwd=ROOT_DIR,
        env=env,
        text=True,
        capture_output=True,
    )

    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("Evaluation failed.")


def main():
    print(f"Projection season: {PROJECT_SEASON}")
    print(f"Training starts:   {TRAINING_START}")
    print(f"Historical stats:  {HISTORICAL_STATS_FILE}")
    print(f"Depth Charts:      {DEPTH_CHARTS_FILE if PROJECT_SEASON > ACTUAL_END else None}")
    print(f"Output file:       {OUTPUT_FILE}")
    print(f"Evaluate after:    {EVALUATE_AFTER_BUILD and PROJECT_SEASON <= 2025}")

    fg, df = build_model_frame()
    show(df, "Model frame")

    rows = train_and_project(df)
    rows = add_projected_woba(rows)

    to_save = build_projection_file(rows)
    save_projection_file(to_save)
    maybe_evaluate()

    print()
    print("Projection summary:")
    print(to_save["xgb_wOBA"].describe().to_string())

    print()
    print("Top projected hitters by xgb_wOBA:")
    top_hitters = to_save.sort_values("xgb_wOBA", ascending=False).head(20)
    print(top_hitters[["IDfg", "Name", "Team", "Season", "projected_PA", "xgb_wOBA"]].to_string(index=False))

    return to_save, rows, df, fg


if __name__ == "__main__":
    main()
