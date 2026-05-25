# %% [markdown]
# # wOBA Projection System Review
#
# Cell-style Python verification workflow for hitter projection systems.
# Run cells individually in VS Code, or run the whole file as a script.

# %%
from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
PROJECTION_SYSTEMS_DIR = ROOT_DIR / "projection_systems" / "hitters" / "2025"
CACHE_FILE = ROOT_DIR / "data" / "cache" / "mlbam_id_cache.csv"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from pybaseball import batting_stats, playerid_lookup, playerid_reverse_lookup
except ImportError:
    batting_stats = playerid_lookup = playerid_reverse_lookup = None

pd.set_option("display.max_columns", None)

# %% [markdown]
# ## Config

# %%
SEASON = 2025
MISSING_PLAYER_MULTIPLIERS = (1.00, 0.95, 0.90)
ACTUAL_STATS_FILE = ROOT_DIR / "data" / "raw" / "fangraphs" / "batting_actuals_2025_qual1.csv"
HISTORICAL_STATS_FILE = ROOT_DIR / "data" / "raw" / "fangraphs" / "batting_actuals_2018_2025_qual1.csv"
ALLOW_LIVE_ID_LOOKUPS = False

SYSTEM_FILES = {
    "OOPSY": "OOPSY 2025.csv",
    "THE_BAT_X": "THE BAT X 2025.csv",
    "Steamer": "Steamer 2025.csv",
    "ZiPS": "ZiPS 2025.csv",
    "ATC": "ATC 2025.csv",
}

MODEL_PROJECTION_NAME = "XGBoost Hitter Projections"
XGB_FILE = Path(
    os.environ.get(
        "XGB_FILE",
        ROOT_DIR / "projection_systems" / "hitters" / "2025" / "toms_hitter_projections_2025.csv",
    )
)

# To avoid live FanGraphs fetches, set `ACTUAL_STATS_FILE` to a local CSV/parquet
# with at least these columns: IDfg, Season, Name, Team, Age, PA, wOBA.
# Example:
# ACTUAL_STATS_FILE = ROOT_DIR / "data" / "raw" / "fangraphs" / "batting_actuals_2025.csv"

# %% [markdown]
# ## Load Projections

# %%
systems = {
    name: pd.read_csv(PROJECTION_SYSTEMS_DIR / filename)
    for name, filename in SYSTEM_FILES.items()
}
toms_hitter_projections = pd.read_csv(XGB_FILE)

# %% [markdown]
# ## Load Actual Results And Player IDs

# %%
def load_actual_batting_stats(season: int, stats_file=None) -> pd.DataFrame:
    if stats_file:
        path = Path(stats_file)
        if path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)

        if "Season" in df.columns:
            df = df[df["Season"] == season].copy()
        return df

    try:
        if batting_stats is None:
            raise RuntimeError(
                "pybaseball is not installed. Set ACTUAL_STATS_FILE to a local "
                "CSV/parquet file, or install pybaseball for live FanGraphs fetches."
            )
        return batting_stats(season, season, qual=1)
    except RuntimeError as exc:
        if "status code 403" not in str(exc):
            raise
        raise RuntimeError(
            "FanGraphs blocked the live batting_stats request with HTTP 403. "
            "Set ACTUAL_STATS_FILE in the config cell to a frozen actuals CSV/parquet "
            "and rerun the verifier. Required columns: IDfg, Season, Name, Team, Age, PA, wOBA."
        ) from exc


actual = load_actual_batting_stats(SEASON, ACTUAL_STATS_FILE)
ids = list(actual["IDfg"])


def load_id_data(ids, allow_live_lookup: bool = False) -> pd.DataFrame:
    cached = pd.DataFrame(columns=["key_fangraphs", "key_mlbam"])
    if CACHE_FILE.exists():
        cached = pd.read_csv(CACHE_FILE)[["key_fangraphs", "key_mlbam"]].dropna(
            subset=["key_fangraphs"]
        )

    if not allow_live_lookup:
        return cached.drop_duplicates(subset=["key_fangraphs"])

    try:
        if playerid_reverse_lookup is None:
            raise RuntimeError("pybaseball is not installed.")
        live = playerid_reverse_lookup(ids, key_type="fangraphs")
    except Exception as exc:
        print(f"Live player ID lookup failed; using cached IDs only. Error: {exc}")
        return cached.drop_duplicates(subset=["key_fangraphs"])

    existing_fg = set(live["key_fangraphs"].dropna())
    new_cached = cached[~cached["key_fangraphs"].isin(existing_fg)]
    return pd.concat(
        [
            live[["key_mlbam", "key_fangraphs"]],
            new_cached[["key_fangraphs", "key_mlbam"]],
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["key_fangraphs"])


id_data = load_id_data(ids, ALLOW_LIVE_ID_LOOKUPS)

actual = actual[["IDfg", "Season", "Name", "Team", "Age", "PA", "wOBA"]]
actual = actual.merge(
    id_data[["key_mlbam", "key_fangraphs"]],
    left_on="IDfg",
    right_on="key_fangraphs",
    how="left",
)
actual = actual[["IDfg", "key_mlbam", "Season", "Name", "Team", "Age", "PA", "wOBA"]]


def load_historical_batting_stats(path: Path, season: int) -> pd.DataFrame:
    if not path.exists():
        print(f"Historical stats file not found; rookie/veteran split will be skipped: {path}")
        return pd.DataFrame(columns=["IDfg", "Season", "PA"])

    if path.suffix.lower() == ".parquet":
        history = pd.read_parquet(path)
    else:
        history = pd.read_csv(path)

    required = {"IDfg", "Season", "PA"}
    missing = sorted(required - set(history.columns))
    if missing:
        print(f"Historical stats file is missing {missing}; rookie/veteran split will be skipped.")
        return pd.DataFrame(columns=["IDfg", "Season", "PA"])

    return history[history["Season"] < season][["IDfg", "Season", "PA"]].copy()


historical_actuals = load_historical_batting_stats(HISTORICAL_STATS_FILE, SEASON)
prior_player_ids = set(historical_actuals.loc[historical_actuals["PA"].fillna(0) > 0, "IDfg"])
actual["is_rookie"] = ~actual["IDfg"].isin(prior_player_ids)

toms_hitter_projections = toms_hitter_projections.merge(
    id_data[["key_mlbam", "key_fangraphs"]],
    left_on="IDfg",
    right_on="key_fangraphs",
    how="left",
)

# %% [markdown]
# ## Fill Missing MLBAM IDs

# %%
def fill_missing_mlbam_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing MLBAM IDs by player name lookup."""
    df = df.copy()
    missing_mask = df["key_mlbam"].isna()

    if missing_mask.any() and not ALLOW_LIVE_ID_LOOKUPS:
        print(
            f"Skipping {missing_mask.sum()} live player ID lookups because "
            "ALLOW_LIVE_ID_LOOKUPS is False."
        )
        return df

    for idx, row in df[missing_mask].iterrows():
        full_name = row["Name"]
        try:
            first, last = full_name.split(" ", 1)
        except ValueError:
            print(f"Could not split name: {full_name}")
            continue

        try:
            if playerid_lookup is None:
                raise RuntimeError("pybaseball is not installed.")
            lookup = playerid_lookup(last, first, fuzzy=True)
            if len(lookup) > 1:
                lookup = lookup[
                    (lookup["key_fangraphs"] == -1)
                    & (lookup["mlb_played_last"] == SEASON)
                ]
                if len(lookup) > 1:
                    print(f"{last} {first}")
                    print(lookup)

            if lookup.empty:
                print(f"No lookup results for {full_name}")
                continue

            lookup = lookup[lookup["key_mlbam"].notna()]
            if lookup.empty:
                print(f"No valid MLBAM ID found for {full_name}")
                continue

            df.at[idx, "key_mlbam"] = lookup.iloc[0]["key_mlbam"]
        except Exception as exc:
            print(f"Error looking up {full_name}: {exc}")

    return df


actual = fill_missing_mlbam_ids(actual)
toms_hitter_projections = fill_missing_mlbam_ids(toms_hitter_projections)

# %% [markdown]
# ## Save ID Cache

# %%
cache_data = (
    pd.concat(
        [
            df[["IDfg", "key_mlbam"]]
            .dropna(subset=["key_mlbam"])
            .rename(columns={"IDfg": "key_fangraphs"})
            for df in [actual, toms_hitter_projections]
        ]
    )
    .drop_duplicates(subset=["key_fangraphs"])
)

if CACHE_FILE.exists():
    existing_cache = pd.read_csv(CACHE_FILE).dropna(subset=["key_fangraphs"])
    cache_data = pd.concat([existing_cache, cache_data]).drop_duplicates(
        subset=["key_fangraphs"]
    )

cache_data.to_csv(CACHE_FILE, index=False)
print(f"Saved {len(cache_data)} ID mappings to cache")

# %% [markdown]
# ## Rescale Projections To Actual League wOBA

# %%
actual_league_average = (actual["wOBA"] * actual["PA"]).sum() / actual["PA"].sum()

projection_frames = {
    name: df[["MLBAMID", "wOBA"]].rename(columns={"wOBA": f"{name}_wOBA"})
    for name, df in systems.items()
}

projection_frames["toms_hitter_projections"] = toms_hitter_projections[["key_mlbam", "xgb_wOBA"]]


def attach_actual_pa(df: pd.DataFrame, projection_col: str) -> pd.DataFrame:
    if "MLBAMID" in df.columns:
        out = df.merge(actual[["key_mlbam", "PA"]], left_on="MLBAMID", right_on="key_mlbam")
    else:
        out = df.merge(actual[["key_mlbam", "PA"]], on="key_mlbam")

    out.loc[out[projection_col].isna(), "PA"] = np.nan
    return out


def adjust_league_avg(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    lg_avg = (df[col] * df["PA"]).sum() / df["PA"].sum()
    print(f"{col} league average: {lg_avg:.4f}")
    df[col] = df[col] - lg_avg + actual_league_average
    return df


adjusted_frames = {}
for name, frame in projection_frames.items():
    col = "xgb_wOBA" if name == "toms_hitter_projections" else f"{name}_wOBA"
    with_pa = attach_actual_pa(frame, col)
    adjusted = adjust_league_avg(with_pa, col)
    adjusted_frames[name] = adjusted.drop(columns=[c for c in ["MLBAMID", "PA"] if c in adjusted.columns])

# %% [markdown]
# ## Combine Actuals And Projections

# %%
woba_projections = actual.copy()
for name in SYSTEM_FILES:
    woba_projections = woba_projections.merge(
        adjusted_frames[name],
        on="key_mlbam",
        how="left",
    )

woba_projections = woba_projections.merge(adjusted_frames["toms_hitter_projections"], on="key_mlbam", how="left")

print(f"Evaluating on {len(woba_projections)} players")
toms_mask = woba_projections["xgb_wOBA"].notna()

# %% [markdown]
# ## Weighted RMSE Helpers

# %%
def weighted_rmse(actual_col, pred_col, weights):
    return np.sqrt(((actual_col - pred_col) ** 2 * weights).sum() / weights.sum())


def naive_projection_value(df: pd.DataFrame, multiplier: float = 1.00) -> float:
    return actual_league_average * multiplier


def better_than_naive(model_rmse: float, naive_rmse: float) -> float:
    distance = np.sqrt(abs((naive_rmse**2) - (model_rmse**2)))
    return distance if model_rmse <= naive_rmse else -distance


def choose_missing_projection(df: pd.DataFrame, col: str, naive_multiplier: float) -> tuple[pd.Series, float, float]:
    best_prediction = None
    best_multiplier = None
    best_rmse = np.inf

    for multiplier in MISSING_PLAYER_MULTIPLIERS:
        prediction = df[col].fillna(actual_league_average * multiplier)
        rmse = weighted_rmse(df["wOBA"], prediction, df["PA"])
        if rmse < best_rmse:
            best_prediction = prediction
            best_multiplier = multiplier
            best_rmse = rmse

    return best_prediction, best_multiplier, best_rmse


def projection_columns() -> dict[str, str]:
    return {**{name: f"{name}_wOBA" for name in SYSTEM_FILES}, MODEL_PROJECTION_NAME: "xgb_wOBA"}


def make_rmse_table(df: pd.DataFrame, label: str, naive_multiplier: float = 1.00) -> pd.DataFrame:
    pa = df["PA"]
    woba = df["wOBA"]
    naive_value = naive_projection_value(df, naive_multiplier)
    naive_rmse = weighted_rmse(woba, naive_value, pa)

    rows = [
        {
            "Model": "Naive",
            "Weighted RMSE": naive_rmse,
            "Better Than Naive": 0.0,
            "Missing Fill": naive_multiplier,
        }
    ]
    for name, col in projection_columns().items():
        prediction, missing_multiplier, rmse = choose_missing_projection(df, col, naive_multiplier)
        rows.append(
            {
                "Model": name,
                "Weighted RMSE": rmse,
                "Better Than Naive": better_than_naive(rmse, naive_rmse),
                "Missing Fill": missing_multiplier,
            }
        )

    table = pd.DataFrame(rows)
    table = table.sort_values("Weighted RMSE").reset_index(drop=True)
    table["Rank"] = table["Weighted RMSE"].rank(method="min").astype(int)
    table = table[["Rank", "Model", "Weighted RMSE", "Better Than Naive", "Missing Fill"]]
    table.attrs["label"] = label
    return table


all_players_rmse = make_rmse_table(woba_projections, "all players")
all_players_rmse

# %%
projected_players = woba_projections[toms_mask]
toms_players_rmse = make_rmse_table(projected_players, "XGBoost projected players only")
toms_players_rmse

# %% [markdown]
# ## Rookie And Veteran Splits

# %%
rookies = woba_projections[woba_projections["is_rookie"]]
veterans = woba_projections[~woba_projections["is_rookie"]]

rookies_rmse = make_rmse_table(rookies, "rookies", naive_multiplier=0.95) if len(rookies) else pd.DataFrame()
veterans_rmse = make_rmse_table(veterans, "veterans") if len(veterans) else pd.DataFrame()
rookies_rmse

# %%
veterans_rmse

# %% [markdown]
# ## Pairwise Matched-Sample Robustness Check

# %%
def make_pairwise_matched_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in SYSTEM_FILES:
        system_col = f"{name}_wOBA"
        matched = df[df["xgb_wOBA"].notna() & df[system_col].notna()].copy()
        if matched.empty:
            continue

        rows.append(
            {
                "Other Model": name,
                "Players": len(matched),
                "XGBoost RMSE": weighted_rmse(
                    matched["wOBA"], matched["xgb_wOBA"], matched["PA"]
                ),
                "Other RMSE": weighted_rmse(matched["wOBA"], matched[system_col], matched["PA"]),
            }
        )

    return pd.DataFrame(rows)


pairwise_matched_rmse = make_pairwise_matched_table(woba_projections)
pairwise_matched_rmse

# %% [markdown]
# ## Script Output

# %%
if __name__ == "__main__":
    pd.set_option("display.float_format", "{:.6f}".format)
    print("\nWeighted RMSE (all players):")
    print(all_players_rmse.to_string(index=False))
    print("\nWeighted RMSE (XGBoost projected players only):")
    print(toms_players_rmse.to_string(index=False))
    if len(rookies_rmse):
        print("\nWeighted RMSE (rookies):")
        print(rookies_rmse.to_string(index=False))
    if len(veterans_rmse):
        print("\nWeighted RMSE (veterans):")
        print(veterans_rmse.to_string(index=False))
    print("\nPairwise matched-sample RMSE:")
    print(pairwise_matched_rmse.to_string(index=False))
