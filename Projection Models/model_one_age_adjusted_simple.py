import pandas as pd
import numpy as np
import sys
import os

pd.set_option('display.max_columns', None)

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR))
from Resources.properly_format_data import GetData

features = ['PA', 'AB']
target_variables = ['BB', 'IBB', 'HBP', '1B', '2B', '3B', 'HR', 'SF', 'SH']
labels = ['IDfg', 'Season', 'Name', 'Team', 'Age']

WOBA_WEIGHTS = {'BB': 0.691, 'HBP': 0.722, '1B': 0.882, '2B': 1.252, '3B': 1.584, 'HR': 2.037}

# Hardcoded Marcel-style parameters (baseline — no optimization)
W1, W2, W3 = 5 / 12, 4 / 12, 3 / 12   # year weights (most recent first), normalized
K = 100          # regression-to-mean constant
PEAK_AGE = 29
DECLINE_RATE = 0.003
IMPROVE_RATE = 0.006


def project_stat_rate(per_pa_stats, stat):
    """Returns projected rate (per PA) for one stat."""
    player_weighted_stat = (
        W1 * per_pa_stats[f'1Prev_{stat}'] +
        W2 * per_pa_stats[f'2Prev_{stat}'] +
        W3 * per_pa_stats[f'3Prev_{stat}']
    )

    player_weighted_pa = (
        per_pa_stats['1Prev_PA'] * W1 +
        per_pa_stats['2Prev_PA'] * W2 +
        per_pa_stats['3Prev_PA'] * W3
    )

    league_rate = (
        (per_pa_stats[f'Prev_1yr_League_Totals_{stat}'] / per_pa_stats['Prev_1yr_League_Totals_PA']) * per_pa_stats['1Prev_PA'] * W1 +
        (per_pa_stats[f'Prev_2yr_League_Totals_{stat}'] / per_pa_stats['Prev_2yr_League_Totals_PA']) * per_pa_stats['2Prev_PA'] * W2 +
        (per_pa_stats[f'Prev_3yr_League_Totals_{stat}'] / per_pa_stats['Prev_3yr_League_Totals_PA']) * per_pa_stats['3Prev_PA'] * W3
    ) / player_weighted_pa

    regressed_rate = (player_weighted_stat + league_rate * K) / (player_weighted_pa + K)

    age_adj = np.where(
        per_pa_stats['Age'] > PEAK_AGE,
        1 / (1 + DECLINE_RATE * (per_pa_stats['Age'] - PEAK_AGE)),
        np.where(
            per_pa_stats['Age'] < PEAK_AGE,
            1 + IMPROVE_RATE * (PEAK_AGE - per_pa_stats['Age']),
            1
        )
    )

    return regressed_rate * age_adj


def build_full_projection(formatted_df, per_pa_stats):
    proj_df = formatted_df[labels + ['PA']].copy()

    for stat in target_variables:
        rate = project_stat_rate(per_pa_stats, stat)
        proj_df[stat] = rate * proj_df['PA']

    proj_df['AB'] = proj_df['PA'] - proj_df['BB'] - proj_df['HBP'] - proj_df['SF'] - proj_df['SH']
    proj_df['wOBA'] = (
        WOBA_WEIGHTS['BB'] * (proj_df['BB'] - proj_df['IBB']) +
        WOBA_WEIGHTS['HBP'] * proj_df['HBP'] +
        WOBA_WEIGHTS['1B'] * proj_df['1B'] +
        WOBA_WEIGHTS['2B'] * proj_df['2B'] +
        WOBA_WEIGHTS['3B'] * proj_df['3B'] +
        WOBA_WEIGHTS['HR'] * proj_df['HR']
    ) / (proj_df['AB'] + proj_df['BB'] - proj_df['IBB'] + proj_df['SF'] + proj_df['HBP'])

    return proj_df


print("Loading 2025 data...")
data_2025 = GetData(2025, 2025, features, target_variables, labels)
formatted_df = data_2025.format_data_for_models(add_2026=False).fillna(0)

per_pa_stats = formatted_df.copy()
per_pa_stats['PA_sum'] = per_pa_stats['1Prev_PA'] + per_pa_stats['2Prev_PA'] + per_pa_stats['3Prev_PA']
per_pa_stats = per_pa_stats[per_pa_stats['PA_sum'] > 75]

proj_df = build_full_projection(formatted_df, per_pa_stats)

# ── Compute wRC+ ──────────────────────────────────────────────────────────────
park_factors = pd.read_csv(os.path.join(os.path.dirname(DIR), 'Resources', 'park_factors_2025.csv'))
park_factors['PF'] = park_factors['PF_5yr'] / 100

proj_df = proj_df.merge(park_factors[['FG_Abbrev', 'PF']], left_on='Team', right_on='FG_Abbrev', how='left')
proj_df['PF'] = proj_df['PF'].fillna(1.0)

# League constants — update these annually from FanGraphs Guts page
WOBA_SCALE = 1.232   # 2024 value
LG_R_PER_PA = 0.118  # 2024 value

lg_wOBA = (proj_df['wOBA'] * proj_df['PA']).sum() / proj_df['PA'].sum()

proj_df['wRAA'] = ((proj_df['wOBA'] - lg_wOBA) / WOBA_SCALE) * proj_df['PA']
proj_df['wRC+'] = (
    (proj_df['wRAA'] / proj_df['PA'] + LG_R_PER_PA + (LG_R_PER_PA - proj_df['PF'] * LG_R_PER_PA))
    / LG_R_PER_PA
) * 100

proj_df = proj_df.drop(columns=['FG_Abbrev', 'wRAA'])

output_path = os.path.join(os.path.dirname(DIR), 'correct wOBA comparison', 'age_adjusted_simple.csv')
proj_df.to_csv(output_path, index=False)
print(f"Saved to {output_path}")
