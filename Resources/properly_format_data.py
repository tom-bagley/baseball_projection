# Import necessary libraries
import numpy as np
import pandas as pd
from pybaseball import batting_stats
import sys
import os
sys.path.append(os.path.abspath(".."))
from Resources.fg_depth_charts import fangraphs_depth_charts_batting
from scipy.optimize import minimize

def add_2026_rows(df):
    base = (
        fangraphs_depth_charts_batting
        .assign(Season=2026)
        .rename(columns={"PA": "DC_pa"})
        [["IDfg", "Season", "Name", "Team", "DC_pa"]]
        .copy()
    )

    # Build final column list (original + DC_pa)
    cols = list(df.columns)
    if "DC_pa" not in cols:
        insert_at = cols.index("Age") + 1 if "Age" in cols else len(cols)
        cols.insert(insert_at, "DC_pa")

    # Reindex to full column set
    base = base.reindex(columns=cols)

    # Match dtypes exactly where possible
    base = base.astype(df.dtypes.to_dict(), errors="ignore")

    # Concatenate
    out = pd.concat([df, base], ignore_index=True)

    cols = list(out.columns)
    if "Age" in cols and "DC_pa" in cols:
        cols.remove("DC_pa")
        cols.insert(cols.index("Age") + 1, "DC_pa")
        out = out[cols]

    return out



    

class GetData():
    def __init__(self, start_year, end_year, non_pa_list=None, pa_list=None, labels=None):
        self.start_year = start_year
        self.end_year = end_year
        self.starting_data = batting_stats(self.start_year-3, self.end_year, qual=1)
        self.non_pa_list = non_pa_list
        self.pa_list = pa_list
        self.labels = labels
        self.combined_list = labels + pa_list + non_pa_list

    def format_data_for_models (self, lags=(1,2,3), add_2026 = False):
        filtered_df = self.starting_data[self.combined_list]
        filtered_df = filtered_df.copy()
        filtered_df['IDfg'] = filtered_df['IDfg'].astype('string')

        if add_2026:
            filtered_df = add_2026_rows(filtered_df)

        df = (
            filtered_df
            .sort_values(['IDfg', 'Season'])
            .dropna(axis=1, how='all')
            .copy()
        )

        exclude = {'IDfg', 'Season', 'Name', 'Team', 'Age', 'DC_pa'}
        features = [c for c in df.columns if c not in exclude]

        

        numeric_features = (
            df
            .drop(columns=exclude, errors='ignore')
            .select_dtypes(include='number')
            .columns
            .tolist()
        )

        player_group = df.groupby('IDfg')

        lag_dfs = []

        for lag in lags:
            lag_df = (
                player_group[features]
                .shift(lag)
                .add_prefix(f'{lag}Prev_')
            )
            lag_dfs.append(lag_df)

        df = pd.concat([df] + lag_dfs, axis=1)

        league_sums = (
            df.groupby('Season')[numeric_features]
            .sum()
            .sort_index()  # IMPORTANT if seasons aren't guaranteed ordered
        )


        league_totals_3yr = (
            pd.concat(
                [
                    league_sums.shift(i).add_prefix(f'Prev_{i}yr_League_Totals_')
                    for i in range(3, 0, -1)
                ],
                axis=1
            )
            .reset_index()
        )

        # league_avgs = (
        #     df.groupby('Season')[numeric_features]
        #     .mean()
        #     .shift(1)
        #     .add_prefix('Prev_League_Avg_')
        #     .reset_index()
        # )

        df = df.merge(league_totals_3yr, on='Season', how='left')

        years = [int(y) for y in df['Season'].unique()]

        if 2025 in years and 2026 in years:
            age_2025 = df[df['Season'] == 2025].set_index('IDfg')['Age']
            df.loc[df['Season'] == 2026, 'Age'] = (
                df.loc[df['Season'] == 2026, 'IDfg'].map(age_2025) + 1
            )

        return df[df['Season'] >= self.start_year]
    
    def format_for_per_pa(self, df):
        for stat in self.pa_list:
            df[stat] = df[stat]/df['PA']
            df[f'1Prev_{stat}'] = df[f'1Prev_{stat}'] / df['1Prev_PA']
            df[f'2Prev_{stat}'] = df[f'2Prev_{stat}'] / df['2Prev_PA']
            df[f'3Prev_{stat}'] = df[f'3Prev_{stat}'] / df['3Prev_PA']

        return df
    
    def filter_by_experience(self, df, year):
        if year == 4:
            result = df.dropna(subset=['3Prev_H'])
            return result
        
        elif year < 4:
            prev_col = f"{year}Prev_H"

            filtered = df[df[prev_col].isna()]

            cols = filtered.columns
            prev_cols_to_drop = [
                col for col in cols
                if any(f'{i}Prev' in col for i in range(year, 4))
            ]

        result = filtered.drop(columns=prev_cols_to_drop, errors="ignore")

        if year > 1:
            result = result.dropna(subset=[f"{year - 1}Prev_H"])

        return result
    
    def get_year_df(self, df, year):
        season_2026 = df[df['Season'] == year]
        if year == 2026:
            cols = ['IDfg', 'Season', 'Name', 'Team', 'DC_pa']
        else:
            cols = ['IDfg', 'Season', 'Name', 'Team']
        season_2026 = season_2026.loc[
            :, cols + season_2026.columns[season_2026.columns.str.contains('Prev')].tolist()
        ]

        return season_2026
    
    
    
    


        


