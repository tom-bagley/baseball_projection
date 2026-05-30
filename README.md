# Baseball Projection System

This is a baseball projections system for hitters using wOBA as the main evaluation metric. The goal of this project is to create a projection system that compares to the best publicaly available systems like Steamer, ZiPS, ATC, THE BAT X, and OOPSY.

My model uses an XGBoost machine learning decision tree model similar to those used on a traditional Stuff+ model. I use three years of previous data including statcast data like bat speed and pull percentage to project home runs, walks, and hits. Then I convert those component rates into a final wOBA. I use that wOBA to evaluate the results with playing-time-weighted RMSE and compare it to other systems.

## Current 2025 Results

The table below comes from the local 2025 evaluation using actual 2025 hitter results. Lower weighted RMSE is better. These projections only use data from the 2024 season and before to create the 2025 projections. 

### All Players

![2025 all players weighted RMSE leaderboard](docs/assets/all_players_rmse.svg)

Summary: This measurement includes all hitters who had an at bat last year. There are some players who did not recieve a projection for 2025. Those players without a projection are projected for the league average. The naive projection is if every player was projected to be league average. Steamer led the all-player board in weighted RMSE, while my model finished close behind the public systems and stayed well ahead of the naive baseline.

### Veteran Split

This is where the current model shines. Among non-rookies, it finished first in this evaluation run.

![2025 veteran split weighted RMSE leaderboard](docs/assets/veteran_split_rmse.svg)

Summary: On veterans, the XGBoost model ranked first at `0.033469` weighted RMSE, narrowly outperforming THE BAT X, Steamer, ATC, ZiPS, and OOPSY.

### Pairwise Matched Samples

These checks compare only players shared by Tom's model and each public system.

![2025 pairwise matched-sample RMSE comparison](docs/assets/pairwise_matched_rmse.svg)

Summary: Matched samples show the model is in the same accuracy band as the public systems, beating OOPSY on the shared-player comparison and trailing the others by small margins.

## Top 10 Projected 2026 Hitters

Sorted by projected wOBA.

![Top 10 projected 2026 hitters by xgb_wOBA](docs/assets/top10_hitters_woba.svg)

Summary: Aaron Judge leads the model's 2026 hitter board at `0.432` projected `xgb_wOBA`, followed by Shohei Ohtani and Kyle Schwarber.

## How The Hitter Model Works

This model projects `HR`, `1B`, `2B`, `3B`, `BB`, `IBB`, `SF`, `SH`, `HBP`, and `SO` individually then combines them into a WOBA projection. Each stat is projected as a rate, so for example home runs would be measured as home runs per plate appearance. 

1. I start by creating Marcel-style baseline rates using weighted recent history, league regression, and age adjustment. Marcel is a simple projection that does well projecting forward. One weakness is that it only uses past home runs to project future home runs. My model starts with the Marcel baseline and adds more on top of it. 
2. Next I train separate XGBoost regressors for each stat. I use features like `BB%`, `K%`, `SLG`, `Barrel%`, `Groundball%`, `Chase%` to create model. Power stats will be more important for the home run model while contact stats will be more important for the walk and strikeout models. Stats like slugging percentage and strikeout percentage and even barrel percentage are fairly common in projection systems. However, my models add stats that deal with swing decisions and batted ball distribution that is unique to other systems. 
3. The main difference in my model compared to others is that I split players by recent playing time. I found that using the Marcel baseline was less effective for players who had little playing time. This is because Marcel uses only past performance in home runs to project home runs and that is volatile in small samples. My model uses other power stats to project home runs and so stabalizes faster than Marcel. I found that if a player had a lot of playing time starting from the Marcel baseline did improve the model. Therefore, I split my projections into two models. For players over 950 plate appearances in their last three seasons I used the Marcel baseline as a feature in my regression model. For players under 950 plate appearances in the last three years it was not included. 

The model does not directly predict wOBA as one black-box target. It predicts the events that create wOBA, which makes the output easier to inspect and gives the projection file more useful downstream columns.

## How Evaluation Works

The evaluation is built to compare projection systems directly using wOBA as the measurement. In order to create the most fair possible comparison:

- Each projection system is rescaled to the actual league-average wOBA environment.
- Accuracy is scored with PA-weighted RMSE, so missing badly on a full-time hitter matters more than missing on a short playing-time player.
- Missing projections are filled with a tested naive value so systems are not silently rewarded or punished only because of coverage.
- Results are shown for all players, this model's projected-player sample, rookies, veterans, and pairwise matched samples.

`Better Than Naive` is the RMSE improvement over a league-average baseline, expressed on the same wOBA-error scale. Positive is good.

## Quick Start

Follow these steps from a fresh checkout.

### 1. Clone the repo

```powershell
git clone <your-repo-url>
cd baseball_projection
```

### 2. Create a Python environment

Python 3.12 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 3. Install dependencies

```powershell
python -m pip install pandas numpy xgboost pyarrow pybaseball
```

`pybaseball` is optional for the default local evaluation path, but useful if you want live FanGraphs/player-ID lookups later.

### 4. Confirm the expected data is present

The hitter model and evaluator expect these local files:

```text
data/raw/fangraphs/batting_actuals_2018_2025_qual1.csv
data/raw/fangraphs/batting_actuals_2025_qual1.csv
data/cache/mlbam_id_cache.csv
projection_systems/hitters/2025/*.csv
```

For a future-season projection, the model also uses:

```text
projection_systems/hitters/2026/Depth Charts 2026.csv
```

### 5. Build hitter projections

```powershell
python model_building_and_testing/model_development/hitter_model.py
```

By default, the script builds 2025 hitter projections and writes:

```text
projection_systems/hitters/2025/toms_hitter_projections_2025.csv
```

The main settings live near the top of [hitter_model.py](model_building_and_testing/model_development/hitter_model.py):

| Setting | What it controls |
| --- | --- |
| `PROJECT_SEASON` | Season to project |
| `TRAINING_START` | First season included in the training frame |
| `HISTORICAL_STATS_FILE` | FanGraphs-style historical batting data |
| `DEPTH_CHARTS_FILE` | Playing-time source for future seasons |
| `OUTPUT_FILE` | Projection CSV destination |
| `EVALUATE_AFTER_BUILD` | Whether to run the evaluation script after building |

### 6. Run the evaluation directly

```powershell
python model_building_and_testing/evaluation/reviewing_systems.py
```

The evaluator compares Tom's model against the projection CSVs in `projection_systems/hitters/2025/`, rescales each system to the actual league wOBA environment, and reports weighted RMSE tables.

## Project Map

```text
baseball_projection/
|-- data/
|   |-- cache/                  # Player ID cache files
|   `-- raw/fangraphs/           # Local actual batting data
|-- model_building_and_testing/
|   |-- evaluation/
|   |   `-- reviewing_systems.py  # Projection-system comparison workflow
|   `-- model_development/
|       `-- hitter_model.py       # Main hitter projection model
`-- projection_systems/
    |-- hitters/
    |   |-- 2025/                 # Public systems plus model output
    |   `-- 2026/                 # Future-season projection inputs/outputs
    `-- pitchers/                 # Pitcher projection files
```

## Useful Tweaks

To project a different completed season, change `PROJECT_SEASON`, make sure the actuals/projection-system files exist for that season, and rerun the model.

To project a future season, set `PROJECT_SEASON` to the next season after the latest actuals file. The script will use Depth Charts projected PA when available and skip completed-season evaluation.

To evaluate a different model output file without editing code:

```powershell
$env:XGB_FILE = "projection_systems/hitters/2025/toms_hitter_projections_2025.csv"
python model_building_and_testing/evaluation/reviewing_systems.py
```

## Notes And Limits

Overall, I was very pleased to see that my model compared favorably to other models. Rookie handling is still the hardest part of the hitter model. I do not use minor league data yet and that will be a big addition to help my rookie and young player projections accuracy. I have also noticed that my projections tend to be a little lower on players than other projection systems. A review there could be beneficial. This is a work in progress but the initial results are promising. 
