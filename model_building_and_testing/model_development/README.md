# Model Development

Start with `hitter_model.py`.

That file contains the current hitter projection workflow in one readable, top-to-bottom script. It keeps the settings, feature lists, lag building, Marcel-style baseline features, XGBoost training, wOBA calculation, export, and optional evaluation in the same place.

Use the settings near the top of the file to change the projection season, training window, input files, output path, and whether to run evaluation after building projections.

The evaluation workflow for hitter projections is in `../evaluation/reviewing_systems.py`.
