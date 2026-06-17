# China Shock-AI REPO Package

This repo reproduces the cleaned data, cutoff outputs, and figures used in the China Shock vs. AI exposure comparison.

The raw files needed for this subset are copied into `data_raw/`. The build scripts write inspectable CSVs to `data_clean/`. The analysis scripts read those cleaned CSVs and write figures to `output/`.

Each Python file is meant to be readable on its own. `run_first.py` only creates the local Python environment and installs the required packages. `master.py` is the pipeline runner. The build and analysis scripts remain independently runnable.

## Run

Run commands from this repo root.

Use Python 3.10 or newer.

First, set up the local Python environment:

```bash
python run_first.py
```

If your computer uses `python3` instead of `python`, run:

```bash
python3 run_first.py
```

This creates `.venv/` and installs `requirements.txt`.

Then activate the environment:

```bash
source .venv/bin/activate
```

On Windows, use:

```bash
.venv\Scripts\activate
```

After activation, run the full pipeline:

```bash
python master.py
```

Run part of the pipeline:

```bash
python master.py build
python master.py analysis
python master.py worker
python master.py cutoff
python master.py geo
python master.py worker_edu
python master.py geo_map
```

See every available target:

```bash
python master.py --list
```

Common target names:

- `all`: rebuilds all cleaned data and recreates all figures.
- `build`: rebuilds all cleaned data.
- `analysis` or `figures`: recreates all figures from existing cleaned data.
- `worker`: rebuilds the worker inputs and recreates the worker figures.
- `cutoff`: recreates the China worker cutoff outputs from existing cleaned data.
- `geo`: rebuilds the geography inputs and recreates the geography figures.
- `worker_figures`: recreates only the worker figures from existing cleaned data.
- `cutoff_figures`: recreates only the China worker cutoff outputs.
- `geo_figures`: recreates only the geography figures from existing cleaned data.

Specific figure targets:

- `worker_edu`
- `worker_wage_tercile`
- `cutoff`
- `cutoff_curve`
- `cutoff_drop`
- `cutoff_additions`
- `geo_map`
- `geo_exposure_rank_bins`
- `geo_edu_bins`

By default, `master.py` uses the Python executable that is running it. To force a different executable:

```bash
python master.py --python python3 all
```

The individual scripts can also be run directly.

Build the cleaned data:

```bash
python build/china.py
python build/gpt.py
python build/workers.py
python build/geography.py
```

Create the figures:

```bash
python analysis/worker_edu.py
python analysis/worker_wage_tercile.py
python analysis/cutoff.py
python analysis/geo_map.py
python analysis/geo_exposure_rank_bins.py
python analysis/geo_edu_bins.py
```

If you run the individual scripts directly, use a Python environment where the packages in `requirements.txt` are installed.

## Outputs

Worker figures:

- `output/worker/01_edu.png`
- `output/worker/02_wage_tercile.png`

Cutoff outputs:

- `output/cutoff/china_worker_cutoff_justification_curve.png`
- `output/cutoff/china_worker_cutoff_justification_drop.png`
- `output/cutoff/china_worker_cutoff_industry_additions.txt`

Geography figures:

- `output/geo/01_map.png`
- `output/geo/02_exposure_rank_bins.png`
- `output/geo/03_ba_bins_pp.png`
- `output/geo/04_hs_bins_pp.png`

## Build Scripts

1. `build/china.py`

   Builds the China exposure inputs.

   Cleaned files:
   - `data_clean/china_import_penetration_ind1990.csv`: China import-penetration exposure assigned to each CPS worker industry.
   - `data_clean/china_worker_industry_exposure.csv`: historical CPS worker weight by assigned China exposure industry, including the exact selected high-exposure worker share.
   - `data_clean/china_cz_import_penetration.csv`: commuting-zone China import penetration for the geography figures.

2. `build/gpt.py`

   Builds the GPT exposure inputs.

   Cleaned files:
   - `data_clean/gpt_worker_occ_scores.csv`: GPTs-are-GPTs occupation exposure mapped to CPS worker occupations.
   - `data_clean/gpt_worker_occupation_exposure.csv`: modern CPS worker weight by GPT-scored occupation.
   - `data_clean/gpt_worker_coverage.csv`: share of modern CPS worker weight matched to a GPT-scored occupation.
   - `data_clean/gpt_cz_industry_scores.csv`: commuting-zone industry employment rows with assigned GPT industry exposure scores.
   - `data_clean/gpt_cz_exposure.csv`: commuting-zone GPT exposure for the geography figures.

3. `build/workers.py`

   Builds the cleaned worker comparison files from the China and GPT exposure inputs.

   Cleaned files:
   - `data_clean/worker_group_summary.csv`: selected China and GPT worker groups and their worker shares.
   - `data_clean/worker_education_profile.csv`: education shares for each selected worker group.
   - `data_clean/worker_weekly_earnings_selected.csv`: selected worker weight by weekly earnings value and wage percentile.
   - `data_clean/worker_wage_tercile_summary.csv`: selected worker shares in the bottom, middle, and top thirds of the wage distribution.

4. `build/geography.py`

   Builds the cleaned geography comparison files from the China and GPT exposure inputs.

   Cleaned files:
   - `data_clean/place_exposure_cz.csv`: one row per commuting zone with China exposure, GPT exposure, population denominators, education shares, z-scores, and percentile ranks.
   - `data_clean/cz_map_paths.csv`: projected commuting-zone polygon paths used by the map figure.
   - `data_clean/geo_exposure_rank_bins.csv`: exposure and population summaries for commuting zones grouped by exposure rank.
   - `data_clean/geo_education_bins_pp.csv`: BA+ and HS-or-less percentage-point differences for commuting zones grouped by exposure rank.

## Analysis Scripts

1. `analysis/worker_edu.py`

   Creates `output/worker/01_edu.png` from:
   - `data_clean/worker_group_summary.csv`
   - `data_clean/worker_education_profile.csv`

2. `analysis/worker_wage_tercile.py`

   Creates `output/worker/02_wage_tercile.png` from:
   - `data_clean/worker_group_summary.csv`
   - `data_clean/worker_wage_tercile_summary.csv`

3. `analysis/cutoff.py`

   Creates:
   - `output/cutoff/china_worker_cutoff_justification_curve.png`
   - `output/cutoff/china_worker_cutoff_justification_drop.png`
   - `output/cutoff/china_worker_cutoff_industry_additions.txt`

   These outputs use:
   - `data_clean/china_worker_industry_exposure.csv`

4. `analysis/geo_map.py`

   Creates `output/geo/01_map.png` from:
   - `data_clean/place_exposure_cz.csv`
   - `data_clean/cz_map_paths.csv`

5. `analysis/geo_exposure_rank_bins.py`

   Creates `output/geo/02_exposure_rank_bins.png` from:
   - `data_clean/geo_exposure_rank_bins.csv`

6. `analysis/geo_edu_bins.py`

   Creates:
   - `output/geo/03_ba_bins_pp.png`
   - `output/geo/04_hs_bins_pp.png`

   Both figures use:
   - `data_clean/geo_education_bins_pp.csv`

## Notes

- The China worker cutoff is exactly `0.04671415737844796`; cutoff outputs display it as `4.67%`.
- Worker China exposure is 1991-2007 China import growth divided by 1991 domestic absorption.
- Worker China exposure is mapped from detailed manufacturing industries to CPS worker industries with 1991 NBER-CES employment weights.
- Geography China exposure uses `d_tradeusch_p1_1991_2007` from the commuting-zone China exposure file.
- GPT exposure comes from the GPTs-are-GPTs occupation beta measure.
- Worker GPT exposure uses CPS occupation mapping.
- Geography GPT exposure projects occupation exposure through industry staffing patterns and local industry employment.
