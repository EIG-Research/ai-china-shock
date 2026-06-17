"""Build the cleaned worker comparison inputs used by the worker figures.

Running this file writes four cleaned files to data_clean:
1. worker_group_summary.csv stores the selected China and GPT worker groups and their worker shares.
2. worker_education_profile.csv stores the education distribution for each selected worker group.
3. worker_weekly_earnings_selected.csv stores selected workers by weekly earnings value and wage percentile.
4. worker_wage_tercile_summary.csv stores the share of each selected worker group in the bottom, middle, and top thirds of the wage distribution.

The script creates those outputs by reading the cleaned China and GPT exposure files, defining the worker groups used in the figures, grouping selected workers by education, and grouping selected workers by wage percentile.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Set up inputs used to build worker comparison files
# ---------------------------------------------------------------------------


# This section defines the files, worker samples, weights, wage variable, education groups, and wage terciles used below.
# The China worker comparisons use employed Current Population Survey workers ages 25 through 64 in 1991, 1992, and 1993.
# The GPT worker comparisons use employed Current Population Survey workers ages 25 through 64 in 2021 and 2022.
# Historical weekly earnings are converted to 1993 dollars before workers are ranked in the historical wage distribution.
# Modern weekly earnings use EARNWEEK2, the CPS weekly earnings variable used throughout this package.

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data_raw"
DATA_CLEAN = ROOT / "data_clean"

HISTORICAL_CPS_PATH = DATA_RAW / "cps_91_92_93.csv.gz"
MODERN_CPS_PATH = DATA_RAW / "cps_2019_to_2025.csv.gz"
HISTORICAL_CPS_YEARS = (1991, 1992, 1993)
MODERN_AI_CPS_YEARS = (2021, 2022)
WORKER_YEAR_LABEL = "2021-2022"

CPS_EMPSTAT_WORKING = (10, 12)
CPS_WORKER_WEIGHT = "WTFINL"
CPS_EARNINGS_WEIGHT = "EARNWT"
CPS_WEEKLY_EARNINGS = "EARNWEEK2"
CPS_CHUNK_SIZE = 250_000
WORKER_AGE_MIN = 25
WORKER_AGE_MAX = 64

CHINA_WORKER_SHARE = 0.04671415737844796
CHINA_WORKER_DISPLAY_SHARE = "5%"
GPT_EXPOSURE_THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90)
CPI_U_ANNUAL_AVG = {1991: 136.2, 1992: 140.3, 1993: 144.5}
BROAD_EDUCATION_ORDER = ("HS or Less", "Some College / AA", "Bachelor's", "Graduate")
WAGE_TERCILES = (
    (0.0, 100.0 / 3.0, "Bottom Third"),
    (100.0 / 3.0, 200.0 / 3.0, "Middle Third"),
    (200.0 / 3.0, 100.0, "Top Third"),
)


# Read a cleaned input file and stop with a clear message if an earlier build step has not run.
def read_required_clean(name: str) -> pd.DataFrame:
    path = DATA_CLEAN / name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run build/china.py and build/gpt.py first.")
    return pd.read_csv(path)


# Apply the worker age restriction used in both the historical and modern CPS worker samples.
# Return True for workers ages 25 through 64 and False otherwise.
def cps_age_mask(age: pd.Series) -> pd.Series:
    return (age >= WORKER_AGE_MIN) & (age <= WORKER_AGE_MAX)


# Convert historical weekly earnings to 1993 dollars using annual CPI-U values.
# The wage chart compares workers to the wage distribution from their own period after this adjustment.
def deflate_to_1993(year: pd.Series, weekly_earnings: pd.Series) -> pd.Series:
    return (weekly_earnings.astype(float) * (CPI_U_ANNUAL_AVG[1993] / year.map(CPI_U_ANNUAL_AVG))).round(0)


# Collapse detailed CPS education codes into the four education groups used in the education figure.
def education_bucket(educ: pd.Series) -> pd.Series:
    out = pd.Series(index=educ.index, dtype=object)
    out.loc[educ.between(0, 73)] = "HS or Less"
    out.loc[educ.isin([80, 81, 90, 91, 92, 100])] = "Some College / AA"
    out.loc[educ.isin([110, 111])] = "Bachelor's"
    out.loc[educ.isin([120, 121, 122, 123, 124, 125])] = "Graduate"
    return out


# ---------------------------------------------------------------------------
# 2. Define the selected worker groups used in the figures
# ---------------------------------------------------------------------------


# This section defines the worker groups compared in the education and wage figures.
# The China group is the exact worker-weighted high-exposure tail built in build/china.py.
# The GPT groups are workers in occupations with GPT exposure at or above 50, 60, 70, 80, or 90 percent.
# The group summary records each group's label and share of the relevant worker sample.
# The selected-unit tables tell later steps which industries or occupations belong to each group.


# Read the cleaned China and GPT worker exposure files and create the selected worker-group definitions.
# Return both the group summary table and the selected industry or occupation units used to build profiles.
def build_worker_group_summary() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    china_workers = read_required_clean("china_worker_industry_exposure.csv")
    gpt_workers = read_required_clean("gpt_worker_occupation_exposure.csv")
    gpt_coverage = read_required_clean("gpt_worker_coverage.csv")
    matched_share = float(gpt_coverage["matched_worker_share"].iloc[0])
    gpt_total_worker_weight = float(gpt_workers["worker_weight"].sum()) / matched_share

    groups = [
        {
            "group_id": "china_top",
            "sort_order": 0,
            "kind": "china",
            "threshold": np.nan,
            "selected_worker_share": CHINA_WORKER_SHARE,
            "worker_year_label": WORKER_YEAR_LABEL,
            "display_label": f"China Shock Top {CHINA_WORKER_DISPLAY_SHARE}",
        }
    ]
    selected = {
        "china_top": china_workers.loc[china_workers["tail_weight"] > 0, ["unit_id", "tail_weight"]].copy()
    }
    for order, threshold in enumerate(GPT_EXPOSURE_THRESHOLDS, start=1):
        group_id = f"gpt_ge_{int(round(threshold * 100)):02d}"
        selected_units = gpt_workers.loc[gpt_workers["gpt_exposure"] >= threshold, ["unit_id"]].copy()
        selected_units["tail_weight"] = 1.0
        selected[group_id] = selected_units
        selected_share = float(gpt_workers.loc[gpt_workers["gpt_exposure"] >= threshold, "worker_weight"].sum()) / gpt_total_worker_weight
        groups.append(
            {
                "group_id": group_id,
                "sort_order": order,
                "kind": "gpt",
                "threshold": threshold,
                "selected_worker_share": selected_share,
                "worker_year_label": WORKER_YEAR_LABEL,
                "display_label": f"AI >= {int(threshold * 100)}% tasks ({100 * selected_share:.1f}%)",
            }
        )
    return pd.DataFrame(groups), selected


# ---------------------------------------------------------------------------
# 3. Build education distributions for selected worker groups
# ---------------------------------------------------------------------------


# This section creates the education data used by the worker education figure.
# It first counts worker weight by industry and education for the historical China sample.
# It then counts worker weight by occupation and education for the modern GPT sample.
# Finally, it applies the selected worker-group definitions and converts education counts into within-group shares.
# The cleaned output is one row per worker group and education bucket.


# Count historical CPS workers by assigned China exposure industry and education bucket.
# Workers outside mapped China-exposed manufacturing are kept as a zero-exposure group.
def build_china_education_aggregates(china_map: pd.DataFrame) -> pd.DataFrame:
    lookup = china_map.set_index("unit_id")["china_import_penetration"]
    rows = []
    usecols = ["YEAR", "AGE", "EMPSTAT", "IND1990", CPS_WORKER_WEIGHT, "EDUC"]
    for chunk in pd.read_csv(HISTORICAL_CPS_PATH, usecols=usecols, chunksize=CPS_CHUNK_SIZE):
        chunk = chunk[
            chunk["YEAR"].isin(HISTORICAL_CPS_YEARS)
            & cps_age_mask(chunk["AGE"])
            & chunk["EMPSTAT"].isin(CPS_EMPSTAT_WORKING)
            & (chunk["IND1990"] > 0)
            & (chunk[CPS_WORKER_WEIGHT] > 0)
        ].copy()
        if chunk.empty:
            continue
        exposure = chunk["IND1990"].map(lookup).fillna(0.0)
        chunk["unit_id"] = np.where(exposure > 0, chunk["IND1990"], 0)
        chunk["bucket"] = education_bucket(chunk["EDUC"])
        chunk = chunk[chunk["bucket"].notna()].copy()
        rows.append(chunk.groupby(["unit_id", "bucket"], as_index=False)[CPS_WORKER_WEIGHT].sum().rename(columns={CPS_WORKER_WEIGHT: "worker_weight"}))
    return pd.concat(rows, ignore_index=True).groupby(["unit_id", "bucket"], as_index=False)["worker_weight"].sum()


# Count modern CPS workers by GPT-scored occupation and education bucket.
# Workers without a matched GPT occupation score are excluded from the GPT comparison groups.
def build_gpt_education_aggregates(gpt_map: pd.DataFrame) -> pd.DataFrame:
    lookup = gpt_map.set_index("unit_id")["gpt_exposure"]
    rows = []
    usecols = ["YEAR", "AGE", "EMPSTAT", "OCC", CPS_WORKER_WEIGHT, "EDUC"]
    for chunk in pd.read_csv(MODERN_CPS_PATH, usecols=usecols, chunksize=CPS_CHUNK_SIZE):
        chunk = chunk[
            chunk["YEAR"].isin(MODERN_AI_CPS_YEARS)
            & cps_age_mask(chunk["AGE"])
            & chunk["EMPSTAT"].isin(CPS_EMPSTAT_WORKING)
            & (chunk["OCC"] > 0)
            & (chunk[CPS_WORKER_WEIGHT] > 0)
        ].copy()
        if chunk.empty:
            continue
        chunk["gpt_exposure"] = chunk["OCC"].map(lookup)
        chunk = chunk[chunk["gpt_exposure"].notna()].copy()
        chunk["bucket"] = education_bucket(chunk["EDUC"])
        chunk = chunk[chunk["bucket"].notna()].copy()
        rows.append(chunk.groupby(["OCC", "bucket"], as_index=False)[CPS_WORKER_WEIGHT].sum().rename(columns={"OCC": "unit_id", CPS_WORKER_WEIGHT: "worker_weight"}))
    return pd.concat(rows, ignore_index=True).groupby(["unit_id", "bucket"], as_index=False)["worker_weight"].sum()


# Apply one selected worker-group definition to education counts.
# Return the selected group's education shares in a fixed display order.
def selected_education_profile(aggregates: pd.DataFrame, selected_units: pd.DataFrame, group_id: str) -> pd.DataFrame:
    merged = aggregates.merge(selected_units[["unit_id", "tail_weight"]], on="unit_id", how="left")
    merged["tail_weight"] = merged["tail_weight"].fillna(0.0)
    merged["selected_worker_weight"] = merged["worker_weight"] * merged["tail_weight"]
    grouped = merged.groupby("bucket", as_index=False)["selected_worker_weight"].sum()
    total = float(grouped["selected_worker_weight"].sum())
    shares = {row.bucket: row.selected_worker_weight / total for row in grouped.itertuples(index=False)}
    return pd.DataFrame(
        [{"group_id": group_id, "bucket": bucket, "share": float(shares.get(bucket, 0.0))} for bucket in BROAD_EDUCATION_ORDER]
    )


# ---------------------------------------------------------------------------
# 4. Build wage-percentile and wage-tercile data for selected worker groups
# ---------------------------------------------------------------------------


# This section creates the wage data used by the worker wage-tercile figure.
# It first builds a wage-percentile lookup for each worker period.
# The China wage percentiles come from the historical CPS sample after weekly earnings are converted to 1993 dollars.
# The GPT wage percentiles come from the modern CPS sample using weekly earnings in the modern period.
# It then records selected worker weight by wage value and percentile, and summarizes those selected workers into bottom, middle, and top thirds.


# Create the weekly earnings value used for ranking workers in a period's wage distribution.
# Historical earnings are deflated to 1993 dollars before ranking.
def weekly_earnings_rank_value(chunk: pd.DataFrame, period: str) -> pd.Series:
    weekly = chunk[CPS_WEEKLY_EARNINGS].astype(float)
    if period == "historical":
        weekly = deflate_to_1993(chunk["YEAR"], weekly)
    return weekly.round(4)


# Build the lookup that maps each weekly earnings value to its worker-weighted wage percentile.
# The lookup is built separately for the historical China period and the modern GPT period.
def build_weekly_earnings_percentile_map(period: str) -> pd.DataFrame:
    cps_path = HISTORICAL_CPS_PATH if period == "historical" else MODERN_CPS_PATH
    years = HISTORICAL_CPS_YEARS if period == "historical" else MODERN_AI_CPS_YEARS
    usecols = ["YEAR", "AGE", "EMPSTAT", CPS_EARNINGS_WEIGHT, CPS_WEEKLY_EARNINGS]
    pieces = []
    for chunk in pd.read_csv(cps_path, usecols=usecols, chunksize=CPS_CHUNK_SIZE):
        chunk = chunk[
            chunk["YEAR"].isin(years)
            & cps_age_mask(chunk["AGE"])
            & chunk["EMPSTAT"].isin(CPS_EMPSTAT_WORKING)
            & (chunk[CPS_EARNINGS_WEIGHT] > 0)
            & (chunk[CPS_WEEKLY_EARNINGS] > 0)
            & (chunk[CPS_WEEKLY_EARNINGS] < 999999)
        ].copy()
        if chunk.empty:
            continue
        chunk["weekly_earnings_rank_value"] = weekly_earnings_rank_value(chunk, period)
        pieces.append(chunk.groupby("weekly_earnings_rank_value", as_index=False)[CPS_EARNINGS_WEIGHT].sum())
    wage = pd.concat(pieces, ignore_index=True).groupby("weekly_earnings_rank_value", as_index=False)[CPS_EARNINGS_WEIGHT].sum()
    wage = wage.sort_values("weekly_earnings_rank_value").reset_index(drop=True)
    wage["cum_weight"] = wage[CPS_EARNINGS_WEIGHT].cumsum()
    total = float(wage[CPS_EARNINGS_WEIGHT].sum())
    wage["wage_percentile"] = 100.0 * (wage["cum_weight"] - 0.5 * wage[CPS_EARNINGS_WEIGHT]) / total
    return wage[["weekly_earnings_rank_value", "wage_percentile"]]


# Count historical CPS earnings weight by China exposure industry and weekly earnings value.
# Keep mapped China-exposed manufacturing workers because the selected China group is drawn from those industries.
def build_china_weekly_earnings_distribution(china_map: pd.DataFrame) -> pd.DataFrame:
    lookup = china_map.set_index("unit_id")["china_import_penetration"]
    rows = []
    usecols = ["YEAR", "AGE", "EMPSTAT", "IND1990", CPS_EARNINGS_WEIGHT, CPS_WEEKLY_EARNINGS]
    for chunk in pd.read_csv(HISTORICAL_CPS_PATH, usecols=usecols, chunksize=CPS_CHUNK_SIZE):
        chunk = chunk[
            chunk["YEAR"].isin(HISTORICAL_CPS_YEARS)
            & cps_age_mask(chunk["AGE"])
            & chunk["EMPSTAT"].isin(CPS_EMPSTAT_WORKING)
            & (chunk["IND1990"] > 0)
            & (chunk[CPS_EARNINGS_WEIGHT] > 0)
            & (chunk[CPS_WEEKLY_EARNINGS] > 0)
            & (chunk[CPS_WEEKLY_EARNINGS] < 999999)
        ].copy()
        if chunk.empty:
            continue
        chunk["china_import_penetration"] = chunk["IND1990"].map(lookup)
        chunk = chunk[chunk["china_import_penetration"].notna() & (chunk["china_import_penetration"] > 0)].copy()
        if chunk.empty:
            continue
        chunk["weekly_earnings_rank_value"] = weekly_earnings_rank_value(chunk, "historical")
        rows.append(chunk.groupby(["IND1990", "weekly_earnings_rank_value"], as_index=False)[CPS_EARNINGS_WEIGHT].sum().rename(columns={"IND1990": "unit_id", CPS_EARNINGS_WEIGHT: "worker_weight"}))
    return pd.concat(rows, ignore_index=True).groupby(["unit_id", "weekly_earnings_rank_value"], as_index=False)["worker_weight"].sum()


# Count modern CPS earnings weight by GPT-scored occupation and weekly earnings value.
# Workers without a matched GPT occupation score are excluded from the GPT comparison groups.
def build_gpt_weekly_earnings_distribution(gpt_map: pd.DataFrame) -> pd.DataFrame:
    lookup = gpt_map.set_index("unit_id")["gpt_exposure"]
    rows = []
    usecols = ["YEAR", "AGE", "EMPSTAT", "OCC", CPS_EARNINGS_WEIGHT, CPS_WEEKLY_EARNINGS]
    for chunk in pd.read_csv(MODERN_CPS_PATH, usecols=usecols, chunksize=CPS_CHUNK_SIZE):
        chunk = chunk[
            chunk["YEAR"].isin(MODERN_AI_CPS_YEARS)
            & cps_age_mask(chunk["AGE"])
            & chunk["EMPSTAT"].isin(CPS_EMPSTAT_WORKING)
            & (chunk["OCC"] > 0)
            & (chunk[CPS_EARNINGS_WEIGHT] > 0)
            & (chunk[CPS_WEEKLY_EARNINGS] > 0)
            & (chunk[CPS_WEEKLY_EARNINGS] < 999999)
        ].copy()
        if chunk.empty:
            continue
        chunk["gpt_exposure"] = chunk["OCC"].map(lookup)
        chunk = chunk[chunk["gpt_exposure"].notna()].copy()
        if chunk.empty:
            continue
        chunk["weekly_earnings_rank_value"] = weekly_earnings_rank_value(chunk, "modern")
        rows.append(chunk.groupby(["OCC", "weekly_earnings_rank_value"], as_index=False)[CPS_EARNINGS_WEIGHT].sum().rename(columns={"OCC": "unit_id", CPS_EARNINGS_WEIGHT: "worker_weight"}))
    return pd.concat(rows, ignore_index=True).groupby(["unit_id", "weekly_earnings_rank_value"], as_index=False)["worker_weight"].sum()


# Apply one selected worker-group definition to an earnings distribution.
# Attach wage percentiles so the selected workers can be summarized by wage tercile.
def selected_earnings_distribution(distribution: pd.DataFrame, wage_map: pd.DataFrame, selected_units: pd.DataFrame, group_id: str) -> pd.DataFrame:
    merged = distribution.merge(selected_units[["unit_id", "tail_weight"]], on="unit_id", how="left")
    merged["tail_weight"] = merged["tail_weight"].fillna(0.0)
    merged["selected_worker_weight"] = merged["worker_weight"] * merged["tail_weight"]
    merged = merged[merged["selected_worker_weight"] > 0].copy()
    merged = merged.merge(wage_map, on="weekly_earnings_rank_value", how="left")
    merged["group_id"] = group_id
    return merged[["group_id", "weekly_earnings_rank_value", "wage_percentile", "selected_worker_weight"]]


# Convert selected worker wage-percentile records into shares in the bottom, middle, and top thirds.
# Each group's shares sum to one across the three wage terciles.
def build_wage_tercile_summary(selected: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_id in groups.sort_values("sort_order")["group_id"]:
        frame = selected[selected["group_id"] == group_id].copy()
        total = float(frame["selected_worker_weight"].sum())
        for lo, hi, bucket in WAGE_TERCILES:
            mask = (frame["wage_percentile"] >= lo) & (frame["wage_percentile"] < hi)
            if hi >= 100:
                mask = (frame["wage_percentile"] >= lo) & (frame["wage_percentile"] <= hi)
            rows.append({"group_id": group_id, "bucket": bucket, "share": float(frame.loc[mask, "selected_worker_weight"].sum()) / total})
    return pd.DataFrame(rows)


def main() -> None:
    DATA_CLEAN.mkdir(parents=True, exist_ok=True)
    china_map = read_required_clean("china_import_penetration_ind1990.csv")
    gpt_map = read_required_clean("gpt_worker_occ_scores.csv")

    print("Defining worker groups...")
    groups, selected = build_worker_group_summary()
    groups.to_csv(DATA_CLEAN / "worker_group_summary.csv", index=False)

    print("Building worker education profiles...")
    china_edu = build_china_education_aggregates(china_map)
    gpt_edu = build_gpt_education_aggregates(gpt_map)
    edu_frames = [selected_education_profile(china_edu, selected["china_top"], "china_top")]
    for threshold in GPT_EXPOSURE_THRESHOLDS:
        group_id = f"gpt_ge_{int(round(threshold * 100)):02d}"
        edu_frames.append(selected_education_profile(gpt_edu, selected[group_id], group_id))
    pd.concat(edu_frames, ignore_index=True).to_csv(DATA_CLEAN / "worker_education_profile.csv", index=False)

    print("Building worker weekly-earnings records...")
    china_wage_map = build_weekly_earnings_percentile_map("historical")
    gpt_wage_map = build_weekly_earnings_percentile_map("modern")
    china_dist = build_china_weekly_earnings_distribution(china_map)
    gpt_dist = build_gpt_weekly_earnings_distribution(gpt_map)
    selected_frames = [selected_earnings_distribution(china_dist, china_wage_map, selected["china_top"], "china_top")]
    for threshold in GPT_EXPOSURE_THRESHOLDS:
        group_id = f"gpt_ge_{int(round(threshold * 100)):02d}"
        selected_frames.append(selected_earnings_distribution(gpt_dist, gpt_wage_map, selected[group_id], group_id))
    selected_earnings = pd.concat(selected_frames, ignore_index=True)
    selected_earnings.to_csv(DATA_CLEAN / "worker_weekly_earnings_selected.csv", index=False)
    build_wage_tercile_summary(selected_earnings, groups).to_csv(DATA_CLEAN / "worker_wage_tercile_summary.csv", index=False)


if __name__ == "__main__":
    main()
