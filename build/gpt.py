"""Build the cleaned GPT exposure inputs used by the worker and geography figures.

Running this file writes five cleaned files to data_clean:
1. gpt_worker_occ_scores.csv stores the GPT exposure score assigned to each Current Population Survey worker occupation.
2. gpt_worker_occupation_exposure.csv stores modern worker counts by occupation and GPT exposure score.
3. gpt_worker_coverage.csv stores how much of the modern worker sample matched to a GPT-scored occupation.
4. gpt_cz_industry_scores.csv stores commuting-zone industry employment with the GPT exposure score assigned to each industry.
5. gpt_cz_exposure.csv stores the final commuting-zone GPT exposure score used by the geography figures.

The script creates those outputs by mapping GPTs-are-GPTs occupation scores to worker occupations, aggregating modern workers by occupation, translating occupation exposure into industry exposure, and averaging industry exposure within commuting zones.
"""

from __future__ import annotations

import io
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Set up inputs used to build GPT exposure files
# ---------------------------------------------------------------------------


# This section defines the files, sample rules, exposure thresholds, and functions for checks, weighted averages, age filtering, and industry-sector matching.
# The worker sample uses employed Current Population Survey workers ages 25 through 64 in 2021 and 2022.
# The exposure thresholds mark occupations where the GPT exposure score is at least 50, 60, 70, 80, or 90 percent.
# The government ownership codes below let public administration industries match the government rows in the industry staffing files.

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data_raw"
DATA_CLEAN = ROOT / "data_clean"

MODERN_CPS_PATH = DATA_RAW / "cps_2019_to_2025.csv.gz"
MODERN_AI_CPS_YEARS = (2021, 2022)
CPS_EMPSTAT_WORKING = (10, 12)
CPS_WORKER_WEIGHT = "WTFINL"
CPS_CHUNK_SIZE = 250_000
WORKER_AGE_MIN = 25
WORKER_AGE_MAX = 64
GPT_EXPOSURE_THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90)

PUBLIC_ADMIN_GOVERNMENT_NAICS = {
    "1": "999100",
    "2": "999200",
    "3": "999300",
}


# Check that a raw or cleaned table has the columns this script expects before using it.
def require_columns(df: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


# Average exposure scores using weights, so larger occupations, industries, or places count more.
# Drop missing values and nonpositive weights because they cannot contribute to a weighted average.
def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    values_arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    weights_arr = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(values_arr) & np.isfinite(weights_arr) & (weights_arr > 0)
    if not ok.any():
        return float("nan")
    return float(np.average(values_arr[ok], weights=weights_arr[ok]))


# Apply the worker age restriction used throughout the modern CPS worker sample.
# Return True for workers ages 25 through 64 and False otherwise.
def cps_age_mask(age: pd.Series) -> pd.Series:
    return (age >= WORKER_AGE_MIN) & (age <= WORKER_AGE_MAX)


# Convert a four-digit NAICS industry code to the sector key used in the industry staffing files.
# NAICS means North American Industry Classification System.
# Manufacturing, retail, transportation, and public administration use combined sector labels in those staffing files.
def oews_sector_key_from_naics4(naics4: str) -> str:
    sector = str(naics4).zfill(4)[:2]
    if sector in {"31", "32", "33"}:
        return "31-33"
    if sector in {"44", "45"}:
        return "44-45"
    if sector in {"48", "49"}:
        return "48-49"
    if sector == "92":
        return "99"
    return sector


# ---------------------------------------------------------------------------
# 2. Build GPT exposure scores for worker occupations
# ---------------------------------------------------------------------------


# This section creates the occupation-level GPT exposure table used by the worker figures.
# The GPTs-are-GPTs source file reports exposure for detailed Standard Occupational Classification occupations.
# The Current Population Survey records worker occupations with Census occupation codes instead.
# The Census occupation crosswalk links each worker occupation to one or more detailed occupation codes.
# When one worker occupation maps to several detailed occupations, this section averages GPT exposure using National Employment Matrix employment weights.
# The cleaned output is one row per worker occupation with the assigned GPT exposure score.


# Load the GPTs-are-GPTs occupation exposure score from the source file.
# The source variable dv_rating_beta is the GPT exposure score used throughout this package.
@lru_cache(maxsize=1)
def load_gpts_occ_scores() -> pd.DataFrame:
    occ = pd.read_csv(DATA_RAW / "gpts_gpts/data/occ_level.csv")
    require_columns(occ, ["O*NET-SOC Code", "dv_rating_beta"], "occ_level.csv")
    occ["soc_code"] = occ["O*NET-SOC Code"].astype(str).str.slice(stop=7)
    occ["gpt_exposure"] = pd.to_numeric(occ["dv_rating_beta"], errors="coerce")
    return occ.dropna(subset=["soc_code", "gpt_exposure"]).groupby("soc_code", as_index=False)["gpt_exposure"].mean()


# Load detailed occupation employment from the National Employment Matrix.
# These employment counts weight the average when one worker occupation maps to multiple detailed occupations.
@lru_cache(maxsize=1)
def load_nem_line_items() -> pd.DataFrame:
    emp = pd.read_excel(DATA_RAW / "nat_emp_matrix.xlsx", sheet_name="Table 1.2").iloc[1:].copy()
    emp.columns = [
        "title", "soc_code", "type", "emp2024", "emp2034", "dist2024", "dist2034",
        "chg_num", "chg_pct", "selfemp", "openings", "median_wage", "education",
        "experience", "training", "ooh",
    ]
    emp = emp[emp["type"] == "Line item"].copy()
    emp["emp2024"] = pd.to_numeric(emp["emp2024"], errors="coerce")
    return emp[["soc_code", "emp2024"]]


# Load the crosswalk that connects modern CPS occupation codes to detailed occupation codes.
# Keep one row per CPS occupation code with the occupation title and detailed occupation-code pattern.
def load_2018_census_occ_soc_crosswalk() -> pd.DataFrame:
    raw = pd.read_excel(
        DATA_RAW / "2018-census-occupation-classification-titles-and-code-list.xlsx",
        sheet_name="2018",
        header=5,
        dtype=str,
    )
    raw.columns = [str(col).strip() for col in raw.columns]
    raw = raw.rename(
        columns={
            "Occupation title": "occ_title",
            "2018 Census Title": "occ_title",
            "2018 Census code": "occ_code",
            "2018 Census Code": "occ_code",
            "2018 SOC code": "soc_pattern",
            "2018 SOC Code": "soc_pattern",
        }
    )
    require_columns(raw, ["occ_title", "occ_code", "soc_pattern"], "2018 occupation crosswalk")
    out = raw[["occ_title", "occ_code", "soc_pattern"]].dropna(subset=["occ_code", "soc_pattern"]).copy()
    out["occ_code"] = out["occ_code"].astype(str).str.strip().str.replace("–", "-", regex=False)
    out["soc_pattern"] = out["soc_pattern"].astype(str).str.strip().str.replace("–", "-", regex=False)
    out = out[out["occ_code"].str.fullmatch(r"\d{4}")].copy()
    out["unit_id"] = out["occ_code"].astype(int)
    return out[["unit_id", "occ_title", "soc_pattern"]].drop_duplicates(subset=["unit_id"])


# Turn a detailed occupation-code pattern into the prefix used to find all matching detailed occupations.
# This handles patterns where a worker occupation maps to a family of detailed occupations rather than one exact code.
def soc_prefix_from_pattern(pattern: str) -> str:
    major, detail = str(pattern).split("-", 1)
    trimmed = detail.rstrip("0X")
    return f"{major}-{trimmed}" if trimmed else f"{major}-"


# Assign GPT exposure to each CPS worker occupation.
# Use an exact detailed occupation match when available, otherwise use all detailed occupations that share the crosswalk prefix.
def build_gpt_worker_occ_scores() -> pd.DataFrame:
    crosswalk = load_2018_census_occ_soc_crosswalk()
    scored = load_gpts_occ_scores().merge(load_nem_line_items(), on="soc_code", how="left")
    rows = []
    for row in crosswalk.itertuples(index=False):
        matches = scored.loc[scored["soc_code"] == row.soc_pattern].copy()
        if matches.empty:
            matches = scored[scored["soc_code"].str.startswith(soc_prefix_from_pattern(str(row.soc_pattern)))].copy()
        if matches.empty:
            score = np.nan
        elif len(matches) == 1:
            score = float(matches["gpt_exposure"].iloc[0])
        else:
            weighted = matches["emp2024"].notna() & (matches["emp2024"] > 0)
            score = (
                weighted_mean(matches.loc[weighted, "gpt_exposure"], matches.loc[weighted, "emp2024"])
                if weighted.any()
                else float(matches["gpt_exposure"].mean())
            )
        rows.append(
            {
                "unit_id": int(row.unit_id),
                "occupation": str(row.occ_title),
                "soc_pattern": str(row.soc_pattern),
                "gpt_exposure": score,
            }
        )
    return pd.DataFrame(rows).sort_values("unit_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Count modern workers by GPT-scored occupation
# ---------------------------------------------------------------------------


# This section applies the worker-occupation GPT exposure table to the modern CPS worker sample.
# The worker sample keeps employed workers ages 25 through 64 in 2021 and 2022.
# Workers are grouped by occupation so the output shows worker weight, occupation title, detailed occupation-code pattern, and GPT exposure score.
# The output also includes threshold flags for the GPT exposure cutoffs used in the worker figures.
# A separate coverage file records the share of worker weight that matched to a GPT-scored occupation.


# Build the modern worker exposure table and the worker-coverage summary.
# The coverage summary checks how much worker weight remains after requiring a matched GPT exposure score.
def build_gpt_worker_occupation_exposure(gpt_map: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    lookup = gpt_map.set_index("unit_id")
    rows = []
    total_worker_weight = 0.0
    matched_worker_weight = 0.0
    usecols = ["YEAR", "AGE", "EMPSTAT", "OCC", CPS_WORKER_WEIGHT]
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
        total_worker_weight += float(chunk[CPS_WORKER_WEIGHT].sum())
        chunk["gpt_exposure"] = chunk["OCC"].map(lookup["gpt_exposure"])
        chunk = chunk[chunk["gpt_exposure"].notna()].copy()
        matched_worker_weight += float(chunk[CPS_WORKER_WEIGHT].sum())
        rows.append(
            chunk.groupby("OCC", as_index=False)[CPS_WORKER_WEIGHT]
            .sum()
            .rename(columns={"OCC": "unit_id", CPS_WORKER_WEIGHT: "worker_weight"})
        )

    out = pd.concat(rows, ignore_index=True).groupby("unit_id", as_index=False)["worker_weight"].sum()
    out["occupation"] = out["unit_id"].map(lookup["occupation"])
    out["soc_pattern"] = out["unit_id"].map(lookup["soc_pattern"])
    out["gpt_exposure"] = out["unit_id"].map(lookup["gpt_exposure"])
    for threshold in GPT_EXPOSURE_THRESHOLDS:
        out[f"selected_ge_{int(threshold * 100):02d}"] = out["gpt_exposure"] >= threshold
    coverage = pd.DataFrame(
        [{"total_worker_weight": total_worker_weight, "matched_worker_weight": matched_worker_weight, "matched_worker_share": matched_worker_weight / total_worker_weight}]
    )
    return out, coverage


# ---------------------------------------------------------------------------
# 4. Translate occupation GPT exposure into industry GPT exposure
# ---------------------------------------------------------------------------


# This section builds GPT exposure scores for industries.
# The geography figures need place-level exposure, but GPT exposure starts as an occupation score.
# The Occupational Employment and Wage Statistics staffing files show which occupations work in each industry and how many workers they represent.
# This section averages occupation GPT exposure within each industry using OEWS employment weights.
# It builds scores at several industry levels because some local industry rows are too detailed to match directly.
# The later geography build uses the most detailed available score for each industry row.


# Load one national Occupational Employment and Wage Statistics occupation-by-industry file from the zipped source data.
# Keep detailed occupations for the national area and the employment count needed for weighting.
@lru_cache(maxsize=8)
def load_oews_2022_detail(member: str) -> pd.DataFrame:
    with zipfile.ZipFile(DATA_RAW / "oesm22in4.zip") as zf:
        raw = zf.read(f"oesm22in4/{member}")
    oews = pd.read_excel(io.BytesIO(raw), dtype={"NAICS": str, "OCC_CODE": str})
    oews.columns = [str(col).upper() for col in oews.columns]
    require_columns(oews, ["AREA", "O_GROUP", "NAICS", "OCC_CODE", "TOT_EMP"], member)
    oews["TOT_EMP"] = pd.to_numeric(oews["TOT_EMP"], errors="coerce")
    oews["NAICS"] = oews["NAICS"].astype(str).str.strip()
    oews["OCC_CODE"] = oews["OCC_CODE"].astype(str).str.strip().str.slice(stop=7)
    oews = oews[(oews["AREA"].astype(str) == "99") & (oews["O_GROUP"] == "detailed")].copy()
    return oews.dropna(subset=["NAICS", "TOT_EMP", "OCC_CODE"]).copy()


# Average occupation GPT exposure into one industry coding level.
# The same logic is used for four-digit industries, three-digit industries, broad sectors, and public administration rows.
def build_oews_exposure_scores(member: str, key_col: str, score_col: str) -> pd.DataFrame:
    detail = load_oews_2022_detail(member).copy()
    if key_col == "naics4":
        detail[key_col] = detail["NAICS"].str.slice(stop=4)
    elif key_col == "naics3":
        detail[key_col] = detail["NAICS"].str.slice(stop=3)
    else:
        detail[key_col] = detail["NAICS"]
    scored = detail.merge(load_gpts_occ_scores(), left_on="OCC_CODE", right_on="soc_code", how="inner")
    return pd.DataFrame(
        [{key_col: key, score_col: weighted_mean(sub["gpt_exposure"], sub["TOT_EMP"])} for key, sub in scored.groupby(key_col)]
    )


# Build four-digit industry exposure scores.
# These are the preferred scores when a local industry row matches at this detail level.
@lru_cache(maxsize=1)
def gpt_oews_4d_scores() -> pd.DataFrame:
    return build_oews_exposure_scores("nat4d_M2022_dl.xlsx", "naics4", "score_4d")


# Build three-digit industry exposure scores.
# These are used when a local industry does not have a four-digit match.
@lru_cache(maxsize=1)
def gpt_oews_3d_scores() -> pd.DataFrame:
    scores = build_oews_exposure_scores("nat3d_M2022_dl.xlsx", "naics3", "score_3d")
    return scores[scores["naics3"] != "999"].copy()


# Build broad sector exposure scores.
# These are used only as a fallback outside public administration and unmatched sector 99 rows.
@lru_cache(maxsize=1)
def gpt_oews_sector_scores() -> pd.DataFrame:
    return build_oews_exposure_scores("natsector_M2022_dl.xlsx", "sector_key", "score_sector")


# Build separate public administration exposure scores by government owner.
# Public administration needs this special match because the local employment file separates federal, state, and local government ownership.
@lru_cache(maxsize=1)
def gpt_oews_public_admin_scores() -> pd.DataFrame:
    scores = build_oews_exposure_scores("nat4d_M2022_dl.xlsx", "gov_key", "score_public_admin")
    return scores[scores["gov_key"].isin(PUBLIC_ADMIN_GOVERNMENT_NAICS.values())].copy()


# Attach the best available industry GPT exposure score to each local industry row.
# Try four-digit industry first, then public administration owner, then three-digit industry, then broad sector.
def add_gpt_industry_scores(industry: pd.DataFrame) -> pd.DataFrame:
    out = industry.copy()
    out["naics4"] = out["naics4"].astype(str).str.zfill(4).str.slice(stop=4)
    out["naics3"] = out["naics3"].astype(str).str.zfill(3).str.slice(stop=3)
    out["sector2"] = out["sector2"].astype(str).str.zfill(2).str.slice(stop=2)
    out["sector_key"] = out["sector_key"].astype(str)
    out["own_code"] = out["own_code"].astype(str)

    out = out.merge(gpt_oews_4d_scores(), on="naics4", how="left")
    out["public_admin_key"] = np.where(
        out["naics4"].str.startswith("92") & out["own_code"].isin(PUBLIC_ADMIN_GOVERNMENT_NAICS),
        out["own_code"].map(PUBLIC_ADMIN_GOVERNMENT_NAICS),
        pd.NA,
    )
    out = out.merge(
        gpt_oews_public_admin_scores().rename(columns={"gov_key": "public_admin_key"}),
        on="public_admin_key",
        how="left",
    )
    out = out.merge(gpt_oews_3d_scores(), on="naics3", how="left")
    out = out.merge(gpt_oews_sector_scores(), on="sector_key", how="left")

    out["industry_score"] = out["score_4d"].fillna(out["score_public_admin"]).fillna(out["score_3d"])
    sector_mask = out["industry_score"].isna() & ~out["sector2"].isin(["92", "99"])
    out.loc[sector_mask, "industry_score"] = out.loc[sector_mask, "score_sector"]
    out["score_match_level"] = np.select(
        [
            out["score_4d"].notna(),
            out["score_public_admin"].notna(),
            out["score_3d"].notna(),
            sector_mask & out["score_sector"].notna(),
        ],
        ["4-digit", "public-admin-owner", "3-digit", "sector"],
        default="unmatched",
    )
    return out.drop(columns=["score_4d", "score_public_admin", "score_3d", "score_sector", "public_admin_key"])


# ---------------------------------------------------------------------------
# 5. Average industry GPT exposure within commuting zones
# ---------------------------------------------------------------------------


# This section builds the place-level GPT exposure files used by the geography figures.
# The county-to-commuting-zone crosswalk assigns each county to a local labor market.
# The QCEW file gives employment by county, ownership, and four-digit NAICS industry.
# The script first aggregates county industries into commuting-zone industries and assigns each row an industry GPT score.
# It then averages those industry scores within each commuting zone using local industry employment as weights.


# Load the crosswalk that assigns counties to 1990 commuting zones.
# Keep the county FIPS code and commuting-zone id needed to aggregate county employment.
def load_cz_crosswalk() -> pd.DataFrame:
    crosswalk = pd.read_excel(DATA_RAW / "cz_2000_crosswalk.xls")
    crosswalk = crosswalk.rename(columns={"FIPS": "fips", "Commuting Zone ID, 1990": "czone", "County population 2000": "pop2000"})
    require_columns(crosswalk, ["fips", "czone"], "cz_2000_crosswalk.xls")
    out = crosswalk[["fips", "czone"]].dropna().copy()
    out["fips"] = out["fips"].astype(int)
    out["czone"] = out["czone"].astype(int)
    return out.drop_duplicates(subset=["fips"])


# Load county employment by ownership and four-digit industry from the 2022 QCEW file.
# Keep annual county rows and remove statewide or territory summary rows that do not map cleanly to counties.
def load_county_naics_2022() -> pd.DataFrame:
    usecols = ["area_fips", "own_code", "industry_code", "agglvl_code", "size_code", "qtr", "annual_avg_emplvl"]
    pieces = []
    for chunk in pd.read_csv(
        DATA_RAW / "2022_annual_singlefile.zip",
        compression="zip",
        usecols=usecols,
        dtype=str,
        chunksize=500_000,
    ):
        chunk = chunk[(chunk["qtr"] == "A") & (chunk["size_code"] == "0") & (chunk["agglvl_code"] == "76")].copy()
        chunk = chunk[chunk["area_fips"].str[-3:] != "999"].copy()
        chunk = chunk[chunk["area_fips"].str[:2] != "78"].copy()
        chunk["fips"] = pd.to_numeric(chunk["area_fips"], errors="coerce")
        chunk["own_code"] = chunk["own_code"].astype(str)
        chunk["naics4"] = chunk["industry_code"].astype(str).str.slice(stop=4).str.zfill(4)
        chunk["naics3"] = chunk["naics4"].str.slice(stop=3)
        chunk["sector2"] = chunk["naics4"].str.slice(stop=2)
        chunk["sector_key"] = chunk["naics4"].map(oews_sector_key_from_naics4)
        chunk["emplvl"] = pd.to_numeric(chunk["annual_avg_emplvl"], errors="coerce")
        pieces.append(chunk[["fips", "own_code", "naics4", "naics3", "sector2", "sector_key", "emplvl"]])
    county = pd.concat(pieces, ignore_index=True).dropna(subset=["fips", "own_code", "naics4", "emplvl"]).copy()
    county["fips"] = county["fips"].astype(int)
    return county.groupby(["fips", "own_code", "naics4", "naics3", "sector2", "sector_key"], as_index=False)["emplvl"].sum()


# Aggregate county industry employment to commuting-zone industry rows and attach industry GPT scores.
# Return both the scored industry rows and total commuting-zone employment for the final place-level average.
def build_gpt_cz_industry_scores() -> tuple[pd.DataFrame, pd.DataFrame]:
    crosswalk = load_cz_crosswalk()
    county_naics = load_county_naics_2022()
    cz_industry = (
        county_naics.merge(crosswalk, on="fips", how="inner")
        .groupby(["czone", "own_code", "naics4", "naics3", "sector2", "sector_key"], as_index=False)
        .agg({"emplvl": "sum"})
        .rename(columns={"emplvl": "cz_industry_emp"})
    )
    county_total = (
        county_naics.groupby("fips", as_index=False)["emplvl"]
        .sum()
        .rename(columns={"emplvl": "county_total_emp"})
        .merge(crosswalk, on="fips", how="inner")
    )
    cz_total = county_total.groupby("czone", as_index=False)["county_total_emp"].sum().rename(columns={"county_total_emp": "cz_total_emp"})
    return add_gpt_industry_scores(cz_industry), cz_total


# Average scored industry rows within each commuting zone using local industry employment weights.
# This produces the final place-level GPT exposure measure used by the geography figures.
def build_gpt_cz_exposure(scored: pd.DataFrame, cz_total: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for czone, sub in scored.dropna(subset=["industry_score"]).groupby("czone"):
        rows.append({"czone": czone, "gpt_exposure": weighted_mean(sub["industry_score"], sub["cz_industry_emp"])})
    return pd.DataFrame(rows).merge(cz_total, on="czone", how="left").sort_values("czone")


def main() -> None:
    DATA_CLEAN.mkdir(parents=True, exist_ok=True)

    print("Building GPT worker occupation scores...")
    gpt_map = build_gpt_worker_occ_scores()
    gpt_map.to_csv(DATA_CLEAN / "gpt_worker_occ_scores.csv", index=False)

    print("Aggregating modern CPS workers by GPT-scored occupation...")
    worker_exposure, coverage = build_gpt_worker_occupation_exposure(gpt_map)
    worker_exposure.to_csv(DATA_CLEAN / "gpt_worker_occupation_exposure.csv", index=False)
    coverage.to_csv(DATA_CLEAN / "gpt_worker_coverage.csv", index=False)

    print("Projecting GPT exposure into CZ industries and CZs...")
    scored, cz_total = build_gpt_cz_industry_scores()
    scored.to_csv(DATA_CLEAN / "gpt_cz_industry_scores.csv", index=False)
    build_gpt_cz_exposure(scored, cz_total).to_csv(DATA_CLEAN / "gpt_cz_exposure.csv", index=False)


if __name__ == "__main__":
    main()
