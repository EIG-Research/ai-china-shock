"""Build the cleaned China exposure inputs used by the worker and geography figures.

Running this file writes three cleaned files to data_clean:
1. china_import_penetration_ind1990.csv stores the China exposure score assigned to each Current Population Survey worker industry.
2. china_worker_industry_exposure.csv stores historical worker counts by assigned China exposure and marks the exact high-exposure China worker group.
3. china_cz_import_penetration.csv stores the commuting-zone China exposure score used by the geography figures.

The script creates those outputs by calculating detailed manufacturing exposure from raw China trade data, translating those detailed industries to worker industries, and applying the result to historical workers.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Set up inputs used to assign China exposure
# ---------------------------------------------------------------------------


# Setting paths
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data_raw"
DATA_CLEAN = ROOT / "data_clean"
HISTORICAL_CPS_PATH = DATA_RAW / "cps_91_92_93.csv.gz"

# Setting parameters
HISTORICAL_CPS_YEARS = (1991, 1992, 1993)
CPS_EMPSTAT_WORKING = (10, 12)
CPS_WORKER_WEIGHT = "WTFINL"
CPS_CHUNK_SIZE = 250_000
WORKER_AGE_MIN = 25
WORKER_AGE_MAX = 64

# Top shaare of workers to select from the sample
CHINA_WORKER_SHARE = 0.04671415737844796

# Helps assign China exposure to CPS workers
# China trade data are measured for detailed manufacturing industriesSIC/SIC87dd codes)
# CPS workers are recorded in broader worker industry categories (IND1990 codes)
# Each row below links one CPS worker industry t the detailed manufacturing industries that should be used for its China score
# When several detailed industries feed into one CPS industry, the script uses 1991 manufacturing employment to compute a weighted average exposure score.
CHINA_IND1990_SIC_RULES = {
    100: {"label": "Meat products", "include": ["201"]},
    101: {"label": "Dairy products", "include": ["202"]},
    102: {"label": "Canned / frozen fruits & vegetables", "include": ["203"]},
    110: {"label": "Grain mill products", "include": ["204"]},
    111: {"label": "Bakery products", "include": ["205"]},
    112: {"label": "Sugar & confectionery", "include": ["206"]},
    120: {"label": "Beverage industries", "include": ["208"]},
    121: {"label": "Misc. food preparations", "include": ["207", "209"]},
    122: {"label": "Not specified food industries", "include": ["201-209"]},
    130: {"label": "Tobacco manufactures", "include": ["21"]},
    132: {"label": "Knitting mills", "include": ["225"]},
    140: {"label": "Dyeing & finishing textiles", "include": ["226"]},
    141: {"label": "Carpets and rugs", "include": ["227"]},
    142: {"label": "Yarn, thread, and fabric mills", "include": ["221-224", "228"]},
    150: {"label": "Misc. textile mill products", "include": ["229"]},
    151: {"label": "Apparel & accessories", "include": ["231-238"]},
    152: {"label": "Misc. fabricated textile products", "include": ["239"]},
    160: {"label": "Pulp, paper, and paperboard mills", "include": ["261-263"]},
    161: {"label": "Misc. paper & pulp products", "include": ["267"]},
    162: {"label": "Paperboard containers and boxes", "include": ["265"]},
    171: {"label": "Newspaper publishing and printing", "include": ["271"]},
    172: {"label": "Printing / publishing except newspapers", "include": ["272-279"]},
    180: {"label": "Plastics, synthetics, and resins", "include": ["282"]},
    181: {"label": "Drugs", "include": ["283"]},
    182: {"label": "Soaps and cosmetics", "include": ["284"]},
    190: {"label": "Paints and varnishes", "include": ["285"]},
    191: {"label": "Agricultural chemicals", "include": ["287"]},
    192: {"label": "Industrial & misc. chemicals", "include": ["281", "286", "289"]},
    200: {"label": "Petroleum refining", "include": ["291"]},
    201: {"label": "Misc. petroleum & coal products", "include": ["295", "299"]},
    210: {"label": "Tires and inner tubes", "include": ["301"]},
    211: {"label": "Other rubber products", "include": ["302-306"]},
    212: {"label": "Misc. plastics products", "include": ["308"]},
    220: {"label": "Leather tanning and finishing", "include": ["311"]},
    221: {"label": "Footwear except rubber/plastic", "include": ["313", "314"]},
    222: {"label": "Leather products except footwear", "include": ["315-317", "319"]},
    230: {"label": "Logging", "include": ["241"]},
    231: {"label": "Sawmills / millwork", "include": ["242", "243"]},
    232: {"label": "Wood buildings and mobile homes", "include": ["245"]},
    241: {"label": "Misc. wood products", "include": ["244", "249"]},
    242: {"label": "Furniture and fixtures", "include": ["25"]},
    250: {"label": "Glass and glass products", "include": ["321-323"]},
    251: {"label": "Cement / concrete / gypsum / plaster", "include": ["324", "327"]},
    252: {"label": "Structural clay products", "include": ["325"]},
    261: {"label": "Pottery and related products", "include": ["326"]},
    262: {"label": "Misc. nonmetallic mineral products", "include": ["328", "329"]},
    270: {"label": "Steelworks and rolling mills", "include": ["331"]},
    271: {"label": "Iron and steel foundries", "include": ["332"]},
    272: {"label": "Primary aluminum industries", "include": ["3334", "3353-3355", "3363", "3365"]},
    280: {"label": "Other primary metal industries", "include": ["3331", "3339", "3341", "3351", "3356", "3357", "3364", "3366", "3369", "339"]},
    281: {"label": "Cutlery / handtools / hardware", "include": ["342"]},
    282: {"label": "Fabricated structural metal products", "include": ["344"]},
    290: {"label": "Screw machine products", "include": ["345"]},
    291: {"label": "Metal forgings and stampings", "include": ["346"]},
    292: {"label": "Ordnance", "include": ["348"]},
    300: {"label": "Misc. fabricated metal products", "include": ["341", "343", "347", "349"]},
    301: {"label": "Not specified metal industries", "include": ["331-349"]},
    310: {"label": "Engines and turbines", "include": ["351"]},
    311: {"label": "Farm machinery and equipment", "include": ["352"]},
    312: {"label": "Construction / material handling machines", "include": ["353"]},
    320: {"label": "Metalworking machinery", "include": ["354"]},
    321: {"label": "Office & accounting machines", "include": ["3578", "3579"]},
    322: {"label": "Computers and related equipment", "include": ["3571-3577"]},
    331: {"label": "Machinery n.e.c.", "include": ["355", "356", "358", "359"]},
    332: {"label": "Not specified machinery", "include": ["351-359"]},
    340: {"label": "Household appliances", "include": ["363"]},
    341: {"label": "Radio / TV / communications equipment", "include": ["365", "366"]},
    342: {"label": "Electrical machinery n.e.c.", "include": ["361", "362", "364", "367", "369"]},
    350: {"label": "Not specified electrical machinery", "include": ["361-369"]},
    351: {"label": "Motor vehicles and equipment", "include": ["371"]},
    352: {"label": "Aircraft and parts", "include": ["372"]},
    360: {"label": "Ship and boat building", "include": ["373"]},
    361: {"label": "Railroad locomotives and equipment", "include": ["374"]},
    362: {"label": "Guided missiles / space vehicles", "include": ["376"]},
    370: {"label": "Cycles & misc. transportation equipment", "include": ["375", "379"]},
    371: {"label": "Scientific & controlling instruments", "include": ["381", "382"], "exclude": ["3827"]},
    372: {"label": "Medical / dental / optical instruments", "include": ["3827", "384", "385"]},
    380: {"label": "Photographic equipment and supplies", "include": ["386"]},
    381: {"label": "Watches and clocks", "include": ["387"]},
    390: {"label": "Toys / amusement / sporting goods", "include": ["394"]},
    391: {"label": "Misc. manufacturing industries", "include": ["39"], "exclude": ["394"]},
    392: {"label": "Not specified manufacturing", "include": ["20-39"]},
}


# The helpers below prepare the industry mapping and worker cutoff used later in this script.
# The mapping table above uses compact manufacturing industry-code patterns such as 21, 201, and 331-349.
# The functions below turn those patterns into detailed industry matches, average China exposure with employment weights, and select the exact high-exposure worker share.
# The same helpers also make the script fail clearly if an input file is missing a variable needed for these steps.


# Check that a raw or cleaned table has the columns this script expects before using it.
def require_columns(df: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


# Average exposure scores using weights, so larger industries or worker groups count more.
# Drop missing values and nonpositive weights because they cannot contribute to a weighted average.
def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    values_arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    weights_arr = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(values_arr) & np.isfinite(weights_arr) & (weights_arr > 0)
    if not ok.any():
        return float("nan")
    return float(np.average(values_arr[ok], weights=weights_arr[ok]))


# Apply the worker age restriction used throughout the historical CPS worker sample.
# Return True for workers ages 25 through 64 and False otherwise.
def cps_age_mask(age: pd.Series) -> pd.Series:
    return (age >= WORKER_AGE_MIN) & (age <= WORKER_AGE_MAX)


# Convert one compact manufacturing industry-code pattern into the full numeric range it covers.
# The mapping table uses two-digit, three-digit, four-digit, and range patterns.
# For example, 21 means 2100-2199, 201 means 2010-2019, 3334 means 3334 only, and 331-349 means 3310-3499.
def sic_token_to_range(token: str) -> tuple[int, int]:
    token = token.strip().replace(" ", "")
    if "-" in token:
        start_text, end_text = token.split("-", 1)
        start = int(start_text)
        end = int(end_text)
        if len(start_text) == 2:
            return start * 100, end * 100 + 99
        if len(start_text) == 3:
            return start * 10, end * 10 + 9
        if len(start_text) == 4:
            return start, end
    value = int(token)
    if len(token) == 2:
        return value * 100, value * 100 + 99
    if len(token) == 3:
        return value * 10, value * 10 + 9
    if len(token) == 4:
        return value, value
    raise ValueError(f"Unsupported SIC token: {token}")


# Use the code ranges above to flag detailed manufacturing industries that belong to one CPS worker-industry rule.
# This is how the include and exclude lists in the mapping table are applied to the detailed industry data.
def build_sic_mask(sic_codes: pd.Series, tokens: list[str]) -> pd.Series:
    mask = pd.Series(False, index=sic_codes.index)
    for token in tokens:
        lo, hi = sic_token_to_range(token)
        mask |= sic_codes.between(lo, hi)
    return mask


# Read the file that converts detailed manufacturing employment codes to the industry codes used in the China trade data.
# The source file is written as Stata commands, so this extracts the old-code to new-code pairs used for recoding.
def load_sic87dd_mapping(path: Path) -> dict[int, int]:
    text = path.read_text()
    return {
        int(source): int(target)
        for target, source in re.findall(
            r"replace\s+sic87dd\s*=\s*(\d+)\s+if\s+sic87dd\s*==\s*(\d+)",
            text,
        )
    }


# Rank worker industries by China exposure and mark the top worker-weighted share used for the China comparison group.
# If the cutoff falls inside one industry, give that boundary industry only the fraction needed to hit the target share exactly.
def top_tail_weights(df: pd.DataFrame, id_col: str, weight_col: str, exposure_col: str, worker_share: float) -> pd.DataFrame:
    ranked = df[[id_col, weight_col, exposure_col]].sort_values(
        [exposure_col, weight_col, id_col],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    target_weight = float(ranked[weight_col].sum()) * worker_share
    ranked["cum_before"] = ranked[weight_col].cumsum() - ranked[weight_col]
    ranked["cum_after"] = ranked[weight_col].cumsum()
    ranked["tail_weight"] = 0.0
    ranked.loc[ranked["cum_after"] <= target_weight, "tail_weight"] = 1.0
    boundary = (ranked["cum_before"] < target_weight) & (ranked["cum_after"] > target_weight)
    ranked.loc[boundary, "tail_weight"] = (
        (target_weight - ranked.loc[boundary, "cum_before"]) / ranked.loc[boundary, weight_col]
    ).clip(0.0, 1.0)
    return ranked[[id_col, "tail_weight"]]


# ---------------------------------------------------------------------------
# 2. Build China exposure scores for detailed manufacturing industries
# ---------------------------------------------------------------------------


# This section starts from the industry-level China trade file.
# The goal is to measure how much each detailed manufacturing industry was exposed to rising imports from China between 1991 and 2007.
# The source variable l_import_usch_1991 is U.S. imports from China in 1991.
# The source variable l_import_usch_2007 is U.S. imports from China in 2007.
# The source variable market1991 is the industry's 1991 U.S. market size.
# Market size means U.S. shipments plus imports minus exports.
# The exposure score is the 1991-2007 increase in imports from China divided by 1991 market size.
# The cleaned output keeps the industry code, 1991 market size, import increase, and exposure score.
def build_sic87dd_import_penetration() -> pd.DataFrame:
    trade = pd.read_stata(DATA_RAW / "d2_sic87dd_exposure_9114/sic87dd_exposure_9114.dta")
    required = ["sic87dd", "market1991", "l_import_usch_1991", "l_import_usch_2007"]
    require_columns(trade, required, "sic87dd_exposure_9114.dta")
    for column in required:
        trade[column] = pd.to_numeric(trade[column], errors="coerce")
    trade = trade.dropna(subset=required).copy()
    trade = trade[trade["market1991"] > 0].copy()
    trade["sic87dd"] = trade["sic87dd"].astype(int)
    trade["delta_china_imports"] = trade["l_import_usch_2007"] - trade["l_import_usch_1991"]
    trade["china_import_penetration"] = trade["delta_china_imports"] / trade["market1991"]
    return trade[["sic87dd", "market1991", "delta_china_imports", "china_import_penetration"]]


# ---------------------------------------------------------------------------
# 3. Convert detailed China exposure scores to CPS worker industries
# ---------------------------------------------------------------------------


# This section creates the bridge between the China trade data and the worker data.
# The China trade data measure exposure by detailed manufacturing industry.
# The CPS worker file records each worker in broader worker industry categories.
# The mapping table above says which detailed manufacturing industries belong to each CPS worker industry.
# When one CPS worker industry contains several detailed industries, this section averages their exposure scores using 1991 manufacturing employment as weights.
# The cleaned output is one row per CPS worker industry with its assigned China exposure score and the 1991 employment used to build it.


# Load 1991 manufacturing employment for detailed industries.
# Recode the employment industry codes to match the industry codes used in the China trade file.
def load_1991_nber_ces_employment() -> pd.DataFrame:
    ces = pd.read_csv(DATA_RAW / "nberces5818v1_s1987.csv", usecols=["sic", "year", "emp"])
    ces["sic"] = pd.to_numeric(ces["sic"], errors="coerce")
    ces["year"] = pd.to_numeric(ces["year"], errors="coerce")
    ces["emp"] = pd.to_numeric(ces["emp"], errors="coerce")
    ces = ces.dropna(subset=["sic", "year", "emp"]).copy()
    ces["sic"] = ces["sic"].astype(int)
    ces["year"] = ces["year"].astype(int)
    ces = ces[(ces["year"] == 1991) & (ces["emp"] > 0)].copy()
    ces["sic87dd"] = ces["sic"].replace(
        load_sic87dd_mapping(DATA_RAW / "shock_replication/Public-Release-Data/other/subfile_sic87dd.do")
    )
    return ces[ces["sic87dd"].between(2011, 3999)].copy()


# Combine detailed exposure scores with 1991 employment and the mapping table.
# Return the worker-industry exposure table that later gets assigned to CPS workers.
def build_china_import_penetration_ind1990() -> pd.DataFrame:
    ces = load_1991_nber_ces_employment()
    sic87dd_scores = build_sic87dd_import_penetration()
    base = ces.merge(sic87dd_scores[["sic87dd", "china_import_penetration"]], on="sic87dd", how="inner")

    rows = []
    for ind1990, rule in CHINA_IND1990_SIC_RULES.items():
        mask = build_sic_mask(base["sic"], rule["include"])
        if "exclude" in rule:
            mask &= ~build_sic_mask(base["sic"], rule["exclude"])
        sub = base.loc[mask].copy()
        if sub.empty:
            continue
        rows.append(
            {
                "unit_id": int(ind1990),
                "industry": str(rule["label"]),
                "china_import_penetration": weighted_mean(sub["china_import_penetration"], sub["emp"]),
                "source_emp_1991": float(sub["emp"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("unit_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Assign China exposure scores to historical workers
# ---------------------------------------------------------------------------


# This section applies the worker-industry exposure table to the 1991-1993 CPS worker sample.
# The worker sample keeps employed workers ages 25 through 64.
# Workers in mapped manufacturing industries receive the China exposure score for their industry.
# Workers outside the mapped manufacturing industries receive zero direct China exposure.
# The script then ranks industries by China exposure and marks the exact top worker-weighted share used in the figures.
# The cleaned output keeps total worker weight, the selected fraction, and selected worker weight for each worker-industry group.


# Build the historical worker exposure table from the CPS and the worker-industry China exposure table.
# Collapse individual workers to industry groups so the selected China worker group can be checked industry by industry.
def build_china_worker_industry_exposure(china_map: pd.DataFrame) -> pd.DataFrame:
    lookup = china_map.set_index("unit_id")
    rows = []
    usecols = ["YEAR", "AGE", "EMPSTAT", "IND1990", CPS_WORKER_WEIGHT]
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
        chunk["china_import_penetration"] = chunk["IND1990"].map(lookup["china_import_penetration"]).fillna(0.0)
        exposed = chunk["china_import_penetration"] > 0
        chunk["unit_id"] = np.where(exposed, chunk["IND1990"], 0)
        chunk["industry"] = np.where(
            exposed,
            chunk["IND1990"].map(lookup["industry"]).fillna("Mapped manufacturing industry"),
            "All other workers (zero direct China exposure)",
        )
        rows.append(
            chunk.groupby(["unit_id", "industry", "china_import_penetration"], as_index=False)[CPS_WORKER_WEIGHT]
            .sum()
            .rename(columns={CPS_WORKER_WEIGHT: "worker_weight"})
        )

    out = pd.concat(rows, ignore_index=True).groupby(
        ["unit_id", "industry", "china_import_penetration"],
        as_index=False,
    )["worker_weight"].sum()
    selected = top_tail_weights(out, "unit_id", "worker_weight", "china_import_penetration", CHINA_WORKER_SHARE)
    out = out.merge(selected, on="unit_id", how="left")
    out["tail_weight"] = out["tail_weight"].fillna(0.0)
    out["selected_worker_weight"] = out["worker_weight"] * out["tail_weight"]
    return out


# ---------------------------------------------------------------------------
# 5. Save China exposure scores for commuting zones
# ---------------------------------------------------------------------------


# This section saves the place-level China exposure measure used in the geography figures.
# The source file already contains the commuting-zone import-penetration measure used by the main analysis.
# The source variable d_tradeusch_p1_1991_2007 is China import penetration from 1991 to 2007 for each commuting zone.
# The cleaned output keeps the commuting-zone id and the China exposure score.


# Load the commuting-zone China exposure variable and keep the two columns needed by the geography build.
def build_china_cz_import_penetration() -> pd.DataFrame:
    china = pd.read_stata(
        DATA_RAW / "adh_persist_repo/data/czone_exposure_by_period_v5_gh.dta",
        columns=["czone", "d_tradeusch_p1_1991_2007"],
    )
    china["czone"] = china["czone"].astype(int)
    china["china_exposure"] = pd.to_numeric(china["d_tradeusch_p1_1991_2007"], errors="coerce")
    return china[["czone", "china_exposure"]].dropna(subset=["china_exposure"]).sort_values("czone")


def main() -> None:
    DATA_CLEAN.mkdir(parents=True, exist_ok=True)

    print("Building detailed and IND1990 China import-penetration exposure...")
    china_map = build_china_import_penetration_ind1990()
    china_map.to_csv(DATA_CLEAN / "china_import_penetration_ind1990.csv", index=False)

    print("Assigning China exposure to historical CPS workers...")
    build_china_worker_industry_exposure(china_map).to_csv(DATA_CLEAN / "china_worker_industry_exposure.csv", index=False)

    print("Saving commuting-zone China import penetration...")
    build_china_cz_import_penetration().to_csv(DATA_CLEAN / "china_cz_import_penetration.csv", index=False)


if __name__ == "__main__":
    main()
