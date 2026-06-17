"""Build the cleaned geography inputs used by the map and rank-bin figures.

Running this file writes four cleaned files to data_clean:
1. place_exposure_cz.csv stores China exposure, GPT exposure, population denominators, ranks, and education shares for each commuting zone.
2. cz_map_paths.csv stores projected commuting-zone polygon paths used by the map figure.
3. geo_exposure_rank_bins.csv stores exposure and population summaries for commuting zones grouped by exposure rank.
4. geo_education_bins_pp.csv stores education-share differences for commuting zones grouped by exposure rank.

The script creates those outputs by joining the cleaned China and GPT commuting-zone exposure files, adding population and education context, projecting map shapes into plot-ready coordinates, and summarizing places by exposure rank.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Set up inputs used to build geography files
# ---------------------------------------------------------------------------


# This section defines the raw and cleaned data folders and the shared functions used by the geography build.
# The shared functions read required input files, check required columns, compute weighted averages, and prepare county industry employment.
# The county industry employment is used to build 2022 job denominators for commuting zones.

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data_raw"
DATA_CLEAN = ROOT / "data_clean"


# Read a cleaned input file and stop with a clear message if an earlier build step has not run.
def read_required_clean(name: str) -> pd.DataFrame:
    path = DATA_CLEAN / name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run build/china.py and build/gpt.py first.")
    return pd.read_csv(path)


# Check that a raw or cleaned table has the columns this script expects before using it.
def require_columns(df: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


# Average exposure or demographic values using weights, so larger counties or places count more.
# Drop missing values and nonpositive weights because they cannot contribute to a weighted average.
def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    values_arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    weights_arr = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(values_arr) & np.isfinite(weights_arr) & (weights_arr > 0)
    if not ok.any():
        return float("nan")
    return float(np.average(values_arr[ok], weights=weights_arr[ok]))


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


# Load county employment by ownership and four-digit NAICS industry from the 2022 QCEW file.
# QCEW means Quarterly Census of Employment and Wages.
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


# ---------------------------------------------------------------------------
# 2. Build commuting-zone population, job, and education context
# ---------------------------------------------------------------------------


# This section builds the place context that gets attached to the exposure file.
# The commuting-zone crosswalk assigns each county to a local labor market.
# The denominators add 2000 population, 2024 working-age population, and 2022 employment for each commuting zone.
# The education files add 1990 education shares for the China comparison and 2024 education shares for the GPT comparison.


# Load the county-to-commuting-zone crosswalk.
# Keep county FIPS, commuting-zone id, and 2000 county population.
def load_cz_crosswalk() -> pd.DataFrame:
    crosswalk = pd.read_excel(DATA_RAW / "cz_2000_crosswalk.xls")
    crosswalk = crosswalk.rename(
        columns={
            "FIPS": "fips",
            "Commuting Zone ID, 1990": "czone",
            "County population 2000": "pop2000",
        }
    )
    require_columns(crosswalk, ["fips", "czone", "pop2000"], "cz_2000_crosswalk.xls")
    out = crosswalk[["fips", "czone", "pop2000"]].dropna(subset=["fips", "czone"]).copy()
    out["fips"] = out["fips"].astype(int)
    out["czone"] = out["czone"].astype(int)
    out["pop2000"] = pd.to_numeric(out["pop2000"], errors="coerce")
    return out


# Build commuting-zone population and employment denominators.
# Working-age population comes from 2024 ACS age counts, and current jobs come from 2022 QCEW county employment.
def load_cz_denominators() -> pd.DataFrame:
    crosswalk = load_cz_crosswalk().drop_duplicates(subset=["fips"]).copy()
    b01001 = pd.read_csv(DATA_RAW / "acs_b01001/ACSDT5Y2024.B01001-Data.csv", low_memory=False).iloc[1:].copy()
    b01001["fips"] = pd.to_numeric(b01001["GEO_ID"].str[-5:], errors="coerce")
    age_cols = [
        "B01001_011E", "B01001_012E", "B01001_013E", "B01001_014E", "B01001_015E",
        "B01001_016E", "B01001_017E", "B01001_018E", "B01001_019E", "B01001_035E",
        "B01001_036E", "B01001_037E", "B01001_038E", "B01001_039E", "B01001_040E",
        "B01001_041E", "B01001_042E", "B01001_043E",
    ]
    require_columns(b01001, ["GEO_ID", *age_cols], "ACS B01001 2024")
    for column in age_cols:
        b01001[column] = pd.to_numeric(b01001[column], errors="coerce")
    b01001["working_age_25_64"] = b01001[age_cols].sum(axis=1, min_count=1)

    county_jobs = load_county_naics_2022().groupby("fips", as_index=False)["emplvl"].sum().rename(columns={"emplvl": "current_jobs_2022"})
    merged = (
        crosswalk.merge(b01001[["fips", "working_age_25_64"]], on="fips", how="left")
        .merge(county_jobs, on="fips", how="left")
        .fillna({"working_age_25_64": 0.0, "current_jobs_2022": 0.0})
    )
    return (
        merged.groupby("czone", as_index=False)
        .agg({"pop2000": "sum", "working_age_25_64": "sum", "current_jobs_2022": "sum"})
        .rename(columns={"pop2000": "pop2000_cz", "working_age_25_64": "working_age_25_64_cz", "current_jobs_2022": "current_jobs_2022_cz"})
    )


# Build 1990 commuting-zone education shares from NHGIS county data.
# The China education figure compares China exposure ranks to 1990 education levels.
def load_nhgis_1990_education_by_cz() -> pd.DataFrame:
    county = pd.read_csv(
        DATA_RAW / "nhgis0001_csv/nhgis0001_ds123_1990_county.csv",
        usecols=["STATEA", "COUNTYA", "E33001", "E33002", "E33003", "E33004", "E33005", "E33006", "E33007"],
        dtype={"STATEA": str, "COUNTYA": str},
    )
    county["fips"] = (county["STATEA"].str.zfill(2) + county["COUNTYA"].str.zfill(3)).astype(int)
    educ_cols = [f"E3300{i}" for i in range(1, 8)]
    for column in educ_cols:
        county[column] = pd.to_numeric(county[column], errors="coerce")
    county["adult25p_1990"] = county[educ_cols].sum(axis=1, min_count=len(educ_cols))
    county["hs_less_1990"] = county[["E33001", "E33002", "E33003"]].sum(axis=1, min_count=3)
    county["ba_plus_1990"] = county["E33006"] + county["E33007"]
    county = county[county["adult25p_1990"].notna() & (county["adult25p_1990"] > 0)].copy()
    county = county.merge(load_cz_crosswalk().drop_duplicates(subset=["fips"])[["fips", "czone"]], on="fips", how="inner")
    out = county.groupby("czone", as_index=False)[["adult25p_1990", "hs_less_1990", "ba_plus_1990"]].sum()
    out["hs_less_share_1990"] = out["hs_less_1990"] / out["adult25p_1990"]
    out["ba_share_1990"] = out["ba_plus_1990"] / out["adult25p_1990"]
    return out


# Build 2024 commuting-zone education shares from ACS county data.
# The GPT education figure compares GPT exposure ranks to 2024 education levels.
def load_education_2024_by_cz() -> pd.DataFrame:
    county_universe = pd.read_excel(DATA_RAW / "aioe/AIOE_DataAppendix.xlsx", sheet_name="Appendix C")
    county_universe = county_universe.rename(columns={"FIPS Code": "fips"})
    county_universe = county_universe[county_universe["fips"].notna()].copy()
    county_universe["fips"] = county_universe["fips"].astype(int)
    county_universe = county_universe[county_universe["fips"] % 1000 != 0].copy()

    b15003 = pd.read_csv(DATA_RAW / "acs_B15003/ACSDT5Y2024.B15003-Data.csv", low_memory=False).iloc[1:].copy()
    b15003["fips"] = b15003["GEO_ID"].str[-5:].astype(int)
    education_cols = [f"B15003_{i:03d}E" for i in range(1, 26)]
    require_columns(b15003, ["GEO_ID", *education_cols], "ACS B15003 2024")
    for column in education_cols:
        b15003[column] = pd.to_numeric(b15003[column], errors="coerce")
    b15003["ba_plus"] = b15003[["B15003_022E", "B15003_023E", "B15003_024E", "B15003_025E"]].sum(axis=1)
    hs_less_cols = [f"B15003_{i:03d}E" for i in range(2, 19)]
    b15003["hs_less"] = b15003[hs_less_cols].sum(axis=1, min_count=len(hs_less_cols))
    b15003["adult25p"] = b15003["B15003_001E"]

    county = (
        county_universe[["fips"]]
        .merge(load_cz_crosswalk(), on="fips", how="inner")
        .merge(b15003[["fips", "adult25p", "hs_less", "ba_plus"]], on="fips", how="inner")
    )
    out = county.groupby("czone", as_index=False)[["adult25p", "hs_less", "ba_plus"]].sum()
    out = out.rename(columns={"adult25p": "adult25p_2024", "hs_less": "hs_less_2024", "ba_plus": "ba_plus_2024"})
    out["hs_less_share_2024"] = out["hs_less_2024"] / out["adult25p_2024"]
    out["ba_share_2024"] = out["ba_plus_2024"] / out["adult25p_2024"]
    return out


# ---------------------------------------------------------------------------
# 3. Build the combined commuting-zone exposure file
# ---------------------------------------------------------------------------


# This section creates the main place-level dataset used by the geography figures.
# It joins the cleaned China and GPT commuting-zone exposure files on the common set of commuting zones.
# It keeps only commuting zones with positive population denominators.
# It adds above-average exposure mass, standardized exposure, exposure percentiles, and education shares.
# The cleaned output is one row per commuting zone.


# Combine China exposure, GPT exposure, denominators, ranks, and education shares into one commuting-zone file.
def build_place_exposure_cz() -> pd.DataFrame:
    china = read_required_clean("china_cz_import_penetration.csv")
    gpt = read_required_clean("gpt_cz_exposure.csv")
    denoms = load_cz_denominators()[["czone", "working_age_25_64_cz", "pop2000_cz"]].copy()
    china = china.merge(denoms, on="czone", how="left")
    gpt = gpt.merge(denoms, on="czone", how="left")

    common = sorted(set(china["czone"]).intersection(gpt["czone"]))
    china = china[china["czone"].isin(common)].copy()
    gpt = gpt[gpt["czone"].isin(common)].copy()
    china = china[(china["working_age_25_64_cz"] > 0) & (china["pop2000_cz"] > 0)].copy()
    gpt = gpt[(gpt["working_age_25_64_cz"] > 0) & (gpt["pop2000_cz"] > 0)].copy()
    common = sorted(set(china["czone"]).intersection(gpt["czone"]))
    out = china[china["czone"].isin(common)].merge(gpt[gpt["czone"].isin(common)][["czone", "gpt_exposure"]], on="czone", how="inner")

    china_mean = weighted_mean(out["china_exposure"], out["working_age_25_64_cz"])
    gpt_mean = weighted_mean(out["gpt_exposure"], out["working_age_25_64_cz"])
    out["china_above_average_mass"] = np.maximum(out["china_exposure"] - china_mean, 0.0) * out["working_age_25_64_cz"]
    out["gpt_above_average_mass"] = np.maximum(out["gpt_exposure"] - gpt_mean, 0.0) * out["working_age_25_64_cz"]
    out["china_exposure_z"] = (out["china_exposure"] - float(out["china_exposure"].mean())) / float(out["china_exposure"].std(ddof=0))
    out["gpt_exposure_z"] = (out["gpt_exposure"] - float(out["gpt_exposure"].mean())) / float(out["gpt_exposure"].std(ddof=0))
    out["china_pctile"] = out["china_exposure"].rank(method="average", pct=True)
    out["gpt_pctile"] = out["gpt_exposure"].rank(method="average", pct=True)
    out = out.merge(load_nhgis_1990_education_by_cz(), on="czone", how="left")
    out = out.merge(load_education_2024_by_cz(), on="czone", how="left")
    return out.sort_values("czone").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Build plot-ready map paths for commuting zones
# ---------------------------------------------------------------------------


# This section converts the commuting-zone shapefile into a simple CSV that the map script can plot directly.
# The shapefile stores boundaries as longitude and latitude points.
# The output stores projected x and y coordinates for each point in each commuting-zone polygon.
# Alaska and Hawaii are projected separately and inset into the same coordinate space as the continental U.S.


# Project commuting-zone polygon points and save them as ordered map paths.
def build_projected_cz_map_paths() -> pd.DataFrame:
    import shapefile
    from pyproj import Transformer

    conus = Transformer.from_crs("EPSG:4326", "+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=23 +lon_0=-96 +datum=WGS84 +units=m +no_defs", always_xy=True)
    alaska = Transformer.from_crs("EPSG:4326", "+proj=aea +lat_1=55 +lat_2=65 +lat_0=50 +lon_0=-154 +datum=WGS84 +units=m +no_defs", always_xy=True)
    hawaii = Transformer.from_crs("EPSG:4326", "+proj=aea +lat_1=8 +lat_2=18 +lat_0=13 +lon_0=-157 +datum=WGS84 +units=m +no_defs", always_xy=True)

    # Project longitude and latitude arrays while preserving missing separators between polygon parts.
    def project(transformer: Transformer, lons: np.ndarray, lats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.full_like(lons, np.nan, dtype=float)
        y = np.full_like(lats, np.nan, dtype=float)
        mask = ~(np.isnan(lons) | np.isnan(lats))
        x[mask], y[mask] = transformer.transform(lons[mask], lats[mask])
        return x, y

    # Scale and move Alaska or Hawaii into an inset position for the final map coordinate system.
    def inset(x: np.ndarray, y: np.ndarray, scale: float, center: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
        x_center = float(np.nanmean([np.nanmin(x), np.nanmax(x)]))
        y_center = float(np.nanmean([np.nanmin(y), np.nanmax(y)]))
        return (x - x_center) * scale + center[0], (y - y_center) * scale + center[1]

    reader = shapefile.Reader(str(DATA_RAW / "cz1990_shapefile/cz1990.shp"))
    fields = [field[0] for field in reader.fields[1:]]
    rows = []
    for shape_rec in reader.iterShapeRecords():
        record = dict(zip(fields, shape_rec.record, strict=True))
        czone = int(record["cz"])
        points = np.asarray(shape_rec.shape.points, dtype=float)
        points[:, 0] = np.where(points[:, 0] > 0, points[:, 0] - 360.0, points[:, 0])
        lons = points[:, 0]
        lats = points[:, 1]
        lon_mid = float((np.nanmin(lons) + np.nanmax(lons)) / 2.0)
        lat_mid = float((np.nanmin(lats) + np.nanmax(lats)) / 2.0)
        if lat_mid < 25.5:
            x, y = project(hawaii, lons, lats)
            x, y = inset(x, y, 0.42, (-0.78e6, 0.34e6))
        elif lon_mid < -130 and lat_mid > 50:
            x, y = project(alaska, lons, lats)
            x, y = inset(x, y, 0.28, (-1.83e6, 0.53e6))
        else:
            x, y = project(conus, lons, lats)
        parts = list(shape_rec.shape.parts)
        for part_idx, start in enumerate(parts):
            stop = parts[part_idx + 1] if part_idx + 1 < len(parts) else len(x)
            for order, point_idx in enumerate(range(start, stop)):
                rows.append({"czone": czone, "part": part_idx, "order": order, "x": x[point_idx], "y": y[point_idx]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Summarize commuting zones by exposure rank
# ---------------------------------------------------------------------------


# This section builds the binned data used by the geography rank-bin figures.
# Commuting zones are ranked separately by China exposure and GPT exposure.
# The top 10, next 40, next 50, and all remaining commuting zones are summarized as rank bins.
# One output summarizes average exposure and population share by rank bin.
# The other output summarizes how each rank bin's education share differs from the average commuting zone.


# Define the exposure-rank groups used in the geography figures.
RANK_BIN_SPECS = (
    {"lo": 1, "hi": 10, "label": "Top 10", "tick_label": "Top 10"},
    {"lo": 11, "hi": 50, "label": "Next 40", "tick_label": "Next 40"},
    {"lo": 51, "hi": 100, "label": "Next 50", "tick_label": "Next 50"},
    {"lo": 101, "hi": None, "label": "All Others", "tick_label": "All Others"},
)


# Sort commuting zones by one exposure measure and assign each place to a rank bin.
def add_rank_bins(df: pd.DataFrame, rank_col: str) -> pd.DataFrame:
    ranked = df.sort_values([rank_col, "working_age_25_64_cz", "czone"], ascending=[False, False, True]).reset_index(drop=True)
    ranked["rank_position"] = np.arange(1, len(ranked) + 1)
    ranked["rank_bin"] = ""
    ranked["rank_bin_order"] = 0
    ranked["tick_label"] = ""
    for order, spec in enumerate(RANK_BIN_SPECS, start=1):
        mask = ranked["rank_position"] >= int(spec["lo"])
        if spec["hi"] is not None:
            mask &= ranked["rank_position"] <= int(spec["hi"])
        ranked.loc[mask, "rank_bin"] = str(spec["label"])
        ranked.loc[mask, "rank_bin_order"] = order
        ranked.loc[mask, "tick_label"] = str(spec["tick_label"])
    return ranked


# Summarize exposure and population share for each China and GPT exposure-rank bin.
# The exposure index expresses each bin's average exposure relative to the overall average.
def build_geo_exposure_rank_bins(place: pd.DataFrame) -> pd.DataFrame:
    total_pop = float(place["working_age_25_64_cz"].sum())
    rows = []
    for series, exposure_col in [("China Shock", "china_exposure"), ("AI Exposure", "gpt_exposure")]:
        ranked = add_rank_bins(place, exposure_col)
        baseline = float(place[exposure_col].mean())
        for spec in RANK_BIN_SPECS:
            sub = ranked[ranked["rank_bin"] == spec["label"]]
            avg_exposure = float(sub[exposure_col].mean())
            rows.append(
                {
                    "series": series,
                    "rank_bin": str(spec["label"]),
                    "rank_bin_order": int(sub["rank_bin_order"].iloc[0]),
                    "tick_label": str(spec["tick_label"]),
                    "cz_count": int(len(sub)),
                    "population_share": float(sub["working_age_25_64_cz"].sum()) / total_pop,
                    "average_exposure": avg_exposure,
                    "exposure_index": avg_exposure / baseline * 100.0,
                }
            )
    return pd.DataFrame(rows)


# Summarize education shares for each China and GPT exposure-rank bin.
# Report percentage-point differences from the average commuting-zone education share.
def build_geo_education_bins_pp(place: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = [
        ("ba_plus", "BA+ share", "ba_share_1990", "ba_share_2024"),
        ("hs_less", "HS-or-less share", "hs_less_share_1990", "hs_less_share_2024"),
    ]
    for metric, metric_label, china_share_col, gpt_share_col in specs:
        for series, exposure_col, share_col, year in [
            ("China Shock", "china_exposure", china_share_col, "1990"),
            ("AI Exposure", "gpt_exposure", gpt_share_col, "2024"),
        ]:
            ranked = add_rank_bins(place.dropna(subset=[share_col]).copy(), exposure_col)
            baseline = float(ranked[share_col].mean())
            for spec in RANK_BIN_SPECS:
                sub = ranked[ranked["rank_bin"] == spec["label"]]
                share = float(sub[share_col].mean())
                rows.append(
                    {
                        "metric": metric,
                        "metric_label": metric_label,
                        "series": series,
                        "rank_bin": str(spec["label"]),
                        "rank_bin_order": int(sub["rank_bin_order"].iloc[0]),
                        "tick_label": str(spec["tick_label"]),
                        "year": year,
                        "cz_count": int(len(sub)),
                        "share": share,
                        "baseline_share": baseline,
                        "pp_diff": (share - baseline) * 100.0,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    DATA_CLEAN.mkdir(parents=True, exist_ok=True)

    print("Building combined CZ exposure file...")
    place = build_place_exposure_cz()
    place.to_csv(DATA_CLEAN / "place_exposure_cz.csv", index=False)

    print("Building map paths...")
    build_projected_cz_map_paths().to_csv(DATA_CLEAN / "cz_map_paths.csv", index=False)

    print("Building geography rank-bin summaries...")
    build_geo_exposure_rank_bins(place).to_csv(DATA_CLEAN / "geo_exposure_rank_bins.csv", index=False)
    build_geo_education_bins_pp(place).to_csv(DATA_CLEAN / "geo_education_bins_pp.csv", index=False)


if __name__ == "__main__":
    main()
