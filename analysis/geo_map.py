"""Create output/geo/01_map.png from the cleaned commuting-zone map data.

The figure maps China and GPT exposure percentile ranks for the same commuting zones.
It reads place_exposure_cz.csv for exposure percentiles and cz_map_paths.csv for projected polygon paths.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_CLEAN = ROOT / "data_clean"
OUTPUT = ROOT / "output"

# Keep matplotlib cache files inside this repo when the script runs.
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd


# Read a cleaned CSV and check that it contains the columns needed for this figure.
def read_csv(name: str, columns: list[str]) -> pd.DataFrame:
    path = DATA_CLEAN / name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run the build scripts first.")
    df = pd.read_csv(path)
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df


# Draw one map panel by coloring each commuting-zone polygon with one exposure percentile.
def draw_panel(ax: plt.Axes, paths: pd.DataFrame, values: pd.DataFrame, value_col: str, title: str, cmap_name: str) -> None:
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    lookup = values.set_index("czone")[value_col]

    # Each commuting zone can have multiple polygon parts, so draw each part separately.
    for (czone, _part), sub in paths.groupby(["czone", "part"], sort=False):
        if czone not in lookup.index:
            continue
        ax.fill(sub["x"], sub["y"], facecolor=cmap(norm(float(lookup.loc[czone]))), edgecolor="white", linewidth=0.12)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main() -> None:
    # Load exposure ranks and plot-ready polygon paths created by build/geography.py.
    place = read_csv("place_exposure_cz.csv", ["czone", "china_pctile", "gpt_pctile"])
    paths = read_csv("cz_map_paths.csv", ["czone", "part", "order", "x", "y"]).sort_values(["czone", "part", "order"])

    # Draw China and GPT exposure ranks on the same map geometry.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    draw_panel(axes[0], paths, place, "china_pctile", "China Shock percentile rank", "YlOrBr")
    draw_panel(axes[1], paths, place, "gpt_pctile", "AI exposure percentile rank", "Oranges")
    fig.suptitle("Commuting-zone exposure ranks")
    fig.tight_layout()

    out = OUTPUT / "geo" / "01_map.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
