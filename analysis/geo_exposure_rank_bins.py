"""Create output/geo/02_exposure_rank_bins.png from cleaned rank-bin data.

The figure compares average China and GPT exposure for commuting zones grouped by exposure rank.
It reads geo_exposure_rank_bins.csv, where each exposure value is indexed to the average commuting zone.
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"China Shock": "#7A5A3A", "AI Exposure": "#C26A1B"}


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


def main() -> None:
    # Load the rank-bin summaries created by build/geography.py.
    df = read_csv("geo_exposure_rank_bins.csv", ["series", "rank_bin_order", "tick_label", "exposure_index"])
    series_order = ["China Shock", "AI Exposure"]
    bin_labels = df.sort_values("rank_bin_order").drop_duplicates("rank_bin_order")["tick_label"].tolist()
    x = np.arange(len(bin_labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(10.5, 5.5))

    # Draw paired China and GPT bars for each exposure-rank bin.
    for i, series in enumerate(series_order):
        sub = df[df["series"] == series].sort_values("rank_bin_order")
        values = sub["exposure_index"].to_numpy(dtype=float)
        xpos = x + (i - 0.5) * width
        ax.bar(xpos, values, width=width * 0.9, color=COLORS[series], label=series, edgecolor="black", linewidth=0.4)
        for xi, yi in zip(xpos, values, strict=True):
            ax.text(xi, yi + 12, f"{yi:.1f}", ha="center", va="bottom", fontsize=8)

    # The dashed line marks the average exposure index across all commuting zones.
    ax.axhline(100, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(x, bin_labels)
    ax.set_ylabel("Average exposure index, all CZs = 100")
    ax.set_xlabel("Commuting-zone rank bin")
    ax.set_title("Exposure by ranked commuting zones")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    out = OUTPUT / "geo" / "02_exposure_rank_bins.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
