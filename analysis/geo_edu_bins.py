"""Create the two education rank-bin figures from cleaned geography data.

The figures compare education shares in commuting zones grouped by China and GPT exposure rank.
They read geo_education_bins_pp.csv and write one BA+ figure and one HS-or-less figure.
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


# Read a cleaned CSV and check that it contains the columns needed for these figures.
def read_csv(name: str, columns: list[str]) -> pd.DataFrame:
    path = DATA_CLEAN / name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run the build scripts first.")
    df = pd.read_csv(path)
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df


# Draw one education rank-bin chart.
# The plotted value is each rank bin's education share minus the average commuting-zone share, in percentage points.
def plot_metric(df: pd.DataFrame, metric: str, title: str, ylabel: str, out_name: str, ylim: tuple[float, float]) -> None:
    sub = df[df["metric"] == metric].copy()
    if sub.empty:
        raise ValueError(f"No rows for metric={metric!r}")
    series_order = ["China Shock", "AI Exposure"]
    bin_labels = sub.sort_values("rank_bin_order").drop_duplicates("rank_bin_order")["tick_label"].tolist()
    x = np.arange(len(bin_labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(10.5, 5.5))

    # Draw paired China and GPT bars for each exposure-rank bin.
    for i, series in enumerate(series_order):
        frame = sub[sub["series"] == series].sort_values("rank_bin_order")
        values = frame["pp_diff"].to_numpy(dtype=float)
        xpos = x + (i - 0.5) * width
        ax.bar(xpos, values, width=width * 0.9, color=COLORS[series], edgecolor="black", linewidth=0.4, label=series)

        # Put labels just outside each bar so positive and negative values remain readable.
        for xi, yi in zip(xpos, values, strict=True):
            va = "bottom" if yi >= 0 else "top"
            offset = 0.35 if yi >= 0 else -0.35
            ax.text(xi, yi + offset, f"{yi:+.1f}", ha="center", va=va, fontsize=8)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, bin_labels)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Commuting-zone rank bin")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    out = OUTPUT / "geo" / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    # Load the education rank-bin summaries created by build/geography.py.
    df = read_csv("geo_education_bins_pp.csv", ["metric", "series", "rank_bin_order", "tick_label", "pp_diff"])

    # Make one chart for BA+ shares and one chart for HS-or-less shares.
    plot_metric(
        df,
        metric="ba_plus",
        title="BA+ share in exposure-ranked commuting zones",
        ylabel="BA+ share relative to average CZ, percentage points",
        out_name="03_ba_bins_pp.png",
        ylim=(-12, 20),
    )
    plot_metric(
        df,
        metric="hs_less",
        title="HS-or-less share in exposure-ranked commuting zones",
        ylabel="HS-or-less share relative to average CZ, percentage points",
        out_name="04_hs_bins_pp.png",
        ylim=(-16, 20),
    )


if __name__ == "__main__":
    main()
