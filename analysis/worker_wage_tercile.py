"""Create output/worker/02_wage_tercile.png from the cleaned worker wage data.

The figure compares where each selected worker group falls in the bottom, middle, and top thirds of its period's wage distribution.
It reads worker_group_summary.csv for group labels and worker_wage_tercile_summary.csv for wage-tercile shares.
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

BUCKETS = ["Bottom Third", "Middle Third", "Top Third"]


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
    # Load the group labels and wage-tercile shares created by build/workers.py.
    groups = read_csv("worker_group_summary.csv", ["group_id", "sort_order", "kind", "display_label"]).sort_values("sort_order")
    summary = read_csv("worker_wage_tercile_summary.csv", ["group_id", "bucket", "share"])
    plot = groups.merge(summary, on="group_id", how="left")
    if plot["share"].isna().any():
        raise ValueError("Some worker groups are missing wage-tercile shares.")

    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    x = np.arange(len(BUCKETS))
    width = 0.12
    offsets = (np.arange(len(groups)) - (len(groups) - 1) / 2) * width

    # Draw one bar per worker group within each wage tercile.
    for offset, row in zip(offsets, groups.itertuples(index=False), strict=True):
        sub = plot[plot["group_id"] == row.group_id].set_index("bucket").reindex(BUCKETS)
        values = sub["share"].to_numpy(dtype=float)
        color = "#7A5A3A" if row.kind == "china" else "#C26A1B"
        alpha = 1.0 if row.kind == "china" else 0.45 + 0.09 * int(row.sort_order)
        ax.bar(x + offset, values, width=width * 0.92, color=color, alpha=alpha, edgecolor="black", linewidth=0.4, label=row.display_label)

        # Put a simple percent label above each bar.
        for xi, yi in zip(x + offset, values, strict=True):
            ax.text(xi, yi + 0.01, f"{yi:.0%}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x, BUCKETS)
    ax.set_ylim(0, 0.70)
    ax.set_ylabel("Share of selected workers")
    ax.set_title("Weekly earnings terciles of selected worker groups")
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1.0))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=8)
    fig.tight_layout()

    out = OUTPUT / "worker" / "02_wage_tercile.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
