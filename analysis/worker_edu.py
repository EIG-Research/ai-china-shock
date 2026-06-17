"""Create output/worker/01_edu.png from the cleaned worker education data.

The figure compares the education mix of the selected China worker group and the selected GPT worker groups.
It reads worker_group_summary.csv for group labels and worker_education_profile.csv for education shares.
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

BUCKETS = ["HS or Less", "Some College / AA", "Bachelor's", "Graduate"]
BUCKET_COLORS = ["#9B9186", "#D9B38C", "#C26A1B", "#7A5A3A"]


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
    # Load the group labels and the education shares created by build/workers.py.
    groups = read_csv("worker_group_summary.csv", ["group_id", "sort_order", "display_label"]).sort_values("sort_order")
    edu = read_csv("worker_education_profile.csv", ["group_id", "bucket", "share"])
    y = np.arange(len(groups))
    left = np.zeros(len(groups))

    # Draw one horizontal stacked bar per worker group.
    fig, ax = plt.subplots(figsize=(11, 5.8))
    for bucket, color in zip(BUCKETS, BUCKET_COLORS, strict=True):
        shares = []
        for group_id in groups["group_id"]:
            value = edu.loc[(edu["group_id"] == group_id) & (edu["bucket"] == bucket), "share"]
            if value.empty:
                raise ValueError(f"Missing education bucket {bucket!r} for group {group_id!r}")
            shares.append(float(value.iloc[0]))
        shares_arr = np.array(shares)
        ax.barh(y, shares_arr, left=left, color=color, edgecolor="white", label=bucket)

        # Label only slices that are large enough to read cleanly.
        for yi, x0, width in zip(y, left, shares_arr, strict=True):
            if width >= 0.08:
                ax.text(x0 + width / 2, yi, f"{width:.0%}", ha="center", va="center", fontsize=8, color="white")
        left += shares_arr

    ax.set_yticks(y, groups["display_label"].tolist())
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of selected workers")
    ax.set_title("Education composition of China Shock and AI-exposed worker groups")
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1.0))
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.27), ncol=4, frameon=False)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()

    out = OUTPUT / "worker" / "01_edu.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
