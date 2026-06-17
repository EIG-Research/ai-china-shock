"""Create the China worker cutoff outputs from cleaned worker exposure data.

Running this file writes three files to output/cutoff:
1. china_worker_cutoff_justification_curve.png shows the worker exposure curve and candidate cutoffs.
2. china_worker_cutoff_justification_drop.png shows the exposure drop at each candidate cutoff boundary.
3. china_worker_cutoff_industry_additions.txt lists the industries added as each candidate cutoff expands.

The script reads china_worker_industry_exposure.csv, which already contains worker weights by assigned China exposure industry.
"""

from __future__ import annotations

import argparse
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
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

CHINA_WORKER_SHARE = 0.04671415737844796
CHINA_WORKER_CUTOFF_LABEL = f"{CHINA_WORKER_SHARE * 100.0:.2f}%"

COLORS = {
    "china": "#7A5A3A",
    "accent": "#C26A1B",
    "dark": "#2F2A25",
}

# Each requested share is snapped to the next industry boundary in the ranked China exposure distribution.
CUTOFF_SPECS = (
    ("1.34%", 0.013444716630490713, "#9A948C", 1.8),
    ("1.88%", 0.01881626006485491, "#7F7A73", 1.8),
    ("2.24%", 0.02242029710519026, "#4B5563", 2.0),
    ("2.37%", 0.023707662655120296, "#2F855A", 2.0),
    ("3.29%", 0.032936255482506865, "#6E7FA6", 1.9),
    (CHINA_WORKER_CUTOFF_LABEL, CHINA_WORKER_SHARE, COLORS["china"], 2.6),
    ("9.94%", 0.09937764909599817, "#E0B350", 1.9),
)

CUTOFF_TEXT = (
    f"The main cutoff uses the exact {CHINA_WORKER_SHARE * 100.0:.4f}% worker-share cutoff "
    "in the current 25-64 CPS sample. "
    "China import penetration is 1991-2007 China import growth divided by 1991 domestic absorption; "
    "chart values are shown in percentage points."
)


# Read a cleaned CSV and check that it contains the columns needed for these outputs.
def read_csv(name: str, columns: list[str]) -> pd.DataFrame:
    path = DATA_CLEAN / name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run build/china.py first.")
    df = pd.read_csv(path)
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df


# Load worker weight by China exposure industry and convert exposure to percentage points.
def load_worker_industry_exposure() -> pd.DataFrame:
    df = read_csv(
        "china_worker_industry_exposure.csv",
        ["unit_id", "industry", "china_import_penetration", "worker_weight"],
    )
    df = df.rename(columns={"industry": "industry_name"}).copy()
    df["china_import_penetration_pp"] = pd.to_numeric(df["china_import_penetration"], errors="coerce") * 100.0
    df["worker_weight"] = pd.to_numeric(df["worker_weight"], errors="coerce")
    df = df.dropna(subset=["china_import_penetration_pp", "worker_weight"])
    return df[df["worker_weight"] > 0].copy()


# Build the exposure curve and the industry-boundary table used by both cutoff charts.
# The curve sorts workers from lowest to highest China exposure, while the boundary table ranks industries from highest to lowest exposure.
def prepare_cutoff_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    exposure = (
        load_worker_industry_exposure()[["unit_id", "industry_name", "worker_weight", "china_import_penetration_pp"]]
        .rename(columns={"china_import_penetration_pp": "exposure"})
        .sort_values(["exposure", "worker_weight", "unit_id"], ascending=[True, True, True])
        .reset_index(drop=True)
    )
    if exposure.empty:
        raise ValueError("No worker exposure data were available for the cutoff outputs.")

    total_worker_weight = float(exposure["worker_weight"].sum())
    exposure["cum_worker_share"] = exposure["worker_weight"].cumsum() / total_worker_weight

    boundaries = (
        exposure[["unit_id", "industry_name", "exposure", "worker_weight"]]
        .drop_duplicates(subset=["unit_id"])
        .sort_values(["exposure", "worker_weight", "unit_id"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    boundaries["worker_share"] = boundaries["worker_weight"] / total_worker_weight
    boundaries["cum_top_share"] = boundaries["worker_share"].cumsum()
    boundaries["prev_cum_top_share"] = boundaries["cum_top_share"].shift(fill_value=0.0)
    boundaries["next_exposure"] = boundaries["exposure"].shift(-1)
    return exposure, boundaries


# Find the industry boundary used for each requested cutoff.
# If a requested cutoff falls inside an industry, use the boundary after including that industry.
def build_cutoff_table(boundaries: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, requested_share, color, linewidth in CUTOFF_SPECS:
        diffs = np.abs(boundaries["cum_top_share"].to_numpy(dtype=float) - requested_share)
        closest_idx = int(np.argmin(diffs))
        if float(diffs[closest_idx]) <= 1e-8:
            boundary_idx = closest_idx
        else:
            matches = boundaries.index[boundaries["cum_top_share"] >= requested_share]
            if len(matches) == 0:
                raise ValueError(f"Requested cutoff {label} exceeds the available worker distribution.")
            boundary_idx = int(matches[0])
        boundary = boundaries.iloc[boundary_idx]
        rows.append(
            {
                "label": label,
                "requested_share": requested_share,
                "actual_boundary_share": float(boundary["cum_top_share"]),
                "color": color,
                "linewidth": linewidth,
                "industry_name": str(boundary["industry_name"]),
                "boundary_exposure": float(boundary["exposure"]),
                "next_exposure": float(boundary["next_exposure"]) if pd.notna(boundary["next_exposure"]) else np.nan,
            }
        )
    return pd.DataFrame(rows)


# Draw the China exposure distribution with vertical lines marking the candidate cutoffs.
def plot_cutoff_curve() -> None:
    exposure, boundaries = prepare_cutoff_data()
    cutoffs = build_cutoff_table(boundaries)

    x = np.r_[0.0, exposure["cum_worker_share"].to_numpy(dtype=float)]
    y = np.r_[float(exposure["exposure"].iloc[0]), exposure["exposure"].to_numpy(dtype=float)]
    y_max = float(exposure["exposure"].max())
    y_pad = max(3.0, y_max * 0.05)

    fig, axes = plt.subplots(3, 1, figsize=(11.5, 9.0), sharey=True)
    panel_specs = [
        ("Full distribution", 0.0, 1.0),
        ("Broader right tail", 0.80, 1.0),
        ("Right tail", 0.89, 1.0),
    ]

    for ax, (title, x_min, x_max) in zip(axes, panel_specs, strict=True):
        ax.step(x, y, where="post", color=COLORS["accent"], linewidth=2.0)
        ax.fill_between(x, y, 0.0, step="post", color=mcolors.to_rgba(COLORS["accent"], 0.12))
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0.0, y_max + y_pad)
        ax.set_title(title, loc="left", fontsize=11)
        ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0))
        ax.grid(axis="both", alpha=0.25)

    for row in cutoffs.itertuples(index=False):
        cutoff_x = 1.0 - float(row.actual_boundary_share)
        axes[-1].axvline(cutoff_x, color=row.color, linewidth=float(row.linewidth), alpha=0.95)

    legend_handles = [
        Line2D([0], [0], color=row.color, linewidth=float(row.linewidth), label=row.label)
        for row in cutoffs.itertuples(index=False)
    ]
    axes[-1].legend(handles=legend_handles, ncol=4, frameon=False, loc="upper left")
    axes[-1].set_xlabel("Worker exposure percentile")
    axes[1].set_ylabel("China import penetration, percentage points")
    fig.tight_layout()

    out = OUTPUT / "cutoff" / "china_worker_cutoff_justification_curve.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# Draw the exposure drop from each cutoff boundary industry to the next lower-exposure industry.
def plot_cutoff_drop() -> None:
    _exposure, boundaries = prepare_cutoff_data()
    cutoffs = build_cutoff_table(boundaries)
    cutoffs["drop_pct"] = (cutoffs["boundary_exposure"] - cutoffs["next_exposure"]) / cutoffs["boundary_exposure"]

    fig, ax = plt.subplots(figsize=(9.8, 5.25))
    y_pos = np.arange(len(cutoffs), dtype=float)
    x_min = float(np.floor(cutoffs[["boundary_exposure", "next_exposure"]].min().min()) - 0.4)
    x_max = float(np.ceil(cutoffs[["boundary_exposure", "next_exposure"]].max().max()) + 0.4)
    for yi, row in zip(y_pos, cutoffs.itertuples(index=False), strict=True):
        ax.hlines(yi, row.next_exposure, row.boundary_exposure, color=mcolors.to_rgba(row.color, 0.55), linewidth=3.0, zorder=1)
        ax.scatter(row.boundary_exposure, yi, s=95, color=row.color, edgecolor="white", linewidth=1.0, zorder=3)
        ax.scatter(row.next_exposure, yi, s=95, facecolor="white", edgecolor=row.color, linewidth=2.0, zorder=3)
        label = f"{row.boundary_exposure:.2f} -> {row.next_exposure:.2f} pp ({row.drop_pct:.1%})"
        if row.boundary_exposure > x_max - 7.5:
            label_x = row.next_exposure - 0.65
            label_ha = "right"
        else:
            label_x = row.boundary_exposure + 0.65
            label_ha = "left"
        ax.text(label_x, yi, label, ha=label_ha, va="center", fontsize=9, color="#5C544D")

    ax.set_yticks(y_pos, cutoffs["label"].tolist())
    ax.invert_yaxis()
    ax.set_xlabel("Boundary and Next-Industry Import Penetration (percentage points)", fontweight="bold")
    ax.set_xlim(x_min, x_max)
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    out = OUTPUT / "cutoff" / "china_worker_cutoff_justification_drop.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# Write the industry names added when moving from one candidate cutoff to the next.
def write_cutoff_industry_additions() -> None:
    _exposure, boundaries = prepare_cutoff_data()
    cutoffs = build_cutoff_table(boundaries)
    ordered = boundaries.sort_values("cum_top_share").reset_index(drop=True)

    lines = ["China worker cutoff industry additions", CUTOFF_TEXT, ""]
    previous_share = 0.0
    for row in cutoffs.itertuples(index=False):
        current_share = float(row.actual_boundary_share)
        added = ordered[
            (ordered["cum_top_share"] > previous_share + 1e-12)
            & (ordered["cum_top_share"] <= current_share + 1e-12)
        ]["industry_name"].tolist()
        noun = "industry" if len(added) == 1 else "industries"
        if previous_share == 0.0:
            lines.append(f"{row.label} includes {len(added)} {noun} total:")
        else:
            lines.append(f"{row.label} adds {len(added)} {noun}:")
        lines.extend([f"- {industry}" for industry in added])
        lines.append("")
        previous_share = current_share

    out = OUTPUT / "cutoff" / "china_worker_cutoff_industry_additions.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines).rstrip() + "\n")
    print(f"Saved {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create China worker cutoff outputs.")
    parser.add_argument(
        "target",
        nargs="?",
        default="all",
        choices=["all", "curve", "drop", "additions"],
        help="Which cutoff output to create.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target in {"all", "curve"}:
        plot_cutoff_curve()
    if args.target in {"all", "drop"}:
        plot_cutoff_drop()
    if args.target in {"all", "additions"}:
        write_cutoff_industry_additions()


if __name__ == "__main__":
    main()
