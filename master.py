"""Run the cleaned-data and figure pipeline from one command.

By default, this file runs every build script and every analysis script.
It can also run only the build files, only the figure files, one output folder,
or one specific figure.

Examples:
    python master.py
    python master.py build
    python master.py analysis
    python master.py worker
    python master.py geo
    python master.py cutoff
    python master.py worker_edu
    python master.py --list
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CommandParts = tuple[str, ...]


BUILD_STEPS = OrderedDict(
    [
        ("china", "build/china.py"),
        ("gpt", "build/gpt.py"),
        ("workers", "build/workers.py"),
        ("geography", "build/geography.py"),
    ]
)


FIGURE_STEPS = OrderedDict(
    [
        ("worker_edu", "analysis/worker_edu.py"),
        ("worker_wage_tercile", "analysis/worker_wage_tercile.py"),
        ("cutoff", "analysis/cutoff.py"),
        ("geo_map", "analysis/geo_map.py"),
        ("geo_exposure_rank_bins", "analysis/geo_exposure_rank_bins.py"),
        ("geo_edu_bins", "analysis/geo_edu_bins.py"),
    ]
)


ARGUMENT_STEPS: OrderedDict[str, CommandParts] = OrderedDict(
    [
        ("cutoff_curve", ("analysis/cutoff.py", "curve")),
        ("cutoff_drop", ("analysis/cutoff.py", "drop")),
        ("cutoff_additions", ("analysis/cutoff.py", "additions")),
    ]
)


TARGETS = {
    "all": [*BUILD_STEPS, *FIGURE_STEPS],
    "build": [*BUILD_STEPS],
    "analysis": [*FIGURE_STEPS],
    "figures": [*FIGURE_STEPS],
    "worker": ["china", "gpt", "workers", "worker_edu", "worker_wage_tercile"],
    "workers": ["workers"],
    "worker_figures": ["worker_edu", "worker_wage_tercile"],
    "cutoff_figures": ["cutoff"],
    "geo": ["china", "gpt", "geography", "geo_map", "geo_exposure_rank_bins", "geo_edu_bins"],
    "geography": ["geography"],
    "geo_figures": ["geo_map", "geo_exposure_rank_bins", "geo_edu_bins"],
    "01_edu": ["worker_edu"],
    "02_wage_tercile": ["worker_wage_tercile"],
    "cutoff_curve_png": ["cutoff_curve"],
    "cutoff_drop_png": ["cutoff_drop"],
    "cutoff_industry_additions": ["cutoff_additions"],
    "01_map": ["geo_map"],
    "02_exposure_rank_bins": ["geo_exposure_rank_bins"],
    "03_ba_bins_pp": ["geo_edu_bins"],
    "04_hs_bins_pp": ["geo_edu_bins"],
}


def all_step_commands() -> dict[str, CommandParts]:
    commands: dict[str, CommandParts] = {}
    commands.update({name: (path,) for name, path in BUILD_STEPS.items()})
    commands.update({name: (path,) for name, path in FIGURE_STEPS.items()})
    commands.update(ARGUMENT_STEPS)
    return commands


def expand_targets(targets: list[str]) -> list[tuple[str, CommandParts]]:
    step_commands = all_step_commands()
    expanded: list[str] = []
    for target in targets:
        if target in step_commands:
            expanded.append(target)
        elif target in TARGETS:
            expanded.extend(TARGETS[target])
        else:
            valid = sorted(set(step_commands) | set(TARGETS))
            raise ValueError(f"Unknown target {target!r}. Use --list to see options. Valid targets include: {', '.join(valid)}")

    unique_steps = list(OrderedDict.fromkeys(expanded))
    return [(step, step_commands[step]) for step in unique_steps]


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining_seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {remaining_seconds:.1f}s"
    hours, remaining_minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(remaining_minutes)}m {remaining_seconds:.1f}s"


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if value < 1024.0 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.1f} GB"


def file_stats(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    files = [item for item in path.rglob("*") if item.is_file()]
    return len(files), sum(item.stat().st_size for item in files)


def largest_files(path: Path, limit: int = 5) -> list[tuple[Path, int]]:
    if not path.exists():
        return []
    files = [(item, item.stat().st_size) for item in path.rglob("*") if item.is_file()]
    return sorted(files, key=lambda item: item[1], reverse=True)[:limit]


def memory_lines() -> list[str]:
    try:
        import resource
    except ImportError:
        return ["Peak memory: not available on this platform."]

    def rss_to_bytes(value: int) -> int:
        if sys.platform == "darwin":
            return int(value)
        return int(value) * 1024

    current = resource.getrusage(resource.RUSAGE_SELF)
    lines = [f"Peak master memory: {format_bytes(rss_to_bytes(current.ru_maxrss))}"]
    if hasattr(resource, "RUSAGE_CHILDREN"):
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
        lines.append(f"Peak child-process memory: {format_bytes(rss_to_bytes(children.ru_maxrss))}")
    return lines


def print_data_summary() -> None:
    print("\nData and output sizes:", flush=True)
    for label, path in [("raw data", ROOT / "data_raw"), ("clean data", ROOT / "data_clean"), ("outputs", ROOT / "output")]:
        count, total = file_stats(path)
        print(f"  {label:<10} {count:>5} files  {format_bytes(total):>9}", flush=True)

    for label, path in [("Largest raw files", ROOT / "data_raw"), ("Largest clean files", ROOT / "data_clean")]:
        files = largest_files(path)
        if not files:
            continue
        print(f"\n{label}:", flush=True)
        for item, size in files:
            print(f"  {item.relative_to(ROOT)}  {format_bytes(size)}", flush=True)

    print("\nMemory:", flush=True)
    for line in memory_lines():
        print(f"  {line}", flush=True)


def print_summary(timings: list[tuple[str, float]], total_seconds: float, dry_run: bool, total_steps: int) -> None:
    if dry_run:
        print("\nDry run complete. No scripts were run.", flush=True)
        return

    name_width = max([len(step_name) for step_name, _duration in timings] + [4])
    print("\nRun summary:", flush=True)
    for step_name, duration in timings:
        print(f"  [ok] {step_name:<{name_width}} {format_duration(duration)}", flush=True)
    print(f"\nCompleted {len(timings)}/{total_steps} files successfully.", flush=True)
    print(f"Everything ran successfully in {format_duration(total_seconds)}.", flush=True)
    print_data_summary()


def print_error_summary(timings: list[tuple[str, float]], failed_step: str, total_seconds: float, error: BaseException) -> None:
    print("\nRun summary:", flush=True)
    if timings:
        name_width = max(len(step_name) for step_name, _duration in timings)
        for step_name, duration in timings:
            print(f"  [ok] {step_name:<{name_width}} {format_duration(duration)}", flush=True)
    print(f"\nPipeline stopped with an error after {format_duration(total_seconds)}.", flush=True)
    if failed_step:
        print(f"Failed step: {failed_step}", flush=True)
    if isinstance(error, subprocess.CalledProcessError):
        print(f"Exit code: {error.returncode}", flush=True)
    else:
        print(f"Error: {error}", flush=True)


def run_step(
    step_name: str,
    command_parts: CommandParts,
    python_executable: str,
    dry_run: bool,
    completed_steps: int,
    total_steps: int,
) -> float:
    script_path = ROOT / command_parts[0]
    relative_path = script_path.relative_to(ROOT)
    command = [python_executable, str(relative_path), *command_parts[1:]]
    print(f"\n==> {step_name}: {' '.join(command)}", flush=True)
    if dry_run:
        return 0.0
    started_at = time.perf_counter()
    try:
        subprocess.run(command, cwd=ROOT, check=True)
    except subprocess.CalledProcessError:
        duration = time.perf_counter() - started_at
        print(f"[error] {step_name} failed after {format_duration(duration)}", flush=True)
        raise
    else:
        duration = time.perf_counter() - started_at
        print(
            f"[ok] {step_name} completed in {format_duration(duration)} "
            f"({completed_steps}/{total_steps} files completed successfully)",
            flush=True,
        )
        return duration


def print_targets() -> None:
    print("Common targets:")
    for target in ["all", "build", "analysis", "worker", "worker_figures", "cutoff", "geo", "geo_figures"]:
        print(f"  {target}")

    print("\nBuild steps:")
    for name, path in BUILD_STEPS.items():
        print(f"  {name:<24} {path}")

    print("\nFigure steps:")
    for name, path in FIGURE_STEPS.items():
        print(f"  {name:<24} {path}")

    print("\nSingle cutoff outputs:")
    for name, command_parts in ARGUMENT_STEPS.items():
        print(f"  {name:<24} {' '.join(command_parts)}")

    print("\nFigure filename aliases:")
    for target in [
        "01_edu",
        "02_wage_tercile",
        "cutoff_curve_png",
        "cutoff_drop_png",
        "cutoff_industry_additions",
        "01_map",
        "02_exposure_rank_bins",
        "03_ba_bins_pp",
        "04_hs_bins_pp",
    ]:
        print(f"  {target}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the cleaned-data and figure pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python master.py\n"
            "  python master.py build\n"
            "  python master.py analysis\n"
            "  python master.py worker\n"
            "  python master.py geo\n"
            "  python master.py cutoff\n"
            "  python master.py worker_edu\n"
            "  python master.py --python python3 all\n"
        ),
    )
    parser.add_argument(
        "targets",
        nargs="*",
        default=["all"],
        help="Targets to run. Use --list to see available targets.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run child scripts. Defaults to the Python running master.py.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the scripts that would run without running them.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available targets and exit.",
    )
    return parser.parse_args()


def main() -> None:
    started_at = time.perf_counter()
    args = parse_args()
    if args.list:
        print_targets()
        return

    steps = expand_targets(args.targets)
    print(f"Repo root: {ROOT}", flush=True)
    print(f"Python: {args.python}", flush=True)
    print(f"Targets: {', '.join(args.targets)}", flush=True)

    timings: list[tuple[str, float]] = []
    failed_step = ""
    try:
        for step_name, command_parts in steps:
            failed_step = step_name
            script_path = ROOT / command_parts[0]
            if not script_path.exists():
                raise FileNotFoundError(f"Expected script does not exist: {script_path}")
            duration = run_step(step_name, command_parts, args.python, args.dry_run, len(timings) + 1, len(steps))
            timings.append((step_name, duration))
    except subprocess.CalledProcessError as error:
        total_seconds = time.perf_counter() - started_at
        print_error_summary(timings, failed_step, total_seconds, error)
        raise SystemExit(error.returncode) from None
    except Exception as error:
        total_seconds = time.perf_counter() - started_at
        print_error_summary(timings, failed_step, total_seconds, error)
        raise SystemExit(1) from None

    total_seconds = time.perf_counter() - started_at
    print_summary(timings, total_seconds, args.dry_run, len(steps))


if __name__ == "__main__":
    main()
