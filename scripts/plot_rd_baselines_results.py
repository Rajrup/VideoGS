#!/usr/bin/env python3
# pyright: reportMissingImports=false

from __future__ import annotations

import csv
import glob
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required for plotting. Install it in your environment "
        "(for example: pip install matplotlib)."
    ) from exc


SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parent
RD_BASELINES_RESULTS_ROOT = SCRIPTS_DIR / "rd_baselines_results"
DEFAULT_CSV = RD_BASELINES_RESULTS_ROOT / "collected" / "rd_baselines_results_all.csv"
DEFAULT_PLOTS_DIR = RD_BASELINES_RESULTS_ROOT / "plots"

PLOT_CSV_PATH = str(DEFAULT_CSV)
PLOT_OUTPUT_DIR = str(DEFAULT_PLOTS_DIR)
PLOT_DATASETS: list[str] | None = None
PLOT_SEQUENCES: list[str] | None = None
PLOT_VIDEOGS_GROUP_SIZES: list[int] | None = [20]
PLOT_SKIP_SAVED_RESULTS = False
PLOT_FORCE_RECOLLECT = False

# -- LivoGS hull overlay ---------------------------------------------------
# Change LIVOGS_RD_SUBDIR to switch between RD experiment folders, e.g.
#   "livogs_rd_nvcomp"  — original per-depth hull (convex_hull_*.csv)
#   "livogs_rd_new"     — AC/DC sweep hull (*_sweep_hull.csv)
LIVOGS_DATA_PATHS: dict[str, str | None] = {
    "HiFi4G": "/synology/rajrup/VideoGS",
    "N3DV": "/synology/rajrup/Queen",
}
LIVOGS_RD_SUBDIR: str = "livogs_rd_new"

RD_BASELINES_ORDER = ["VideoGS", "LivoGS", "DracoGS", "MesonGS"]
RD_BASELINES_STYLES: dict[str, dict[str, Any]] = {
    "VideoGS": {"color": "#d62728", "marker": "^", "alpha": 0.7, "size": 35},
    "LivoGS":  {"color": "#ff7f0e", "marker": "D", "alpha": 0.7, "size": 35},
    "MesonGS": {"color": "#2ca02c", "marker": "s", "alpha": 0.45, "size": 18},
    "DracoGS": {"color": "#1f77b4", "marker": "o", "alpha": 0.45, "size": 18},
}

LIVOGS_DATASET_DIR_ALIASES: dict[str, tuple[str, ...]] = {
    "HiFi4G": ("HiFi4G_Dataset",),
    "N3DV": ("Neural_3D_Video",),
}

LIVOGS_OUTPUT_ROOT_DIRS: dict[str, tuple[str, ...]] = {
    "HiFi4G": ("train_output",),
    "N3DV": ("pretrained_output", "train_output"),
}

LIVOGS_SEQUENCE_DIR_ALIASES: dict[str, tuple[str, ...]] = {
    "HiFi4G": (),
    "N3DV": ("queen_compressed_{sequence}",),
}


def _to_float(raw: str | None) -> float | None:
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def read_rows(csv_path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parsed: dict[str, Any] = dict(row)
            parsed["compressed_mb"] = _to_float(row.get("compressed_mb"))
            parsed["decomp_psnr"] = _to_float(row.get("decomp_psnr"))
            parsed["gt_psnr"] = _to_float(row.get("gt_psnr"))
            parsed["frame_id"] = int(row.get("frame_id", "0"))
            raw_group_size = row.get("group_size")
            if raw_group_size in (None, ""):
                parsed["group_size"] = None
            else:
                try:
                    parsed["group_size"] = int(raw_group_size)
                except ValueError:
                    parsed["group_size"] = None
            rows.append(parsed)
    return rows


def pareto_frontier(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not points:
        return []
    ordered = sorted(points, key=lambda p: p[0])
    frontier = [ordered[0]]
    for point in ordered[1:]:
        if point[1] > frontier[-1][1]:
            frontier.append(point)
    return frontier


def _find_livogs_hull_csv(dataset: str, sequence: str, frame_id: int) -> str | None:
    livogs_data_path = LIVOGS_DATA_PATHS.get(dataset)
    if livogs_data_path is None:
        return None

    output_roots = LIVOGS_OUTPUT_ROOT_DIRS.get(dataset, ("train_output",))
    dataset_dirs = [dataset, *LIVOGS_DATASET_DIR_ALIASES.get(dataset, ())]
    sequence_dirs = [
        sequence,
        *[template.format(sequence=sequence) for template in LIVOGS_SEQUENCE_DIR_ALIASES.get(dataset, ())],
    ]

    plot_dirs: list[str] = []
    for output_root in output_roots:
        root_dir = os.path.join(livogs_data_path, output_root)
        if not os.path.isdir(root_dir):
            continue

        for dataset_dir in dataset_dirs:
            for sequence_dir in sequence_dirs:
                plot_dir = os.path.join(
                    root_dir,
                    dataset_dir,
                    sequence_dir,
                    "compression",
                    LIVOGS_RD_SUBDIR,
                    "plots",
                )
                if os.path.isdir(plot_dir):
                    plot_dirs.append(plot_dir)

        if not plot_dirs:
            for sequence_dir in sequence_dirs:
                wildcard_pattern = os.path.join(
                    root_dir,
                    "*",
                    sequence_dir,
                    "compression",
                    LIVOGS_RD_SUBDIR,
                    "plots",
                )
                plot_dirs.extend(
                    path for path in sorted(glob.glob(wildcard_pattern)) if os.path.isdir(path)
                )

    if not plot_dirs:
        return None

    for plot_dir in plot_dirs:
        sweep = os.path.join(plot_dir, f"acdc_psnr_size_curve_frame{frame_id}_sweep_hull.csv")
        if os.path.isfile(sweep):
            return sweep

        convex = os.path.join(plot_dir, f"convex_hull_{dataset}_{sequence}_frame{frame_id}.csv")
        if os.path.isfile(convex):
            return convex

        for pattern in (
            f"*_frame{frame_id}_sweep_hull.csv",
            f"convex_hull_*_frame{frame_id}.csv",
        ):
            matches = sorted(glob.glob(os.path.join(plot_dir, pattern)))
            if matches:
                return matches[0]
    return None


def _load_livogs_hull_points(csv_path: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                points.append((float(row["compressed_mb"]), float(row["decomp_psnr"])))
            except (KeyError, ValueError):
                continue
    points.sort(key=lambda p: p[0])
    return points


def plot_group(
    dataset: str,
    sequence: str,
    frame_id: int,
    rows: list[dict[str, Any]],
    output_root: str,
    skip_saved_results: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))

    by_baseline: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_baseline[str(row.get("baseline", ""))].append(row)

    for baseline in RD_BASELINES_ORDER:
        style = RD_BASELINES_STYLES[baseline]
        baseline_rows = by_baseline.get(baseline, [])

        if baseline == "LivoGS" and not baseline_rows:
            hull_csv = _find_livogs_hull_csv(dataset, sequence, frame_id)
            if hull_csv is None:
                continue
            hull_pts = _load_livogs_hull_points(hull_csv)
            if not hull_pts:
                continue
            hx, hy = zip(*hull_pts)
            ax.scatter(
                hx, hy,
                color=style["color"],
                marker=style["marker"],
                s=style["size"],
                alpha=style["alpha"],
                label=f"LivoGS ({len(hull_pts)} pts)",
                edgecolors="none",
                zorder=3,
            )
            if len(hull_pts) >= 2:
                ax.plot(
                    hx, hy,
                    color=style["color"],
                    linewidth=1.5,
                    alpha=0.85,
                    zorder=4,
                )
            continue

        if not baseline_rows:
            continue
        xs = [float(r["compressed_mb"]) for r in baseline_rows if r.get("compressed_mb") is not None]
        ys = [float(r["decomp_psnr"]) for r in baseline_rows if r.get("decomp_psnr") is not None]
        if not xs or not ys:
            continue

        ax.scatter(
            xs,
            ys,
            color=style["color"],
            marker=style["marker"],
            s=style["size"],
            alpha=style["alpha"],
            label=f"{baseline} ({len(xs)} pts)",
            edgecolors="none",
            zorder=3,
        )

        frontier = pareto_frontier(list(zip(xs, ys)))
        if len(frontier) >= 2:
            fx, fy = zip(*frontier)
            ax.plot(
                fx,
                fy,
                color=style["color"],
                linewidth=1.5,
                alpha=0.85,
                zorder=4,
            )

    gt_candidates = [float(r["gt_psnr"]) for r in rows if r.get("gt_psnr") is not None]
    if gt_candidates:
        gt = gt_candidates[0]
        ax.axhline(
            gt,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label=f"GT ({gt:.2f} dB)",
            zorder=1,
        )

    ax.set_xlabel("Compressed Size (MB)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(f"R-D Curve | {dataset} | {sequence} | Frame {frame_id}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    dataset_dir = os.path.join(output_root, dataset)
    os.makedirs(dataset_dir, exist_ok=True)
    out_path = os.path.join(dataset_dir, f"rd_baselines_curve_{sequence}_frame{frame_id}.png")
    if skip_saved_results and os.path.isfile(out_path):
        print(f"SKIP saved plot: {out_path}")
        plt.close(fig)
        return

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def _collect_baselines_to_csv(csv_path: str) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_rd_baselines_experiments",
        str(SCRIPTS_DIR / "run_rd_baselines_experiments.py"),
    )
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)

    all_rows: list[dict[str, Any]] = []
    for ds_name, cfg in runner.ALL_DATASETS.items():
        for sequence in runner.SEQUENCE_SETTINGS[cfg.name]:
            for bl_key, collector in runner.BASELINE_COLLECTORS.items():
                rows = collector(cfg, sequence)
                all_rows.extend(rows)
                if rows:
                    print(f"  Collected {cfg.name} | {sequence} | {bl_key}: {len(rows)} rows")

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=runner.CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    print(f"  Collected {len(all_rows)} total rows -> {csv_path}")


def main() -> None:
    csv_path = PLOT_CSV_PATH

    if PLOT_FORCE_RECOLLECT:
        print("Re-collecting baseline results ...")
        _collect_baselines_to_csv(csv_path)

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rows = read_rows(csv_path)
    if PLOT_DATASETS:
        allowed = set(PLOT_DATASETS)
        rows = [r for r in rows if str(r.get("dataset", "")) in allowed]
    if PLOT_SEQUENCES:
        allowed = set(PLOT_SEQUENCES)
        rows = [r for r in rows if str(r.get("sequence", "")) in allowed]
    if PLOT_VIDEOGS_GROUP_SIZES is not None:
        allowed = set(PLOT_VIDEOGS_GROUP_SIZES)
        rows = [
            r
            for r in rows
            if str(r.get("baseline", "")) != "VideoGS" or r.get("group_size") in allowed
        ]

    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("dataset", "")),
            str(row.get("sequence", "")),
            int(row.get("frame_id", 0)),
        )
        groups[key].append(row)

    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    print(f"Input CSV:  {csv_path}")
    print(f"Output dir: {PLOT_OUTPUT_DIR}")
    print(f"Groups:     {len(groups)}")
    for (dataset, sequence, frame_id), group_rows in sorted(groups.items()):
        plot_group(dataset, sequence, frame_id, group_rows, PLOT_OUTPUT_DIR, PLOT_SKIP_SAVED_RESULTS)


if __name__ == "__main__":
    main()
