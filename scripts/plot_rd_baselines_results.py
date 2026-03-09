#!/usr/bin/env python3
# pyright: reportMissingImports=false

from __future__ import annotations

import csv
import os
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
PLOT_SKIP_SAVED_RESULTS = False

RD_BASELINES_ORDER = ["VideoGS", "DracoGS", "MesonGS"]
RD_BASELINES_STYLES: dict[str, dict[str, Any]] = {
    "VideoGS": {"color": "#d62728", "marker": "^", "alpha": 0.7, "size": 35},
    "MesonGS": {"color": "#2ca02c", "marker": "s", "alpha": 0.45, "size": 18},
    "DracoGS": {"color": "#1f77b4", "marker": "o", "alpha": 0.45, "size": 18},
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
        baseline_rows = by_baseline.get(baseline, [])
        if not baseline_rows:
            continue
        style = RD_BASELINES_STYLES[baseline]
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


def main() -> None:
    csv_path = PLOT_CSV_PATH
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rows = read_rows(csv_path)
    if PLOT_DATASETS:
        allowed = set(PLOT_DATASETS)
        rows = [r for r in rows if str(r.get("dataset", "")) in allowed]
    if PLOT_SEQUENCES:
        allowed = set(PLOT_SEQUENCES)
        rows = [r for r in rows if str(r.get("sequence", "")) in allowed]

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
