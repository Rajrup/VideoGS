#!/usr/bin/env python3
"""Plot RD curves for LiVoGS single-frame compression experiments.

Scans the experiment output directory structure produced by the compression
worker and plots:
  - x-axis: compressed size (MB)
  - y-axis: PSNR (dB)
  - one curve per beta value (beta_under_depth) or per depth (depth_under_beta)

Usable as a library (``plot.main(...)``) or standalone CLI::

    python scripts/livogs_baseline/rd_pipeline/plot.py --frame_id 0 --output_root ...
"""

import csv
import json
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------

def load_experiment_result(exp_dir: str, frame_id: int) -> dict | None:
    """Load one experiment's QP config and metrics.  Returns None on failure."""
    qp_config_path = os.path.join(exp_dir, "qp_config.json")
    benchmark_path = os.path.join(exp_dir, "benchmark_livogs.csv")
    eval_json_path = os.path.join(exp_dir, "evaluation", "evaluation_results.json")

    for p in (qp_config_path, benchmark_path, eval_json_path):
        if not os.path.exists(p):
            return None

    try:
        with open(qp_config_path, encoding="utf-8") as f:
            qp_config = json.load(f)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(f"  [SKIP] Invalid qp_config.json in {os.path.basename(exp_dir)}: {exc}")
        return None

    # Compressed size for the target frame
    compressed_bytes = None
    with open(benchmark_path, newline="") as f:
        for row in csv.DictReader(f):
            if int(row["frame_id"]) == frame_id:
                compressed_bytes = int(row["compressed_size_bytes"])
                break
    if compressed_bytes is None:
        return None

    try:
        with open(eval_json_path, encoding="utf-8") as f:
            eval_data = json.load(f)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(f"  [SKIP] Invalid evaluation_results.json in {os.path.basename(exp_dir)}: {exc}")
        return None

    # Per-frame metrics (prefer specific frame; fall back to summary)
    decomp_psnr = None
    gt_psnr = None
    for fr in eval_data.get("per_frame", []):
        if fr["frame"] == frame_id:
            decomp_psnr = fr["decomp_psnr"]
            gt_psnr = fr["gt_psnr"]
            break
    if decomp_psnr is None:
        summary = eval_data.get("summary", {})
        decomp_psnr = summary.get("decomp_psnr")
        gt_psnr = summary.get("gt_psnr")
    if decomp_psnr is None:
        return None

    try:
        baseline_qp = float(qp_config["baseline_qp"])
        beta = float(qp_config["beta"])
    except (KeyError, TypeError, ValueError):
        print(f"  [SKIP] Missing/invalid baseline_qp or beta in {os.path.basename(exp_dir)}")
        return None

    return {
        "label":            qp_config.get("label", os.path.basename(exp_dir)),
        "baseline_qp":      baseline_qp,
        "beta":             beta,
        "compressed_bytes":  compressed_bytes,
        "compressed_mb":     compressed_bytes / (1024 * 1024),
        "decomp_psnr":      decomp_psnr,
        "gt_psnr":          gt_psnr,
    }


def collect_results(
    frame_dir: str, frame_id: int, depth: Optional[int] = None,
) -> list[dict]:
    """Scan all experiment subdirectories and return a list of result dicts."""
    results: list[dict] = []
    if not os.path.isdir(frame_dir):
        print(f"[WARN] Directory not found: {frame_dir}")
        return results

    for entry in sorted(os.listdir(frame_dir)):
        exp_dir = os.path.join(frame_dir, entry)
        if not os.path.isdir(exp_dir):
            continue
        r = load_experiment_result(exp_dir, frame_id)
        if r is None:
            print(f"  [SKIP] Incomplete results in: {entry}")
        else:
            if depth is not None:
                r["depth"] = depth
            results.append(r)

    print(f"Loaded {len(results)} experiments from {frame_dir}")
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_rd_curves_by_beta(
    results: list[dict],
    frame_id: int,
    output_path: str,
    sequence_name: str,
    octree_depth: int,
    beta_values: list[float],
    psnr_range: Optional[tuple[float, float]] = None,
) -> None:
    """Plot RD curves with one curve per beta value."""
    if not results:
        print("[WARN] No results to plot.")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    gt_psnr = next((r["gt_psnr"] for r in results if r.get("gt_psnr") is not None), None)

    for beta in beta_values:
        beta_pts = [r for r in results if float(r["beta"]) == float(beta)]
        if not beta_pts:
            continue
        beta_pts.sort(key=lambda r: r["compressed_mb"])
        x = [r["compressed_mb"] for r in beta_pts]
        y = [r["decomp_psnr"]   for r in beta_pts]
        ax.plot(x, y, marker="o", linewidth=1.6, label=f"β={beta:.1f}")

    if gt_psnr is not None:
        ax.axhline(gt_psnr, color="black", linestyle="--", linewidth=1.4,
                    label=f"Uncompressed ({gt_psnr:.2f} dB)")

    ax.set_xlabel("Compressed size (MB)")
    ax.set_ylabel("PSNR (dB)")
    if psnr_range is not None:
        ax.set_ylim(psnr_range)
    ax.set_title(f"LiVoGS RD Curves — {sequence_name}  frame {frame_id}  J={octree_depth}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {output_path}")


def plot_rd_curves_by_depth(
    results: list[dict],
    frame_id: int,
    output_path: str,
    sequence_name: str,
    target_beta: float,
    plot_depths: list[int],
    psnr_range: Optional[tuple[float, float]] = None,
) -> None:
    """Plot RD curves with one curve per octree depth."""
    if not results:
        print("[WARN] No results to plot.")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    gt_psnr = next((r["gt_psnr"] for r in results if r.get("gt_psnr") is not None), None)

    beta_key = round(float(target_beta), 6)
    drawn = 0
    for depth in plot_depths:
        depth_pts = [
            r for r in results
            if int(r.get("depth", -1)) == int(depth) and round(float(r["beta"]), 6) == beta_key
        ]
        if not depth_pts:
            continue
        depth_pts.sort(key=lambda r: r["compressed_mb"])
        x = [r["compressed_mb"] for r in depth_pts]
        y = [r["decomp_psnr"]   for r in depth_pts]
        ax.plot(x, y, marker="o", linewidth=1.6, label=f"J={depth}")
        drawn += 1

    if drawn == 0:
        print(f"[WARN] No depth curves found for beta={target_beta:.6g}.")
        plt.close(fig)
        return

    if gt_psnr is not None:
        ax.axhline(gt_psnr, color="black", linestyle="--", linewidth=1.4,
                    label=f"Uncompressed ({gt_psnr:.2f} dB)")

    ax.set_xlabel("Compressed size (MB)")
    ax.set_ylabel("PSNR (dB)")
    if psnr_range is not None:
        ax.set_ylim(psnr_range)
    ax.set_title(f"LiVoGS RD Curves — {sequence_name}  frame {frame_id}  beta={target_beta:.6g}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {output_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _format_beta_tag(beta: float) -> str:
    return str(beta).replace("-", "m").replace(".", "p")


def _normalize_psnr_range(
    psnr_range: Optional[list[float] | tuple[float, float]],
) -> Optional[tuple[float, float]]:
    if psnr_range is None:
        return None
    if len(psnr_range) != 2:
        print(f"[WARN] Ignoring invalid psnr_range={psnr_range!r}: expected [min, max].")
        return None

    try:
        psnr_min = float(psnr_range[0])
        psnr_max = float(psnr_range[1])
    except (TypeError, ValueError):
        print(f"[WARN] Ignoring invalid psnr_range={psnr_range!r}: values must be numeric.")
        return None

    if psnr_min >= psnr_max:
        print(f"[WARN] Ignoring invalid psnr_range={psnr_range!r}: min must be < max.")
        return None

    return psnr_min, psnr_max


def main(
    frame_id: int,
    output_root: str,
    plot_output_dir: str,
    sequence_name: str,
    beta_values: list[float],
    baseline_qps: Optional[list[float]] = None,
    plot_mode: str = "beta_under_depth",
    octree_depth: int = config.J,
    target_beta: Optional[float] = None,
    plot_depths: Optional[list[int]] = None,
    psnr_range: Optional[list[float] | tuple[float, float]] = None,
) -> None:
    """Plot RD curves from experiment results.

    Parameters
    ----------
    output_root : str
        Directory containing ``frame_{id}/J_{depth}/...`` experiment trees.
    plot_output_dir : str
        Where to save the generated PNG plots.
    """
    if plot_mode == "beta_under_depth":
        frame_dir = os.path.join(output_root, f"frame_{frame_id}", f"J_{octree_depth}")
        results = collect_results(frame_dir, frame_id, depth=octree_depth)
    elif plot_mode == "depth_under_beta":
        depths = list(plot_depths) if plot_depths else [octree_depth]
        results = []
        for depth in depths:
            frame_dir = os.path.join(output_root, f"frame_{frame_id}", f"J_{depth}")
            results.extend(collect_results(frame_dir, frame_id, depth=depth))
        if target_beta is None:
            print("[WARN] --target_beta is required for plot_mode=depth_under_beta")
            return
    else:
        print(f"[WARN] Unsupported plot mode: {plot_mode}")
        return

    # Filter by baseline_qps if specified
    if baseline_qps is not None:
        qp_set = {round(q, 6) for q in baseline_qps}
        results = [r for r in results if round(r["baseline_qp"], 6) in qp_set]

    if not results:
        print("No valid results found — nothing to plot.")
        return

    normalized_psnr_range = _normalize_psnr_range(psnr_range)

    if plot_mode == "beta_under_depth":
        print(f"\n{'label':<30} {'beta':>6} {'qp':>8} {'MB':>8} {'PSNR':>8}")
        print("-" * 65)
        for r in sorted(results, key=lambda r: (r["beta"], r["baseline_qp"])):
            print(f"  {r['label']:<28} {r['beta']:>6.1f} {r['baseline_qp']:>8.4f} "
                  f"{r['compressed_mb']:>8.3f} {r['decomp_psnr']:>8.3f}")

        plot_path = os.path.join(
            plot_output_dir,
            f"rd_curves_{sequence_name}_frame{frame_id}_J{octree_depth}.png",
        )
        plot_rd_curves_by_beta(
            results,
            frame_id,
            plot_path,
            sequence_name,
            octree_depth,
            beta_values,
            psnr_range=normalized_psnr_range,
        )
    else:
        depths = list(plot_depths) if plot_depths else [octree_depth]
        beta_value = float(target_beta) if target_beta is not None else float("nan")
        print(f"\n{'label':<30} {'depth':>6} {'beta':>6} {'qp':>8} {'MB':>8} {'PSNR':>8}")
        print("-" * 73)
        for r in sorted(results, key=lambda r: (r.get("depth", -1), r["baseline_qp"])):
            print(f"  {r['label']:<28} {int(r.get('depth', -1)):>6d} {r['beta']:>6.1f} {r['baseline_qp']:>8.4f} "
                  f"{r['compressed_mb']:>8.3f} {r['decomp_psnr']:>8.3f}")

        beta_tag = _format_beta_tag(beta_value)
        plot_path = os.path.join(
            plot_output_dir,
            f"rd_curves_{sequence_name}_frame{frame_id}_beta{beta_tag}_across_depths.png",
        )
        plot_rd_curves_by_depth(
            results,
            frame_id,
            plot_path,
            sequence_name,
            beta_value,
            depths,
            psnr_range=normalized_psnr_range,
        )


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse as _ap

    _parser = _ap.ArgumentParser(description="Plot LiVoGS RD curves for a single frame")
    _parser.add_argument("--frame_id",        type=int,   default=0)
    _parser.add_argument("--output_root",     default=None,
                         help="Directory containing frame_N/ subdirs")
    _parser.add_argument("--plot_output_dir", default=None)
    _parser.add_argument("--sequence_name",   default="4K_Actor1_Greeting")
    _parser.add_argument("--beta_values",     type=float, nargs="*",
                         default=[0.0, 0.4, 0.8, 1.0, 1.2, 1.6, 2.0])
    _parser.add_argument("--baseline_qps",    type=float, nargs="*", default=None)
    _parser.add_argument("--plot_mode",       default="beta_under_depth",
                         choices=["beta_under_depth", "depth_under_beta"])
    _parser.add_argument("--octree_depth",    type=int, default=config.J)
    _parser.add_argument("--plot_depths",     type=int, nargs="*", default=None)
    _parser.add_argument("--target_beta",     type=float, default=None)
    _args = _parser.parse_args()

    _default_root = config.rd_output_root(
        config.DATA_PATH, "HiFi4G_Dataset", _args.sequence_name,
    )
    _output_root = _args.output_root or _default_root
    _plot_dir = _args.plot_output_dir or os.path.join(_output_root, "plots")

    main(
        frame_id=_args.frame_id,
        output_root=_output_root,
        plot_output_dir=_plot_dir,
        sequence_name=_args.sequence_name,
        beta_values=_args.beta_values,
        baseline_qps=_args.baseline_qps,
        plot_mode=_args.plot_mode,
        octree_depth=_args.octree_depth,
        target_beta=_args.target_beta,
        plot_depths=_args.plot_depths,
    )
