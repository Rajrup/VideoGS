#!/usr/bin/env python3
"""Collect results from DracoGS, MesonGS, and VideoGS baseline experiments
and generate RD comparison plots (PSNR vs compressed size).

Reads data produced by ``run_selected_experiments.sh``:
  - {output_folder}/benchmark_{baseline}.csv           → compressed_size_bytes
  - {output_folder}/evaluation/evaluation_results.json  → PSNR, SSIM

Usage::

    python scripts/collect_and_plot_baselines.py

Edit the CONFIGURATION section below to match your experiment setup.
"""

import csv
import json
import os
import sys
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


# ============================================================================
# CONFIGURATION  — edit to match your run_selected_experiments.sh settings
# ============================================================================

DATASET_NAME = "HiFi4G_Dataset"
DATA_PATH = "/synology/rajrup/VideoGS"

# Sequences and frame IDs (must match run_selected_experiments.sh EXPERIMENTS)
EXPERIMENTS: dict[str, list[int]] = {
    "4K_Actor1_Greeting":            [0, 10, 20, 50, 100, 150, 200],
    "4K_Actor2_Dancing":             [0, 10, 20, 50, 100, 150, 200],
    "4K_Actor3_Violin":              [0, 10, 20, 50, 100, 150, 200],
    "4K_Actor4_Dancing":             [0, 10, 20, 50, 100, 150, 200],
    "4K_Actor5_Oil-paper_Umbrella":  [0, 10, 20, 50, 100, 150, 200],
    "4K_Actor6_Changing_Clothes":    [0, 10, 20, 50, 100, 150, 200],
    "4K_Actor7_Nunchaku":            [0, 10, 20, 50, 100, 150, 200],
}

# Baseline configs — output_tag must match run_selected_experiments.sh paths
BASELINES: dict[str, dict[str, Any]] = {
    "DracoGS": {
        "output_tag": "eg_16_eo_16_et_16_es_16_cl_10",
        "subdir": "dracogs",
        "benchmark_csv": "benchmark_dracogs.csv",
    },
    "MesonGS": {
        "output_tag": "params_default",
        "subdir": "mesongs",
        "benchmark_csv": "benchmark_mesongs.csv",
    },
    "VideoGS": {
        "output_tag": "qp_25",
        "subdir": "videogs",
        "benchmark_csv": "benchmark_videogs_pipeline.csv",
    },
}

# Where to write collected CSV and plots
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "baseline_comparison_res")

# Plot style
BASELINE_STYLES: dict[str, dict[str, Any]] = {
    "DracoGS": {"color": "#1f77b4", "marker": "o", "label": "DracoGS"},
    "MesonGS": {"color": "#2ca02c", "marker": "s", "label": "MesonGS"},
    "VideoGS": {"color": "#d62728", "marker": "^", "label": "VideoGS"},
}

LIVOGS_HULL_ENABLED = True
LIVOGS_RD_SUBDIR = "livogs_rd_nvcomp"
LIVOGS_HULL_STYLE = {
    "color": "#ff7f0e",
    "linewidth": 1.8,
    "linestyle": "-",
    "marker": "D",
    "markersize": 4,
}


# ============================================================================
# COLLECTION
# ============================================================================

CSV_COLUMNS = [
    "sequence_name",
    "baseline",
    "frame_id",
    "compressed_size_bytes",
    "compressed_mb",
    "uncompressed_size_bytes",
    "uncompressed_mb",
    "encode_ms",
    "decode_ms",
    "gt_psnr",
    "gt_ssim",
    "decomp_psnr",
    "decomp_ssim",
    "psnr_drop",
    "ssim_drop",
]


def _get_output_folder(sequence: str, baseline_key: str) -> str:
    """Build the output folder path for a given baseline + sequence."""
    cfg = BASELINES[baseline_key]
    return os.path.join(
        DATA_PATH, "train_output", DATASET_NAME, sequence,
        "compression", cfg["subdir"], cfg["output_tag"],
    )


def _first_float(row: dict[str, str], keys: tuple[str, ...]) -> Optional[float]:
    for key in keys:
        raw = row.get(key)
        if raw in (None, ""):
            continue
        try:
            return float(raw)
        except ValueError:
            continue
    return None


def _load_sequence_results(
    output_folder: str,
    sequence: str,
    baseline: str,
    benchmark_csv_name: str,
    frame_ids: list[int],
) -> list[dict[str, Any]]:
    """Load all frame results for one baseline+sequence combination.

    Reads:
      - {output_folder}/{benchmark_csv_name}  →  compressed/uncompressed sizes
      - {output_folder}/evaluation/evaluation_results.json  →  PSNR, SSIM
    Returns a list of per-frame result dicts.
    """
    rows: list[dict[str, Any]] = []

    # --- Load benchmark CSV (compressed sizes) ---
    benchmark_path = os.path.join(output_folder, benchmark_csv_name)
    benchmark_by_frame: dict[int, dict[str, Any]] = {}
    if os.path.isfile(benchmark_path):
        try:
            with open(benchmark_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    fid = int(row["frame_id"])
                    benchmark_by_frame[fid] = {
                        "compressed_size_bytes": int(row["compressed_size_bytes"]),
                        "uncompressed_size_bytes": int(row.get("uncompressed_size_bytes", 0)),
                        "encode_ms": _first_float(
                            row,
                            ("total_encode_ms", "encode_time_ms", "encode_ms"),
                        ),
                        "decode_ms": _first_float(
                            row,
                            ("total_decode_ms", "decode_time_ms", "decode_ms"),
                        ),
                    }
        except (OSError, KeyError, ValueError) as e:
            print(f"  [WARN] Failed to read {benchmark_path}: {e}")
    else:
        print(f"  [WARN] Benchmark CSV not found: {benchmark_path}")

    # --- Load evaluation JSON (PSNR / SSIM) ---
    eval_json_path = os.path.join(output_folder, "evaluation", "evaluation_results.json")
    metrics_by_frame: dict[int, dict[str, float]] = {}
    if os.path.isfile(eval_json_path):
        try:
            with open(eval_json_path, encoding="utf-8") as f:
                eval_data = json.load(f)
            for fr in eval_data.get("per_frame", []):
                fid = int(fr["frame"])
                metrics_by_frame[fid] = {
                    "gt_psnr": float(fr["gt_psnr"]),
                    "gt_ssim": float(fr["gt_ssim"]),
                    "decomp_psnr": float(fr["decomp_psnr"]),
                    "decomp_ssim": float(fr["decomp_ssim"]),
                }
        except (OSError, json.JSONDecodeError, KeyError) as e:
            print(f"  [WARN] Failed to read {eval_json_path}: {e}")
    else:
        print(f"  [WARN] Evaluation JSON not found: {eval_json_path}")

    # --- Merge per frame ---
    for fid in frame_ids:
        if fid not in benchmark_by_frame:
            print(f"  [SKIP] {baseline} | {sequence} | frame {fid} (no benchmark data)")
            continue
        if fid not in metrics_by_frame:
            print(f"  [SKIP] {baseline} | {sequence} | frame {fid} (no evaluation data)")
            continue

        benchmark = benchmark_by_frame[fid]
        metrics = metrics_by_frame[fid]
        compressed = benchmark["compressed_size_bytes"]
        uncompressed = benchmark["uncompressed_size_bytes"]

        rows.append({
            "sequence_name": sequence,
            "baseline": baseline,
            "frame_id": fid,
            "compressed_size_bytes": compressed,
            "compressed_mb": compressed / (1024 * 1024),
            "uncompressed_size_bytes": uncompressed,
            "uncompressed_mb": uncompressed / (1024 * 1024),
            "encode_ms": benchmark.get("encode_ms"),
            "decode_ms": benchmark.get("decode_ms"),
            "gt_psnr": metrics["gt_psnr"],
            "gt_ssim": metrics["gt_ssim"],
            "decomp_psnr": metrics["decomp_psnr"],
            "decomp_ssim": metrics["decomp_ssim"],
            "psnr_drop": metrics["gt_psnr"] - metrics["decomp_psnr"],
            "ssim_drop": metrics["gt_ssim"] - metrics["decomp_ssim"],
        })

    return rows


def collect_all_results() -> list[dict[str, Any]]:
    """Walk all experiments and collect per-frame results."""
    rows: list[dict[str, Any]] = []
    for sequence, frame_ids in EXPERIMENTS.items():
        for baseline, cfg in BASELINES.items():
            output_folder = _get_output_folder(sequence, baseline)
            seq_rows = _load_sequence_results(
                output_folder, sequence, baseline,
                cfg["benchmark_csv"], frame_ids,
            )
            rows.extend(seq_rows)
    return rows


def write_csv(rows: list[dict[str, Any]], path: str) -> None:
    """Write collected rows to CSV."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in CSV_COLUMNS})
    print(f"  Wrote {len(rows)} rows to: {path}")


# ============================================================================
# PLOTTING
# ============================================================================

def _group_by(
    rows: list[dict[str, Any]], key: str,
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row[key], []).append(row)
    return groups


def _iter_frame_groups(
    rows: list[dict[str, Any]],
) -> list[tuple[str, int, list[dict[str, Any]]]]:
    by_seq = _group_by(rows, "sequence_name")
    grouped: list[tuple[str, int, list[dict[str, Any]]]] = []
    for seq_name, seq_rows in sorted(by_seq.items()):
        frame_ids = sorted({int(r["frame_id"]) for r in seq_rows})
        for frame_id in frame_ids:
            frame_rows = [r for r in seq_rows if int(r["frame_id"]) == frame_id]
            if frame_rows:
                grouped.append((seq_name, frame_id, frame_rows))
    return grouped


def _resolve_livogs_hull_csv(sequence: str, frame_id: int) -> Optional[str]:
    plot_dir = os.path.join(
        DATA_PATH,
        "train_output",
        DATASET_NAME,
        sequence,
        "compression",
        LIVOGS_RD_SUBDIR,
        "plots",
    )
    exact_name = f"convex_hull_{DATASET_NAME}_{sequence}_frame{frame_id}.csv"
    exact_path = os.path.join(plot_dir, exact_name)
    if os.path.isfile(exact_path):
        return exact_path

    if not os.path.isdir(plot_dir):
        return None

    suffix = f"_frame{frame_id}.csv"
    prefix = f"convex_hull_{DATASET_NAME}_"
    candidates = [
        os.path.join(plot_dir, name)
        for name in os.listdir(plot_dir)
        if name.startswith(prefix) and name.endswith(suffix)
    ]
    if not candidates:
        return None

    candidates.sort()
    return candidates[0]


def _load_livogs_hull_points(sequence: str, frame_id: int) -> list[tuple[float, float]]:
    hull_csv = _resolve_livogs_hull_csv(sequence, frame_id)
    if hull_csv is None:
        return []

    points: list[tuple[float, float]] = []
    try:
        with open(hull_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                x = float(row["compressed_mb"])
                y = float(row["decomp_psnr"])
                points.append((x, y))
    except (OSError, KeyError, ValueError):
        return []

    points.sort(key=lambda p: p[0])
    return points


def plot_psnr_size_per_frame(
    rows: list[dict[str, Any]],
    plot_dir: str,
) -> None:
    """PSNR-size figure per sequence-frame."""
    for seq_name, frame_id, frame_rows in _iter_frame_groups(rows):
        fig, ax = plt.subplots(figsize=(9, 6))
        by_baseline = _group_by(frame_rows, "baseline")

        for baseline in BASELINES:
            if baseline not in by_baseline:
                continue
            bl_rows = by_baseline[baseline]
            style = BASELINE_STYLES[baseline]

            xs = [r["compressed_mb"] for r in bl_rows]
            ys = [r["decomp_psnr"] for r in bl_rows]
            avg_x = float(np.mean(xs))
            avg_y = float(np.mean(ys))
            ax.scatter(
                [avg_x], [avg_y],
                color=style["color"],
                marker=style["marker"],
                s=120, alpha=1.0, zorder=4,
                edgecolors="black", linewidths=0.8,
                label=f"{style['label']} ({avg_y:.2f} dB, {avg_x:.2f} MB)",
            )

        if LIVOGS_HULL_ENABLED:
            hull_points = _load_livogs_hull_points(seq_name, frame_id)
            if hull_points:
                print(f"  LiVoGS hull: {seq_name} frame {frame_id} ({len(hull_points)} points)")
                hx = [p[0] for p in hull_points]
                hy = [p[1] for p in hull_points]
                ax.plot(
                    hx,
                    hy,
                    color=LIVOGS_HULL_STYLE["color"],
                    linewidth=LIVOGS_HULL_STYLE["linewidth"],
                    linestyle=LIVOGS_HULL_STYLE["linestyle"],
                    marker=LIVOGS_HULL_STYLE["marker"],
                    markersize=LIVOGS_HULL_STYLE["markersize"],
                    alpha=0.95,
                    zorder=3,
                    label="LiVoGS hull",
                )

        gt_psnrs = [r["gt_psnr"] for r in frame_rows if r.get("gt_psnr")]
        if gt_psnrs:
            gt_mean = float(np.mean(gt_psnrs))
            ax.axhline(
                gt_mean, color="black", linestyle="--", linewidth=1.4,
                label=f"Uncompressed ({gt_mean:.2f} dB)", zorder=1,
            )

        ax.set_xlabel("Compressed Size (MB)", fontsize=11)
        ax.set_ylabel("PSNR (dB)", fontsize=11)
        ax.set_title(f"PSNR-Size | {seq_name} | Frame {frame_id}", fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="lower right")
        fig.tight_layout()

        out_path = os.path.join(plot_dir, f"psnr_size_{seq_name}_frame{frame_id}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out_path}")


def plot_all_sequences_summary(
    rows: list[dict[str, Any]],
    plot_dir: str,
) -> None:
    """Summary plot: one subplot per sequence arranged in a grid.

    Each subplot shows per-baseline average (size, PSNR).
    """
    by_seq = _group_by(rows, "sequence_name")
    seq_names = sorted(by_seq.keys())
    n = len(seq_names)
    if n == 0:
        return

    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()

    for idx, seq_name in enumerate(seq_names):
        ax = axes_flat[idx]
        seq_rows = by_seq[seq_name]
        by_baseline = _group_by(seq_rows, "baseline")

        for baseline in BASELINES:
            if baseline not in by_baseline:
                continue
            bl_rows = by_baseline[baseline]
            style = BASELINE_STYLES[baseline]
            avg_x = float(np.mean([r["compressed_mb"] for r in bl_rows]))
            avg_y = float(np.mean([r["decomp_psnr"] for r in bl_rows]))

            ax.scatter(
                [avg_x], [avg_y],
                color=style["color"],
                marker=style["marker"],
                s=80, alpha=1.0, zorder=3,
                edgecolors="black", linewidths=0.6,
                label=style["label"],
            )

        # GT reference
        gt_psnrs = [r["gt_psnr"] for r in seq_rows if r.get("gt_psnr")]
        if gt_psnrs:
            ax.axhline(
                float(np.mean(gt_psnrs)),
                color="black", linestyle="--", linewidth=1.0,
                alpha=0.6, zorder=1,
            )

        # Short sequence name for title
        short_name = seq_name.replace("4K_", "").replace("_", " ")
        ax.set_title(short_name, fontsize=10)
        ax.set_xlabel("Size (MB)", fontsize=8)
        ax.set_ylabel("PSNR (dB)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)
        if idx == 0:
            ax.legend(fontsize=7, loc="lower right")

    # Hide unused subplots
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        "Baseline Comparison — PSNR vs Compressed Size",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    out_path = os.path.join(plot_dir, "rd_summary_all_sequences.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_averaged_bar(
    rows: list[dict[str, Any]],
    plot_dir: str,
) -> None:
    """Bar chart: average PSNR and average size per baseline across all
    sequences.  Two subplots side by side.
    """
    by_baseline = _group_by(rows, "baseline")
    baselines_present = [b for b in BASELINES if b in by_baseline]
    if not baselines_present:
        return

    avg_psnr = []
    avg_size = []
    colors = []
    labels = []

    for bl in baselines_present:
        bl_rows = by_baseline[bl]
        avg_psnr.append(float(np.mean([r["decomp_psnr"] for r in bl_rows])))
        avg_size.append(float(np.mean([r["compressed_mb"] for r in bl_rows])))
        colors.append(BASELINE_STYLES[bl]["color"])
        labels.append(BASELINE_STYLES[bl]["label"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    x = np.arange(len(baselines_present))
    bar_width = 0.5

    # PSNR bars
    bars1 = ax1.bar(x, avg_psnr, bar_width, color=colors, edgecolor="black",
                     linewidth=0.6)
    ax1.set_ylabel("PSNR (dB)", fontsize=11)
    ax1.set_title("Average PSNR", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars1, avg_psnr):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    # Size bars
    bars2 = ax2.bar(x, avg_size, bar_width, color=colors, edgecolor="black",
                     linewidth=0.6)
    ax2.set_ylabel("Compressed Size (MB)", fontsize=11)
    ax2.set_title("Average Compressed Size", fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars2, avg_size):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(
        "Baseline Comparison — Averaged Across All Sequences & Frames",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()

    out_path = os.path.join(plot_dir, "rd_averaged_bars.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_latency_method_per_frame(
    rows: list[dict[str, Any]],
    plot_dir: str,
) -> None:
    for seq_name, frame_id, frame_rows in _iter_frame_groups(rows):
        by_baseline = _group_by(frame_rows, "baseline")
        baselines_present = [b for b in BASELINES if b in by_baseline]
        if not baselines_present:
            continue

        labels = [BASELINE_STYLES[b]["label"] for b in baselines_present]
        encode_vals: list[float] = []
        decode_vals: list[float] = []

        for bl in baselines_present:
            bl_rows = by_baseline[bl]
            enc = [float(r["encode_ms"]) for r in bl_rows if r.get("encode_ms") is not None]
            dec = [float(r["decode_ms"]) for r in bl_rows if r.get("decode_ms") is not None]
            encode_vals.append(float(np.mean(enc)) if enc else np.nan)
            decode_vals.append(float(np.mean(dec)) if dec else np.nan)

        x = np.arange(len(baselines_present))
        width = 0.34

        fig, ax = plt.subplots(figsize=(9, 5.5))
        enc_plot = [v if np.isfinite(v) else 0.0 for v in encode_vals]
        dec_plot = [v if np.isfinite(v) else 0.0 for v in decode_vals]

        bars_enc = ax.bar(x - width / 2, enc_plot, width, color="#4e79a7", label="Encode")
        bars_dec = ax.bar(x + width / 2, dec_plot, width, color="#f28e2b", label="Decode")

        for i, v in enumerate(encode_vals):
            if np.isfinite(v):
                ax.text(x[i] - width / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
            else:
                bars_enc[i].set_alpha(0.18)
                ax.text(x[i] - width / 2, 0.0, "N/A", ha="center", va="bottom", fontsize=8)

        for i, v in enumerate(decode_vals):
            if np.isfinite(v):
                ax.text(x[i] + width / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
            else:
                bars_dec[i].set_alpha(0.18)
                ax.text(x[i] + width / 2, 0.0, "N/A", ha="center", va="bottom", fontsize=8)

        finite_vals = [v for v in encode_vals + decode_vals if np.isfinite(v)]
        ymax = max(finite_vals) * 1.18 if finite_vals else 1.0
        ax.set_ylim(0.0, ymax)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("Latency (ms/frame)", fontsize=11)
        ax.set_title(f"Latency by Method | {seq_name} | Frame {frame_id}", fontsize=13)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()

        out_path = os.path.join(plot_dir, f"latency_method_{seq_name}_frame{frame_id}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")


def plot_size_method_per_frame(
    rows: list[dict[str, Any]],
    plot_dir: str,
) -> None:
    for seq_name, frame_id, frame_rows in _iter_frame_groups(rows):
        by_baseline = _group_by(frame_rows, "baseline")
        baselines_present = [b for b in BASELINES if b in by_baseline]
        if not baselines_present:
            continue

        labels = [BASELINE_STYLES[b]["label"] for b in baselines_present]
        compressed_vals: list[float] = []
        uncompressed_vals: list[float] = []
        colors: list[str] = []

        for bl in baselines_present:
            bl_rows = by_baseline[bl]
            comp = [float(r["compressed_mb"]) for r in bl_rows if r.get("compressed_mb") is not None]
            uncomp = [float(r["uncompressed_mb"]) for r in bl_rows if r.get("uncompressed_mb") is not None]
            compressed_vals.append(float(np.mean(comp)) if comp else np.nan)
            uncompressed_vals.append(float(np.mean(uncomp)) if uncomp else np.nan)
            colors.append(BASELINE_STYLES[bl]["color"])

        x = np.arange(len(baselines_present))
        fig, ax = plt.subplots(figsize=(8.5, 5.5))

        comp_plot = [v if np.isfinite(v) else 0.0 for v in compressed_vals]
        bars = ax.bar(x, comp_plot, width=0.55, color=colors, edgecolor="black", linewidth=0.7)

        for i, v in enumerate(compressed_vals):
            if np.isfinite(v):
                ax.text(x[i], v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
            else:
                bars[i].set_alpha(0.18)
                ax.text(x[i], 0.0, "N/A", ha="center", va="bottom", fontsize=8)

        finite_comp = [v for v in compressed_vals if np.isfinite(v)]
        ymax = max(finite_comp) * 1.18 if finite_comp else 1.0
        ax.set_ylim(0.0, ymax)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("Compressed Size (MB/frame)", fontsize=11)
        ax.set_title(f"Size by Method | {seq_name} | Frame {frame_id}", fontsize=13)
        ax.grid(axis="y", alpha=0.3)

        legend_handles = []
        legend_labels = []
        for i, bl in enumerate(baselines_present):
            handle = bars[i]
            if np.isfinite(uncompressed_vals[i]):
                legend_text = f"{BASELINE_STYLES[bl]['label']} (Uncompressed: {uncompressed_vals[i]:.2f} MB)"
            else:
                legend_text = f"{BASELINE_STYLES[bl]['label']} (Uncompressed: N/A)"
            legend_handles.append(handle)
            legend_labels.append(legend_text)
        ax.legend(legend_handles, legend_labels, fontsize=8, loc="upper left")

        fig.tight_layout()
        out_path = os.path.join(plot_dir, f"size_method_{seq_name}_frame{frame_id}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    sep = "=" * 70
    print(sep)
    print("Baseline Comparison — Collect & Plot")
    print(f"  Sequences:  {len(EXPERIMENTS)}")
    print(f"  Baselines:  {', '.join(BASELINES.keys())}")
    print(f"  Output:     {OUTPUT_DIR}")
    print(sep)

    # Step 1: Collect
    print(f"\n{sep}\nStep 1: Collect results\n{sep}")
    rows = collect_all_results()
    print(f"\n  Total results collected: {len(rows)}")

    if not rows:
        print("[ERROR] No results found. Did run_selected_experiments.sh complete?")
        sys.exit(1)

    # Quick summary
    by_baseline = _group_by(rows, "baseline")
    for bl in BASELINES:
        bl_rows = by_baseline.get(bl, [])
        if bl_rows:
            avg_psnr = np.mean([r["decomp_psnr"] for r in bl_rows])
            avg_size = np.mean([r["compressed_mb"] for r in bl_rows])
            print(f"    {bl:10s}: {len(bl_rows):3d} frames, "
                  f"avg PSNR={avg_psnr:.2f} dB, avg size={avg_size:.2f} MB")

    # Write CSV
    csv_path = os.path.join(OUTPUT_DIR, "baseline_results.csv")
    write_csv(rows, csv_path)

    # Step 2: Plot
    print(f"\n{sep}\nStep 2: Generate plots\n{sep}")
    plot_dir = os.path.join(OUTPUT_DIR, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    for name in os.listdir(plot_dir):
        if name.endswith(".png"):
            os.remove(os.path.join(plot_dir, name))

    plot_psnr_size_per_frame(rows, plot_dir)
    plot_latency_method_per_frame(rows, plot_dir)
    plot_size_method_per_frame(rows, plot_dir)

    print(f"\n{sep}")
    print(f"Done! All outputs in: {OUTPUT_DIR}")
    print(sep)


if __name__ == "__main__":
    main()
