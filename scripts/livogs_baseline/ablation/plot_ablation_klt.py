#!/usr/bin/env python3
"""
Plot KLT color space ablation results.

Reads ablation_klt.csv from each sequence directory and optionally
evaluation_results.json for quality metrics.

Generates:
  1. klt_compressed_size     -- grouped bar chart of total compressed size
  2. klt_size_breakdown      -- stacked bar chart of per-attribute size breakdown
  3. klt_quality_vs_size     -- scatter plot of PSNR vs compressed size (if eval exists)
  4. klt_transform_time      -- grouped bar chart of encode/decode time overhead vs RGB
  5. Prints a text summary table to stdout
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Dataset registry: label -> path to ablation output directory
# ---------------------------------------------------------------------------
DATASETS = {
    "N3DV\nFlame Salmon": (
        "/synology/rajrup/Queen/pretrained_output/Neural_3D_Video"
        "/queen_compressed_flame_salmon_1/ablation/livogs_klt"
    ),
    "N3DV\nSear Steak": (
        "/synology/rajrup/Queen/pretrained_output/Neural_3D_Video"
        "/queen_compressed_sear_steak/ablation/livogs_klt"
    ),
    "HiFi4G\nActor1": (
        "/synology/rajrup/VideoGS/train_output/HiFi4G_Dataset"
        "/4K_Actor1_Greeting/ablation/livogs_klt"
    ),
}

VARIANT_ORDER = ["klt", "yuv", "rgb"]
VARIANT_LABELS = {"klt": "KLT", "yuv": "YUV", "rgb": "RGB"}
VARIANT_COLORS = {"klt": "#2ca02c", "yuv": "#1f77b4", "rgb": "#d62728"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_csv(path):
    """Load ablation_klt.csv into a list of dicts."""
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            parsed = {}
            for k, v in row.items():
                if k == "variant":
                    parsed[k] = v
                else:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = v
            parsed["frame_id"] = int(parsed["frame_id"])
            rows.append(parsed)
    return rows


def load_evaluation(eval_dir):
    """Load evaluation_results.json if it exists.
    Returns the 'summary' sub-dict (with decomp_psnr, decomp_ssim, etc.)."""
    json_path = os.path.join(eval_dir, "evaluation_results.json")
    if os.path.isfile(json_path):
        with open(json_path) as f:
            data = json.load(f)
        # Support both flat and nested (summary) JSON formats
        if "summary" in data:
            return data["summary"]
        return data
    return None


def aggregate(rows):
    """Aggregate metrics per variant (mean across frames)."""
    by_variant = defaultdict(list)
    for r in rows:
        by_variant[r["variant"]].append(r)

    agg = {}
    for v, vrows in by_variant.items():
        agg[v] = {
            "compressed_size_bytes": np.mean(
                [r["compressed_size_bytes"] for r in vrows]
            ),
            "compressed_size_bytes_std": np.std(
                [r["compressed_size_bytes"] for r in vrows]
            ),
            "position_compressed_bytes": np.mean(
                [r["position_compressed_bytes"] for r in vrows]
            ),
            "attribute_compressed_bytes": np.mean(
                [r["attribute_compressed_bytes"] for r in vrows]
            ),
            "encode_time_ms": np.mean([r["encode_time_ms"] for r in vrows]),
            "decode_time_ms": np.mean([r["decode_time_ms"] for r in vrows]),
        }

        # Per-attribute breakdown from CSV columns
        quats_cols = [f"quats_dim{i}_compressed_bytes" for i in range(4)]
        scales_cols = [f"scales_dim{i}_compressed_bytes" for i in range(3)]
        opacity_col = "opacity_dim0_compressed_bytes"
        sh_dc_cols = [f"sh_dim{i}_compressed_bytes" for i in range(3)]
        sh_rest_cols = sorted(
            [k for k in vrows[0] if k.startswith("sh_dim")
             and k not in {f"sh_dim{i}_compressed_bytes" for i in range(3)}]
        )

        agg[v]["quats_bytes"] = np.mean(
            [sum(r.get(c, 0) for c in quats_cols) for r in vrows]
        )
        agg[v]["scales_bytes"] = np.mean(
            [sum(r.get(c, 0) for c in scales_cols) for r in vrows]
        )
        agg[v]["opacity_bytes"] = np.mean(
            [r.get(opacity_col, 0) for r in vrows]
        )
        agg[v]["sh_dc_bytes"] = np.mean(
            [sum(r.get(c, 0) for c in sh_dc_cols) for r in vrows]
        )
        agg[v]["sh_rest_bytes"] = np.mean(
            [sum(r.get(c, 0) for c in sh_rest_cols) for r in vrows]
        )
    return agg


# ---------------------------------------------------------------------------
# Figure 1: Compressed size comparison
# ---------------------------------------------------------------------------
def plot_compressed_size(all_data, dataset_labels, output_folder, fmt):
    """Grouped bar chart: total compressed size per variant per dataset."""
    variants = [v for v in VARIANT_ORDER if any(v in d for d in all_data.values())]
    n_variants = len(variants)
    n_datasets = len(dataset_labels)
    bar_width = 0.8 / n_variants
    x = np.arange(n_datasets)

    fig, ax = plt.subplots(figsize=(max(8, n_datasets * 3), 5))

    for vi, variant in enumerate(variants):
        offset = (vi - n_variants / 2 + 0.5) * bar_width
        color = VARIANT_COLORS[variant]
        sizes_mb = [
            all_data[ds].get(variant, {}).get("compressed_size_bytes", 0) / (1024 * 1024)
            for ds in dataset_labels
        ]
        bars = ax.bar(
            x + offset, sizes_mb, bar_width, color=color, alpha=0.85,
            edgecolor="white", label=VARIANT_LABELS[variant],
        )
        for bar, val in zip(bars, sizes_mb):
            ax.annotate(
                f"{val:.3f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center", va="bottom", fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(dataset_labels, fontsize=10)
    ax.set_ylabel("Avg Compressed Size (MB)", fontsize=11)
    ax.set_title("KLT Ablation: Compressed Size by Color Space", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    out = os.path.join(output_folder, f"klt_compressed_size.{fmt}")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 2: Per-attribute size breakdown
# ---------------------------------------------------------------------------
def plot_size_breakdown(all_data, dataset_labels, output_folder, fmt):
    """Stacked bar chart: size breakdown by attribute per variant per dataset."""
    variants = [v for v in VARIANT_ORDER if any(v in d for d in all_data.values())]
    n_cols = len(dataset_labels)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5), sharey=True)
    if n_cols == 1:
        axes = [axes]

    attr_names = ["Position", "Quats", "Scales", "Opacity", "SH DC", "SH Rest"]
    attr_keys = [
        "position_compressed_bytes", "quats_bytes", "scales_bytes",
        "opacity_bytes", "sh_dc_bytes", "sh_rest_bytes",
    ]
    attr_colors = ["#ff9800", "#e91e63", "#9c27b0", "#00bcd4", "#4caf50", "#2196f3"]

    for di, (ds_label, ax) in enumerate(zip(dataset_labels, axes)):
        agg = all_data[ds_label]
        x = np.arange(len(variants))
        bottom = np.zeros(len(variants))

        for attr_name, attr_key, attr_color in zip(attr_names, attr_keys, attr_colors):
            values_mb = [
                agg.get(v, {}).get(attr_key, 0) / (1024 * 1024)
                for v in variants
            ]
            ax.bar(
                x, values_mb, 0.6, bottom=bottom, color=attr_color, alpha=0.85,
                label=attr_name if di == 0 else None, edgecolor="white",
            )
            bottom += np.array(values_mb)

        # Total labels on top
        for i, v in enumerate(variants):
            total = agg.get(v, {}).get("compressed_size_bytes", 0) / (1024 * 1024)
            ax.annotate(
                f"{total:.3f}", xy=(i, bottom[i]),
                ha="center", va="bottom", fontsize=9,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([VARIANT_LABELS[v] for v in variants], fontsize=10)
        ax.set_title(ds_label.replace("\n", " "), fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(bottom=0)

    axes[0].set_ylabel("Avg Compressed Size (MB)", fontsize=11)
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="upper center",
        ncol=len(attr_names), fontsize=9, bbox_to_anchor=(0.5, 1.05),
    )
    fig.suptitle("KLT Ablation: Size Breakdown by Attribute", fontsize=13, y=1.10)
    fig.tight_layout()
    out = os.path.join(output_folder, f"klt_size_breakdown.{fmt}")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 3: Quality vs Size (if evaluation data available)
# ---------------------------------------------------------------------------
def plot_quality_vs_size(all_data, all_eval, dataset_labels, output_folder, fmt):
    """Scatter plot of PSNR vs compressed size per variant per dataset."""
    has_eval = any(
        ds in all_eval and any(all_eval[ds].get(v) for v in VARIANT_ORDER)
        for ds in dataset_labels
    )
    if not has_eval:
        print("  No evaluation data found, skipping quality vs size plot.")
        return

    variants = [v for v in VARIANT_ORDER if any(v in d for d in all_data.values())]
    n_datasets = len(dataset_labels)
    fig, axes = plt.subplots(1, n_datasets, figsize=(5 * n_datasets, 5))
    if n_datasets == 1:
        axes = [axes]

    for di, (ds_label, ax) in enumerate(zip(dataset_labels, axes)):
        agg = all_data[ds_label]
        eval_data = all_eval.get(ds_label, {})

        for variant in variants:
            size_mb = agg.get(variant, {}).get("compressed_size_bytes", 0) / (1024 * 1024)
            ev = eval_data.get(variant)
            if ev is None:
                continue
            psnr = ev.get("decomp_psnr", 0)
            color = VARIANT_COLORS[variant]
            ax.scatter(
                size_mb, psnr, color=color, s=120, zorder=5,
                label=VARIANT_LABELS[variant] if di == 0 else None,
                edgecolors="black", linewidths=0.5,
            )
            ax.annotate(
                f"{VARIANT_LABELS[variant]}\n{psnr:.2f} dB\n{size_mb:.3f} MB",
                xy=(size_mb, psnr), fontsize=8, ha="center", va="bottom",
                xytext=(0, 12), textcoords="offset points",
            )

        ax.set_xlabel("Compressed Size (MB)", fontsize=10)
        ax.set_ylabel("PSNR (dB)", fontsize=10)
        ax.set_title(ds_label.replace("\n", " "), fontsize=11)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.grid(True, alpha=0.3)

    if any(ax.has_data() for ax in axes):
        fig.legend(
            *axes[0].get_legend_handles_labels(), loc="upper center",
            ncol=len(variants), fontsize=10, bbox_to_anchor=(0.5, 1.02),
        )
    fig.suptitle("KLT Ablation: Quality vs Compressed Size", fontsize=13, y=1.06)
    fig.tight_layout()

    out = os.path.join(output_folder, f"klt_quality_vs_size.{fmt}")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 4: Color transform time overhead
# ---------------------------------------------------------------------------
def plot_transform_time(all_data, dataset_labels, output_folder, fmt):
    """Grouped bar chart: encode & decode time per variant, annotated with
    overhead vs RGB (the no-transform baseline)."""
    variants = [v for v in VARIANT_ORDER if any(v in d for d in all_data.values())]
    n_variants = len(variants)
    n_datasets = len(dataset_labels)
    bar_width = 0.8 / n_variants
    x = np.arange(n_datasets)

    fig, (ax_enc, ax_dec) = plt.subplots(1, 2, figsize=(max(10, n_datasets * 4), 5))

    for ax, time_key, title in [
        (ax_enc, "encode_time_ms", "Encode Time"),
        (ax_dec, "decode_time_ms", "Decode Time"),
    ]:
        for vi, variant in enumerate(variants):
            offset = (vi - n_variants / 2 + 0.5) * bar_width
            color = VARIANT_COLORS[variant]
            times = [
                all_data[ds].get(variant, {}).get(time_key, 0)
                for ds in dataset_labels
            ]
            bars = ax.bar(
                x + offset, times, bar_width, color=color, alpha=0.85,
                edgecolor="white", label=VARIANT_LABELS[variant],
            )
            for bar, val in zip(bars, times):
                ax.annotate(
                    f"{val:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=9,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(dataset_labels, fontsize=10)
        ax.set_ylabel("Time (ms)", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.suptitle(
        "KLT Ablation: Color Transform Time Overhead (vs RGB baseline)",
        fontsize=13,
    )
    fig.tight_layout()
    out = os.path.join(output_folder, f"klt_transform_time.{fmt}")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Text summary table
# ---------------------------------------------------------------------------
def print_summary_table(all_data, all_eval, dataset_labels):
    """Print a text summary table to stdout."""
    variants = [v for v in VARIANT_ORDER if any(v in d for d in all_data.values())]

    print("\n" + "=" * 90)
    print("SUMMARY TABLE: KLT Color Space Ablation")
    print("=" * 90)

    header = f"{'Sequence':<25s}"
    for v in variants:
        header += f"  {VARIANT_LABELS[v]:>12s}"
    print(header)
    print("-" * 90)

    for ds_label in dataset_labels:
        agg = all_data[ds_label]
        eval_data = all_eval.get(ds_label, {})
        ds_short = ds_label.replace("\n", " ")

        # Compressed size
        line = f"  {'Size (MB)':<23s}"
        for v in variants:
            size_mb = agg.get(v, {}).get("compressed_size_bytes", 0) / (1024 * 1024)
            line += f"  {size_mb:>12.4f}"
        print(ds_short)
        print(line)

        # Attribute size (SH only: DC + rest)
        line = f"  {'SH DC (MB)':<23s}"
        for v in variants:
            val = agg.get(v, {}).get("sh_dc_bytes", 0) / (1024 * 1024)
            line += f"  {val:>12.4f}"
        print(line)

        line = f"  {'SH Rest (MB)':<23s}"
        for v in variants:
            val = agg.get(v, {}).get("sh_rest_bytes", 0) / (1024 * 1024)
            line += f"  {val:>12.4f}"
        print(line)

        # PSNR / SSIM (if available)
        has_psnr = any(eval_data.get(v) for v in variants)
        if has_psnr:
            line = f"  {'PSNR (dB)':<23s}"
            for v in variants:
                ev = eval_data.get(v)
                if ev:
                    line += f"  {ev.get('decomp_psnr', 0):>12.4f}"
                else:
                    line += f"  {'N/A':>12s}"
            print(line)

            line = f"  {'SSIM':<23s}"
            for v in variants:
                ev = eval_data.get(v)
                if ev:
                    line += f"  {ev.get('decomp_ssim', 0):>12.6f}"
                else:
                    line += f"  {'N/A':>12s}"
            print(line)

        # Relative to KLT
        klt_size = agg.get("klt", {}).get("compressed_size_bytes", 1)
        line = f"  {'vs KLT (%)':<23s}"
        for v in variants:
            v_size = agg.get(v, {}).get("compressed_size_bytes", 0)
            diff_pct = (v_size - klt_size) / klt_size * 100
            line += f"  {diff_pct:>+12.2f}"
        print(line)
        print()

    print("=" * 90)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Plot KLT color space ablation results",
    )
    p.add_argument(
        "--output_folder", type=str,
        default=str(Path(__file__).resolve().parent / "plots"),
    )
    p.add_argument("--format", type=str, choices=["pdf", "png"], default="png")
    args = p.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    all_data = {}
    all_eval = {}
    dataset_labels = []

    for label, ablation_dir in DATASETS.items():
        csv_path = os.path.join(ablation_dir, "ablation_klt.csv")
        if not os.path.isfile(csv_path):
            print(f"  WARNING: {csv_path} not found, skipping {label}")
            continue

        rows = load_csv(csv_path)
        print(
            f"Loaded {len(rows)} rows for {label.replace(chr(10), ' ')} "
            f"from {csv_path}"
        )
        all_data[label] = aggregate(rows)
        dataset_labels.append(label)

        # Load evaluation results for each variant
        eval_per_variant = {}
        for variant in VARIANT_ORDER:
            eval_dir = os.path.join(ablation_dir, variant, "evaluation")
            ev = load_evaluation(eval_dir)
            if ev:
                eval_per_variant[variant] = ev
                print(
                    f"  Loaded evaluation for {variant}: "
                    f"PSNR={ev.get('decomp_psnr', 'N/A')}"
                )
        all_eval[label] = eval_per_variant

    if not dataset_labels:
        print("No data found!")
        return

    print(f"\nComparing {len(dataset_labels)} datasets")
    print_summary_table(all_data, all_eval, dataset_labels)
    plot_compressed_size(all_data, dataset_labels, args.output_folder, args.format)
    plot_size_breakdown(all_data, dataset_labels, args.output_folder, args.format)
    plot_quality_vs_size(all_data, all_eval, dataset_labels, args.output_folder, args.format)
    plot_transform_time(all_data, dataset_labels, args.output_folder, args.format)
    print("Done!")


if __name__ == "__main__":
    main()

"""
python scripts/livogs_baseline/ablation/plot_ablation_klt.py --format png
python scripts/livogs_baseline/ablation/plot_ablation_klt.py --format pdf
"""
