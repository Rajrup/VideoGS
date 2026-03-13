#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Box / whisker comparison plots: VideoGS vs LivoGS  and  MesonGS vs LivoGS.

Reads the CSV files produced by  pick_livogs_settings.py  and generates:
  * Per-sequence paired bar charts (PSNR & size) for each comparison.
  * Per-dataset summary box / whisker plots showing the *delta* across
    sequences (PSNR difference and rate ratio).

Usage:
    python scripts/plot_livogs_boxwhisker.py
"""

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
    import matplotlib.ticker as mticker
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "matplotlib and numpy are required. Install them (pip install matplotlib numpy)."
    ) from exc

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_CSV = SCRIPT_DIR / "rd_baselines_results" / "livogs_matched_settings_all.csv"
OUTPUT_DIR = SCRIPT_DIR / "rd_baselines_results" / "plots_livogs_comparison"

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

METHOD_COLORS: dict[str, str] = {
    "VideoGS": "#d62728",
    "MesonGS": "#2ca02c",
    "LivoGS-samerate": "#ff7f0e",
    "LivoGS-samepsnr": "#1f77b4",
}

METHOD_LABELS: dict[str, str] = {
    "VideoGS": "VideoGS",
    "MesonGS": "MesonGS",
    "LivoGS-samerate": "LivoGS\n(same rate)",
    "LivoGS-samepsnr": "LivoGS\n(same PSNR)",
}

SEQ_SHORT: dict[str, str] = {
    "4K_Actor1_Greeting": "Actor1",
    "4K_Actor2_Dancing": "Actor2",
    "4K_Actor3_Violin": "Actor3",
    "4K_Actor4_Dancing": "Actor4",
    "4K_Actor5_Oil-paper_Umbrella": "Actor5",
    "4K_Actor6_Changing_Clothes": "Actor6",
    "4K_Actor7_Nunchaku": "Actor7",
    "cook_spinach": "cook_sp",
    "coffee_martini": "coff_mar",
    "cut_roasted_beef": "cut_beef",
    "flame_salmon_1": "fl_salm",
    "flame_steak": "fl_steak",
    "sear_steak": "sear_st",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["compressed_mb"] = float(row["compressed_mb"])
            row["decomp_psnr"] = float(row["decomp_psnr"])
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Per-sequence grouped bar chart (PSNR + size side by side)
# ---------------------------------------------------------------------------

def plot_per_sequence_bars(
    rows: list[dict[str, Any]],
    comparison: str,
    output_dir: str,
) -> None:
    """One figure per sequence: grouped bars for PSNR and size."""
    # Group by (dataset, sequence)
    by_seq: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r["comparison"] == comparison:
            by_seq[(r["dataset"], r["sequence"])].append(r)

    baseline_name = "VideoGS" if comparison == "vs_VideoGS" else "MesonGS"
    method_order = [baseline_name, "LivoGS-samerate", "LivoGS-samepsnr"]

    for (dataset, sequence), seq_rows in sorted(by_seq.items()):
        by_method: dict[str, dict[str, Any]] = {}
        for r in seq_rows:
            by_method[r["method"]] = r

        methods_present = [m for m in method_order if m in by_method]
        if len(methods_present) < 2:
            continue

        psnrs = [by_method[m]["decomp_psnr"] for m in methods_present]
        sizes = [by_method[m]["compressed_mb"] for m in methods_present]
        colors = [METHOD_COLORS[m] for m in methods_present]
        labels = [METHOD_LABELS[m] for m in methods_present]

        fig, (ax_psnr, ax_size) = plt.subplots(1, 2, figsize=(10, 4.5))

        x = np.arange(len(methods_present))
        width = 0.5

        # PSNR bars
        bars_p = ax_psnr.bar(x, psnrs, width, color=colors, edgecolor="black",
                             linewidth=0.6)
        ax_psnr.set_ylabel("PSNR (dB)", fontsize=11)
        ax_psnr.set_xticks(x)
        ax_psnr.set_xticklabels(labels, fontsize=9)
        ax_psnr.set_title("Reconstruction Quality", fontsize=11)
        # Show values on bars
        for bar, val in zip(bars_p, psnrs):
            ax_psnr.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f"{val:.2f}", ha="center", va="bottom", fontsize=8)
        # Tighten y-axis around data
        psnr_min = min(psnrs)
        psnr_max = max(psnrs)
        psnr_range = max(psnr_max - psnr_min, 0.5)
        ax_psnr.set_ylim(psnr_min - psnr_range * 0.3, psnr_max + psnr_range * 0.5)
        ax_psnr.grid(axis="y", alpha=0.3)

        # Size bars
        bars_s = ax_size.bar(x, sizes, width, color=colors, edgecolor="black",
                             linewidth=0.6)
        ax_size.set_ylabel("Compressed Size (MB)", fontsize=11)
        ax_size.set_xticks(x)
        ax_size.set_xticklabels(labels, fontsize=9)
        ax_size.set_title("Compressed Size", fontsize=11)
        for bar, val in zip(bars_s, sizes):
            ax_size.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        ax_size.set_ylim(0, max(sizes) * 1.25)
        ax_size.grid(axis="y", alpha=0.3)

        seq_short = SEQ_SHORT.get(sequence, sequence)
        fig.suptitle(f"{dataset} | {sequence}\n{baseline_name} vs LivoGS",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.90])

        seq_dir = os.path.join(output_dir, dataset)
        os.makedirs(seq_dir, exist_ok=True)
        fname = f"bars_{comparison}_{sequence}.png"
        fig.savefig(os.path.join(seq_dir, fname), dpi=150)
        plt.close(fig)
        print(f"  Saved: {seq_dir}/{fname}")


# ---------------------------------------------------------------------------
# Per-dataset box / whisker  (delta-PSNR and rate-ratio across sequences)
# ---------------------------------------------------------------------------

def plot_dataset_boxwhisker(
    rows: list[dict[str, Any]],
    comparison: str,
    output_dir: str,
) -> None:
    """One figure per dataset: box/whisker across sequences for delta metrics."""
    baseline_name = "VideoGS" if comparison == "vs_VideoGS" else "MesonGS"

    # Group by (dataset, sequence)
    by_ds_seq: dict[str, dict[str, dict[str, Any]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for r in rows:
        if r["comparison"] == comparison:
            by_ds_seq[r["dataset"]][r["sequence"]][r["method"]] = r

    for dataset, seqs in sorted(by_ds_seq.items()):
        # Compute deltas for each LivoGS variant
        seq_names: list[str] = []
        # samerate: PSNR deltas and rate ratios
        sr_dpsnr: list[float] = []
        sr_rrate: list[float] = []
        # samepsnr: PSNR deltas and rate ratios
        sp_dpsnr: list[float] = []
        sp_rrate: list[float] = []

        for sequence in sorted(seqs.keys()):
            methods = seqs[sequence]
            if baseline_name not in methods:
                continue
            base = methods[baseline_name]
            seq_names.append(SEQ_SHORT.get(sequence, sequence))

            if "LivoGS-samerate" in methods:
                lr = methods["LivoGS-samerate"]
                sr_dpsnr.append(lr["decomp_psnr"] - base["decomp_psnr"])
                sr_rrate.append(lr["compressed_mb"] / base["compressed_mb"]
                                if base["compressed_mb"] > 0 else float("nan"))
            else:
                sr_dpsnr.append(float("nan"))
                sr_rrate.append(float("nan"))

            if "LivoGS-samepsnr" in methods:
                lp = methods["LivoGS-samepsnr"]
                sp_dpsnr.append(lp["decomp_psnr"] - base["decomp_psnr"])
                sp_rrate.append(lp["compressed_mb"] / base["compressed_mb"]
                                if base["compressed_mb"] > 0 else float("nan"))
            else:
                sp_dpsnr.append(float("nan"))
                sp_rrate.append(float("nan"))

        if not seq_names:
            continue

        n = len(seq_names)
        x = np.arange(n)

        # ---- Figure: PSNR delta per sequence + box/whisker summary ----
        fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                                 gridspec_kw={"width_ratios": [3, 1]})
        ax_bar, ax_box = axes

        width = 0.35
        ax_bar.bar(x - width / 2, sr_dpsnr, width,
                   color=METHOD_COLORS["LivoGS-samerate"], edgecolor="black",
                   linewidth=0.5, label="LivoGS (same rate)")
        ax_bar.bar(x + width / 2, sp_dpsnr, width,
                   color=METHOD_COLORS["LivoGS-samepsnr"], edgecolor="black",
                   linewidth=0.5, label="LivoGS (same PSNR)")
        ax_bar.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax_bar.set_ylabel(f"PSNR - {baseline_name} PSNR  (dB)", fontsize=10)
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(seq_names, rotation=30, ha="right", fontsize=9)
        ax_bar.legend(fontsize=9, loc="best")
        ax_bar.grid(axis="y", alpha=0.3)
        ax_bar.set_title("PSNR difference per sequence", fontsize=11)

        # Box/whisker summary
        bp_data = [
            [v for v in sr_dpsnr if not (v != v)],
            [v for v in sp_dpsnr if not (v != v)],
        ]
        bp = ax_box.boxplot(
            bp_data,
            tick_labels=["same\nrate", "same\nPSNR"],
            patch_artist=True,
            widths=0.5,
        )
        for patch, color in zip(bp["boxes"],
                                [METHOD_COLORS["LivoGS-samerate"],
                                 METHOD_COLORS["LivoGS-samepsnr"]]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax_box.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax_box.set_ylabel("ΔPSNR (dB)", fontsize=10)
        ax_box.set_title("Distribution", fontsize=11)
        ax_box.grid(axis="y", alpha=0.3)

        fig.suptitle(
            f"{dataset} | LivoGS vs {baseline_name} — PSNR difference",
            fontsize=12, fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        os.makedirs(output_dir, exist_ok=True)
        fname = f"boxwhisker_psnr_{comparison}_{dataset}.png"
        fig.savefig(os.path.join(output_dir, fname), dpi=150)
        plt.close(fig)
        print(f"  Saved: {output_dir}/{fname}")

        # ---- Figure: Rate ratio per sequence + box/whisker summary ----
        fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                                 gridspec_kw={"width_ratios": [3, 1]})
        ax_bar, ax_box = axes

        ax_bar.bar(x - width / 2, sr_rrate, width,
                   color=METHOD_COLORS["LivoGS-samerate"], edgecolor="black",
                   linewidth=0.5, label="LivoGS (same rate)")
        ax_bar.bar(x + width / 2, sp_rrate, width,
                   color=METHOD_COLORS["LivoGS-samepsnr"], edgecolor="black",
                   linewidth=0.5, label="LivoGS (same PSNR)")
        ax_bar.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
        ax_bar.set_ylabel(f"LivoGS rate / {baseline_name} rate", fontsize=10)
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(seq_names, rotation=30, ha="right", fontsize=9)
        ax_bar.legend(fontsize=9, loc="best")
        ax_bar.grid(axis="y", alpha=0.3)
        ax_bar.set_title("Rate ratio per sequence", fontsize=11)

        bp_data = [
            [v for v in sr_rrate if not (v != v)],
            [v for v in sp_rrate if not (v != v)],
        ]
        bp = ax_box.boxplot(
            bp_data,
            tick_labels=["same\nrate", "same\nPSNR"],
            patch_artist=True,
            widths=0.5,
        )
        for patch, color in zip(bp["boxes"],
                                [METHOD_COLORS["LivoGS-samerate"],
                                 METHOD_COLORS["LivoGS-samepsnr"]]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax_box.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
        ax_box.set_ylabel("Rate ratio", fontsize=10)
        ax_box.set_title("Distribution", fontsize=11)
        ax_box.grid(axis="y", alpha=0.3)

        fig.suptitle(
            f"{dataset} | LivoGS vs {baseline_name} — rate ratio",
            fontsize=12, fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fname = f"boxwhisker_rate_{comparison}_{dataset}.png"
        fig.savefig(os.path.join(output_dir, fname), dpi=150)
        plt.close(fig)
        print(f"  Saved: {output_dir}/{fname}")


# ---------------------------------------------------------------------------
# Per-dataset "absolute values" box / whisker (PSNR and size across seqs)
# ---------------------------------------------------------------------------

def plot_absolute_boxwhisker(
    rows: list[dict[str, Any]],
    comparison: str,
    output_dir: str,
) -> None:
    """Box/whisker of absolute PSNR and size values across sequences."""
    baseline_name = "VideoGS" if comparison == "vs_VideoGS" else "MesonGS"
    method_order = [baseline_name, "LivoGS-samerate", "LivoGS-samepsnr"]

    by_ds: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    by_ds_size: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rows:
        if r["comparison"] != comparison:
            continue
        by_ds[r["dataset"]][r["method"]].append(r["decomp_psnr"])
        by_ds_size[r["dataset"]][r["method"]].append(r["compressed_mb"])

    for dataset in sorted(by_ds.keys()):
        methods_with_data = [m for m in method_order if by_ds[dataset].get(m)]
        if len(methods_with_data) < 2:
            continue

        fig, (ax_psnr, ax_size) = plt.subplots(1, 2, figsize=(12, 5))

        # PSNR box/whisker
        psnr_data = [by_ds[dataset][m] for m in methods_with_data]
        bp = ax_psnr.boxplot(
            psnr_data,
            tick_labels=[METHOD_LABELS[m] for m in methods_with_data],
            patch_artist=True,
            widths=0.5,
        )
        for patch, m in zip(bp["boxes"], methods_with_data):
            patch.set_facecolor(METHOD_COLORS[m])
            patch.set_alpha(0.6)
        # Overlay individual data points
        for i, (m, vals) in enumerate(zip(methods_with_data, psnr_data), 1):
            jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(vals))
            ax_psnr.scatter(i + jitter, vals, color=METHOD_COLORS[m],
                            s=20, alpha=0.8, zorder=5, edgecolors="black",
                            linewidths=0.3)
        ax_psnr.set_ylabel("PSNR (dB)", fontsize=11)
        ax_psnr.set_title("PSNR across sequences", fontsize=11)
        ax_psnr.grid(axis="y", alpha=0.3)

        # Size box/whisker
        size_data = [by_ds_size[dataset][m] for m in methods_with_data]
        bp = ax_size.boxplot(
            size_data,
            tick_labels=[METHOD_LABELS[m] for m in methods_with_data],
            patch_artist=True,
            widths=0.5,
        )
        for patch, m in zip(bp["boxes"], methods_with_data):
            patch.set_facecolor(METHOD_COLORS[m])
            patch.set_alpha(0.6)
        for i, (m, vals) in enumerate(zip(methods_with_data, size_data), 1):
            jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(vals))
            ax_size.scatter(i + jitter, vals, color=METHOD_COLORS[m],
                            s=20, alpha=0.8, zorder=5, edgecolors="black",
                            linewidths=0.3)
        ax_size.set_ylabel("Compressed Size (MB)", fontsize=11)
        ax_size.set_title("Compressed Size across sequences", fontsize=11)
        ax_size.grid(axis="y", alpha=0.3)

        fig.suptitle(
            f"{dataset} | {baseline_name} vs LivoGS — distribution across sequences",
            fontsize=12, fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        os.makedirs(output_dir, exist_ok=True)
        fname = f"boxwhisker_absolute_{comparison}_{dataset}.png"
        fig.savefig(os.path.join(output_dir, fname), dpi=150)
        plt.close(fig)
        print(f"  Saved: {output_dir}/{fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(
            f"Input CSV not found: {INPUT_CSV}\n"
            "Run  pick_livogs_settings.py  first."
        )

    output_dir = str(OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    rows = load_csv(INPUT_CSV)
    print(f"Loaded {len(rows)} rows from {INPUT_CSV}\n")

    for comparison in ("vs_VideoGS", "vs_MesonGS"):
        baseline = "VideoGS" if comparison == "vs_VideoGS" else "MesonGS"
        print(f"\n{'='*50}")
        print(f"  {baseline} comparison")
        print(f"{'='*50}")

        # 1) Per-sequence bar charts
        print("\n--- Per-sequence bar charts ---")
        plot_per_sequence_bars(rows, comparison, output_dir)

        # 2) Per-dataset delta box/whisker
        print("\n--- Per-dataset delta box/whisker ---")
        plot_dataset_boxwhisker(rows, comparison, output_dir)

        # 3) Per-dataset absolute box/whisker
        print("\n--- Per-dataset absolute box/whisker ---")
        plot_absolute_boxwhisker(rows, comparison, output_dir)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
