#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Collect baseline outputs and generate per-frame comparison plots for VideoGS."""

from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

try:
    import matplotlib  # type: ignore[reportMissingImports]

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required for plotting. Install it in your environment "
        "(for example: pip install matplotlib)."
    ) from exc
import numpy as np


DATASET_NAME = "HiFi4G_Dataset"
DATA_PATH = "/synology/rajrup/VideoGS"
VIDEOGS_QPS = [25]
VIDEOGS_GROUP_SIZE = 20

DRACOGS_DEFAULT_CONFIG_TAG = "eg_16_eo_16_et_16_es_16_cl_10"
MESONGS_DEFAULT_CONFIG_TAG = "d12_nb8_nblk57_cb2048"

GPCC_DEFAULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpcc_defaults.json")

EXPERIMENTS: dict[str, list[int]] = {
    "4K_Actor1_Greeting": list(range(1, 201, 20)),
    "4K_Actor2_Dancing": list(range(1, 201, 20)),
    "4K_Actor3_Violin": list(range(1, 201, 20)),
    "4K_Actor4_Dancing": list(range(1, 201, 20)),
    "4K_Actor5_Oil-paper_Umbrella": list(range(1, 201, 20)),
    "4K_Actor6_Changing_Clothes": list(range(1, 201, 20)),
    "4K_Actor7_Nunchaku": list(range(1, 201, 20)),
}

BASELINES: dict[str, dict[str, Any]] = {
    "DracoGS": {
        "subdir": "dracogs",
        "output_tag": DRACOGS_DEFAULT_CONFIG_TAG,
        "benchmark_csv": "benchmark_dracogs.csv",
    },
    "MesonGS": {
        "subdir": "mesongs",
        "output_tag": MESONGS_DEFAULT_CONFIG_TAG,
        "benchmark_csv": "benchmark_mesongs.csv",
    },
    "VideoGS": {
        "subdir": "videogs",
        "output_tags": [f"qp_{qp}" for qp in VIDEOGS_QPS],
        "benchmark_csv": "benchmark_videogs_pipeline.csv",
    },
}

PLOT_YLIM: dict[str, tuple[float, float]] = {
    "PSNR": (30, 40),
    "SSIM": (0.95, 1.0),
}

BASELINE_STYLES: dict[str, dict[str, Any]] = {
    "DracoGS": {"color": "#1f77b4", "marker": "o", "label": "LTS-Draco"},
    "MesonGS": {"color": "#2ca02c", "marker": "s", "label": "MesonGS"},
    "VideoGS": {"color": "#d62728", "marker": "^", "label": "V3-2D"},
    "GPCC": {"color": "#ff7f0e", "marker": "D", "label": "GPCC"},
}

# Internal key → user-facing display name
DISPLAY_NAMES: dict[str, str] = {
    "DracoGS": "LTS-Draco",
    "VideoGS": "V3-2D",
    "Ours": "GS-NFS (Ours)",
    "LiVoGS": "GS-NFS (Ours)",
}


def _display_name(name: str) -> str:
    return DISPLAY_NAMES.get(name, name)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "baseline_comparison_res")


LARGE_CSV_COLUMNS = [
    "sequence_name",
    "baseline",
    "frame_id",
    "gt_psnr",
    "gt_ssim",
    "decomp_psnr",
    "decomp_ssim",
    "size",
]


def _group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(row)
    return groups


def _model_root(sequence: str) -> str:
    return os.path.join(
        DATA_PATH,
        "train_output",
        DATASET_NAME,
        sequence,
    )


def _selected_to_span(frame_ids: list[int]) -> tuple[int, int, int]:
    if not frame_ids:
        raise ValueError("Frame list must not be empty")
    sorted_ids = sorted(set(int(v) for v in frame_ids))
    return sorted_ids[0], sorted_ids[-1] + 1, 1


@lru_cache(maxsize=None)
def _sequence_max_frame(sequence: str) -> int:
    checkpoint_root = os.path.join(_model_root(sequence), "checkpoint")
    if not os.path.isdir(checkpoint_root):
        raise FileNotFoundError(f"Checkpoint root not found: {checkpoint_root}")

    frame_ids = sorted(
        int(name)
        for name in os.listdir(checkpoint_root)
        if name.isdigit()
        and os.path.isdir(os.path.join(checkpoint_root, name))
        and os.path.isdir(os.path.join(checkpoint_root, name, "point_cloud"))
    )
    if not frame_ids:
        raise FileNotFoundError(
            f"No frame folders with point_cloud found under {checkpoint_root}"
        )

    return frame_ids[-1]


def _videogs_gop_frame_ids(sequence: str, anchor: int) -> list[int]:
    max_frame = _sequence_max_frame(sequence)
    if anchor > max_frame:
        raise ValueError(
            f"VideoGS anchor frame {anchor} exceeds last available frame {max_frame} for sequence {sequence}"
        )

    gop_end = min(int(anchor) + VIDEOGS_GROUP_SIZE, max_frame + 1)
    frame_ids = list(range(int(anchor), gop_end))
    if not frame_ids:
        raise ValueError(
            f"Resolved empty VideoGS GOP for sequence {sequence}: anchor={anchor}, end={gop_end}"
        )
    return frame_ids


def _frame_span_tag(frame_start: int, frame_end: int, interval: int) -> str:
    return f"frames_{frame_start}_{frame_end - 1}_int_{interval}"


def _frame_output_tag(frame_id: int) -> str:
    return f"frame{int(frame_id)}"


def _candidate_output_folders(
    sequence: str,
    subdir: str,
    output_tag: str,
    frame_start: int,
    frame_end: int,
    interval: int,
    frame_id: int | None = None,
) -> list[str]:
    legacy_root = os.path.join(_model_root(sequence), "compression", subdir, output_tag)
    candidates: list[str] = []
    if frame_id is not None:
        candidates.extend(
            [
                os.path.join(legacy_root, _frame_output_tag(frame_id)),
                os.path.join(legacy_root, "frame_results"),
            ]
        )
    candidates.extend(
        [
            os.path.join(legacy_root, _frame_span_tag(frame_start, frame_end, interval)),
            legacy_root,
        ]
    )
    return candidates


def _folder_has_frame_data(folder: str, benchmark_csv_name: str, frame_id: int) -> bool:
    benchmark_path = os.path.join(folder, benchmark_csv_name)
    eval_json_path = os.path.join(folder, "evaluation", "evaluation_results.json")

    has_benchmark = False
    if os.path.isfile(benchmark_path):
        try:
            with open(benchmark_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if int(row.get("frame_id", -1)) == frame_id:
                        has_benchmark = True
                        break
        except (OSError, ValueError):
            pass

    has_eval = False
    if os.path.isfile(eval_json_path):
        try:
            with open(eval_json_path, encoding="utf-8") as f:
                eval_data = json.load(f)
            for fr in eval_data.get("per_frame", []):
                if int(fr.get("frame", -1)) == frame_id:
                    has_eval = True
                    break
        except (OSError, json.JSONDecodeError, ValueError):
            pass

    return has_benchmark or has_eval


def _resolve_output_folder(
    sequence: str,
    subdir: str,
    output_tag: str,
    frame_start: int,
    frame_end: int,
    interval: int,
    benchmark_csv_name: str,
    frame_id: int | None = None,
) -> str:
    candidate_folders = _candidate_output_folders(
        sequence,
        subdir,
        output_tag,
        frame_start,
        frame_end,
        interval,
        frame_id=frame_id,
    )
    for folder in candidate_folders:
        if frame_id is not None:
            if _folder_has_frame_data(folder, benchmark_csv_name, frame_id):
                return folder
        else:
            benchmark_path = os.path.join(folder, benchmark_csv_name)
            eval_json_path = os.path.join(folder, "evaluation", "evaluation_results.json")
            if os.path.isfile(benchmark_path) or os.path.isfile(eval_json_path):
                return folder

    return candidate_folders[0]


def _load_gpcc_defaults_for_plotting() -> dict[str, dict[str, Any]]:
    """Load GPCC per-sequence default params. Returns empty dict if file missing."""
    if not os.path.isfile(GPCC_DEFAULTS_FILE):
        print(f"  [INFO] GPCC defaults file not found: {GPCC_DEFAULTS_FILE} — skipping GPCC in bar charts")
        return {}
    try:
        with open(GPCC_DEFAULTS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  [WARN] Failed to load GPCC defaults: {exc}")
        return {}


def _load_gpcc_frame_result(
    output_folder: str,
    sequence: str,
    frame_id: int,
) -> dict[str, Any] | None:
    # Separate from _load_sequence_results: GPCC CSVs use frame_idx / total_compressed_bytes columns
    bench_path = os.path.join(output_folder, "benchmark_gpcc.csv")
    eval_path = os.path.join(output_folder, "evaluation", "evaluation_results.json")

    compressed_bytes: int | None = None
    if os.path.isfile(bench_path):
        try:
            with open(bench_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if int(row["frame_idx"]) == frame_id:
                        compressed_bytes = int(row["total_compressed_bytes"])
                        break
        except (OSError, KeyError, ValueError) as exc:
            print(f"  [WARN] GPCC benchmark: {exc}")

    if compressed_bytes is None:
        return None

    metrics: dict[str, float] | None = None
    if os.path.isfile(eval_path):
        try:
            with open(eval_path, encoding="utf-8") as f:
                data = json.load(f)
            for fr in data.get("per_frame", []):
                if int(fr["frame"]) == frame_id:
                    metrics = {
                        "gt_psnr": float(fr["gt_psnr"]),
                        "gt_ssim": float(fr["gt_ssim"]),
                        "decomp_psnr": float(fr["decomp_psnr"]),
                        "decomp_ssim": float(fr["decomp_ssim"]),
                    }
                    break
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
            print(f"  [WARN] GPCC evaluation: {exc}")

    if metrics is None:
        return None

    return {
        "sequence_name": sequence,
        "baseline": "GPCC",
        "baseline_family": "GPCC",
        "videogs_qp": None,
        "frame_id": frame_id,
        "gop_anchor_frame": frame_id,
        "compressed_size_bytes": compressed_bytes,
        "compressed_mb": compressed_bytes / (1024 * 1024),
        "uncompressed_size_bytes": 0,
        "uncompressed_mb": 0.0,
        "gt_psnr": metrics["gt_psnr"],
        "gt_ssim": metrics["gt_ssim"],
        "decomp_psnr": metrics["decomp_psnr"],
        "decomp_ssim": metrics["decomp_ssim"],
    }


def _load_sequence_results(
    output_folder: str,
    sequence: str,
    baseline: str,
    baseline_family: str,
    videogs_qp: int | None,
    benchmark_csv_name: str,
    frame_ids: list[int],
    gop_anchor_frame: int | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

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
                    }
        except (OSError, KeyError, ValueError) as exc:
            print(f"  [WARN] Failed to read {benchmark_path}: {exc}")
    else:
        print(f"  [WARN] Benchmark CSV not found: {benchmark_path}")

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
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
            print(f"  [WARN] Failed to read {eval_json_path}: {exc}")
    else:
        print(f"  [WARN] Evaluation JSON not found: {eval_json_path}")

    available_frame_ids = sorted(set(benchmark_by_frame.keys()) & set(metrics_by_frame.keys()))
    selected_frame_ids = list(frame_ids)
    unavailable_frame_ids = sorted(set(selected_frame_ids) - set(available_frame_ids))
    if unavailable_frame_ids:
        print(
            f"  [INFO] {baseline} | {sequence}: unavailable requested frames {unavailable_frame_ids}; "
            f"available frames {available_frame_ids}"
        )
    if (
        selected_frame_ids
        and available_frame_ids
        and not any(fid in available_frame_ids for fid in selected_frame_ids)
    ):
        print(
            f"  [INFO] {baseline} | {sequence}: requested frames {selected_frame_ids} not present in "
            f"both benchmark/eval; using available frames {available_frame_ids}"
        )
        selected_frame_ids = available_frame_ids

    for fid in selected_frame_ids:
        if fid not in benchmark_by_frame:
            print(f"  [SKIP] {baseline} | {sequence} | frame {fid} (no benchmark data)")
            continue
        if fid not in metrics_by_frame:
            print(f"  [SKIP] {baseline} | {sequence} | frame {fid} (no evaluation data)")
            continue

        b = benchmark_by_frame[fid]
        m = metrics_by_frame[fid]
        comp = b["compressed_size_bytes"]
        uncomp = b["uncompressed_size_bytes"]
        rows.append(
            {
                "sequence_name": sequence,
                "baseline": baseline,
                "baseline_family": baseline_family,
                "videogs_qp": videogs_qp,
                "frame_id": fid,
                "gop_anchor_frame": int(gop_anchor_frame) if gop_anchor_frame is not None else fid,
                "compressed_size_bytes": comp,
                "compressed_mb": comp / (1024 * 1024),
                "uncompressed_size_bytes": uncomp,
                "uncompressed_mb": uncomp / (1024 * 1024),
                "gt_psnr": m["gt_psnr"],
                "gt_ssim": m["gt_ssim"],
                "decomp_psnr": m["decomp_psnr"],
                "decomp_ssim": m["decomp_ssim"],
            }
        )

    return rows


def _baseline_sort_key(rows_for_baseline: list[dict[str, Any]]) -> tuple[int, float, str]:
    sample = rows_for_baseline[0]
    family = str(sample.get("baseline_family", sample.get("baseline", "")))
    label = str(sample.get("baseline", family))
    family_rank = {"DracoGS": 0, "LTS-Draco": 0, "MesonGS": 1, "VideoGS": 2, "V3-2D": 2, "GPCC": 3}.get(family, 99)
    qp = sample.get("videogs_qp")
    qp_sort = float(qp) if isinstance(qp, (int, float)) else -1.0
    return family_rank, qp_sort, label


def _style_for_baseline(rows_for_baseline: list[dict[str, Any]]) -> dict[str, Any]:
    sample = rows_for_baseline[0]
    family = str(sample.get("baseline_family", sample.get("baseline", "")))
    label = str(sample.get("baseline", family))

    base_style = BASELINE_STYLES.get(
        family,
        {"color": "#7f7f7f", "marker": "o", "label": label},
    )
    style = {
        "color": base_style["color"],
        "marker": base_style["marker"],
        "label": label,
    }

    qp = sample.get("videogs_qp")
    if family == "VideoGS" and isinstance(qp, int):
        qp_palette = {
            0: "#1b9e77",
            4: "#d95f02",
            10: "#7570b3",
            15: "#e7298a",
            20: "#66a61e",
        }
        qp_markers = {
            0: "o",
            4: "s",
            10: "D",
            15: "^",
            20: "v",
        }
        style["color"] = qp_palette.get(qp, style["color"])
        style["marker"] = qp_markers.get(qp, "o")

    return style


def collect_all_results() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sequence, frame_ids in EXPERIMENTS.items():
        frame_start, frame_end, interval = _selected_to_span(frame_ids)
        for baseline_family, cfg in BASELINES.items():
            output_tags = cfg.get("output_tags", [cfg.get("output_tag")])

            if baseline_family == "VideoGS":
                for output_tag in output_tags:
                    if not output_tag:
                        continue

                    videogs_qp: int | None = None
                    display = _display_name(baseline_family)
                    baseline_label = display
                    if str(output_tag).startswith("qp_"):
                        qp_suffix = str(output_tag).split("_", maxsplit=1)[1]
                        try:
                            videogs_qp = int(qp_suffix)
                        except ValueError:
                            videogs_qp = None
                        baseline_label = (
                            f"{display} (QP={videogs_qp})"
                            if videogs_qp is not None
                            else f"{display} ({output_tag})"
                        )

                    for anchor in frame_ids:
                        gop_frame_ids = _videogs_gop_frame_ids(sequence, int(anchor))
                        output_folder = _resolve_output_folder(
                            sequence,
                            cfg["subdir"],
                            str(output_tag),
                            gop_frame_ids[0],
                            gop_frame_ids[-1] + 1,
                            1,
                            cfg["benchmark_csv"],
                        )
                        rows.extend(
                            _load_sequence_results(
                                output_folder,
                                sequence,
                                baseline_label,
                                baseline_family,
                                videogs_qp,
                                cfg["benchmark_csv"],
                                gop_frame_ids,
                                gop_anchor_frame=int(anchor),
                            )
                        )
                continue

            default_tag = cfg.get("output_tag")
            if not default_tag:
                continue

            for fid in frame_ids:
                output_folder = _resolve_output_folder(
                    sequence,
                    cfg["subdir"],
                    str(default_tag),
                    frame_start,
                    frame_end,
                    interval,
                    cfg["benchmark_csv"],
                    frame_id=int(fid),
                )
                rows.extend(
                    _load_sequence_results(
                        output_folder,
                        sequence,
                        _display_name(baseline_family),
                        baseline_family,
                        None,
                        cfg["benchmark_csv"],
                        [int(fid)],
                    )
                )

    gpcc_defaults = _load_gpcc_defaults_for_plotting()
    for sequence, frame_ids in EXPERIMENTS.items():
        if sequence not in gpcc_defaults:
            continue
        p = gpcc_defaults[sequence]
        params_tag = f"J{p['voxel_depth']}_rest{p['qp_rest']}_dc{p['qp_dc']}_op{p['qp_opacity']}"
        for fid in frame_ids:
            output_folder = os.path.join(
                _model_root(sequence), "compression", "gpcc", params_tag, f"frame{fid}"
            )
            result = _load_gpcc_frame_result(output_folder, sequence, int(fid))
            if result is not None:
                rows.append(result)
            else:
                print(f"  [SKIP] GPCC | {sequence} | frame {fid} (no data)")

    return rows


def write_large_csv(rows: list[dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LARGE_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sequence_name": row.get("sequence_name", ""),
                    "baseline": row.get("baseline", ""),
                    "frame_id": row.get("frame_id", ""),
                    "gt_psnr": row.get("gt_psnr", ""),
                    "gt_ssim": row.get("gt_ssim", ""),
                    "decomp_psnr": row.get("decomp_psnr", ""),
                    "decomp_ssim": row.get("decomp_ssim", ""),
                    "size": row.get("compressed_size_bytes", ""),
                }
            )
    print(f"  Wrote {len(rows)} rows to: {path}")


def _dedup_by_frame(
    seq_rows: list[dict[str, Any]], key: str,
) -> list[float]:
    """Extract one value per unique frame_id, skipping None and zero."""
    seen: set[int] = set()
    vals: list[float] = []
    for r in seq_rows:
        fid = int(r["frame_id"])
        if fid in seen:
            continue
        v = r.get(key)
        if v is None or v == 0:
            continue
        seen.add(fid)
        vals.append(float(v))
    return vals


def _get_ordered_baselines(
    rows: list[dict[str, Any]],
) -> list[str]:
    by_bl = _group_by(rows, "baseline")
    return sorted(by_bl.keys(), key=lambda b: _baseline_sort_key(by_bl[b]))


def _bar_colors(
    baseline_labels: list[str],
    by_baseline: dict[str, list[dict[str, Any]]],
) -> list[str]:
    colors = ["#999999"]  # uncompressed
    for bl in baseline_labels:
        style = _style_for_baseline(by_baseline[bl])
        colors.append(style["color"])
    return colors


def plot_size_by_sequence(rows: list[dict[str, Any]], plot_dir: str) -> None:
    """Grouped bar chart — size (MB, log₁₀) per sequence + average."""
    sequences = list(dict.fromkeys(r["sequence_name"] for r in rows))
    baseline_labels = _get_ordered_baselines(rows)
    by_seq = _group_by(rows, "sequence_name")
    by_baseline = _group_by(rows, "baseline")

    bar_labels = ["Uncompressed"] + list(baseline_labels)
    x_labels = sequences + ["Average"]
    n_groups = len(x_labels)
    n_bars = len(bar_labels)

    means = np.zeros((n_bars, n_groups))
    stds = np.zeros((n_bars, n_groups))

    for j, seq in enumerate(sequences):
        seq_rows = by_seq.get(seq, [])
        # Uncompressed — deduplicate across baselines for the same frame
        vals = _dedup_by_frame(seq_rows, "uncompressed_mb")
        if vals:
            means[0, j] = float(np.mean(vals))
            stds[0, j] = float(np.std(vals))
        # Per-baseline compressed
        seq_by_bl = _group_by(seq_rows, "baseline")
        for i, bl in enumerate(baseline_labels):
            vals = [
                float(r["compressed_mb"])
                for r in seq_by_bl.get(bl, [])
                if r.get("compressed_mb") is not None
            ]
            if vals:
                means[i + 1, j] = float(np.mean(vals))
                stds[i + 1, j] = float(np.std(vals))

    # Average column — micro-average across all frames
    vals = _dedup_by_frame(rows, "uncompressed_mb")
    if vals:
        means[0, -1] = float(np.mean(vals))
        stds[0, -1] = float(np.std(vals))
    for i, bl in enumerate(baseline_labels):
        vals = [
            float(r["compressed_mb"])
            for r in by_baseline.get(bl, [])
            if r.get("compressed_mb") is not None
        ]
        if vals:
            means[i + 1, -1] = float(np.mean(vals))
            stds[i + 1, -1] = float(np.std(vals))

    # --- draw ---
    colors = _bar_colors(baseline_labels, by_baseline)
    fig, ax = plt.subplots(figsize=(max(14, n_groups * 2.2), 7))
    x = np.arange(n_groups, dtype=float)
    width = 0.8 / n_bars

    for i in range(n_bars):
        offset = (i - n_bars / 2 + 0.5) * width
        ax.bar(
            x + offset,
            means[i],
            width,
            yerr=stds[i],
            label=bar_labels[i],
            color=colors[i],
            edgecolor="black",
            linewidth=0.5,
            capsize=3,
            zorder=3,
        )

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Size (MB, log scale)", fontsize=11)
    ax.set_title("Compressed Size by Sequence", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    fig.tight_layout()

    out_path = os.path.join(plot_dir, "size_by_sequence.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_quality_by_sequence(
    rows: list[dict[str, Any]],
    plot_dir: str,
    decomp_key: str,
    gt_key: str,
    ylabel: str,
    title: str,
    filename: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Grouped bar chart — quality metric per sequence + average."""
    sequences = list(dict.fromkeys(r["sequence_name"] for r in rows))
    baseline_labels = _get_ordered_baselines(rows)
    by_seq = _group_by(rows, "sequence_name")
    by_baseline = _group_by(rows, "baseline")

    bar_labels = ["Uncompressed"] + list(baseline_labels)
    x_labels = sequences + ["Average"]
    n_groups = len(x_labels)
    n_bars = len(bar_labels)

    means = np.zeros((n_bars, n_groups))
    stds = np.zeros((n_bars, n_groups))

    for j, seq in enumerate(sequences):
        seq_rows = by_seq.get(seq, [])
        # Uncompressed quality — deduplicate across baselines
        vals = _dedup_by_frame(seq_rows, gt_key)
        if vals:
            means[0, j] = float(np.mean(vals))
            stds[0, j] = float(np.std(vals))
        # Per-baseline compressed quality
        seq_by_bl = _group_by(seq_rows, "baseline")
        for i, bl in enumerate(baseline_labels):
            vals = [
                float(r[decomp_key])
                for r in seq_by_bl.get(bl, [])
                if r.get(decomp_key) is not None
            ]
            if vals:
                means[i + 1, j] = float(np.mean(vals))
                stds[i + 1, j] = float(np.std(vals))

    # Average column
    vals = _dedup_by_frame(rows, gt_key)
    if vals:
        means[0, -1] = float(np.mean(vals))
        stds[0, -1] = float(np.std(vals))
    for i, bl in enumerate(baseline_labels):
        vals = [
            float(r[decomp_key])
            for r in by_baseline.get(bl, [])
            if r.get(decomp_key) is not None
        ]
        if vals:
            means[i + 1, -1] = float(np.mean(vals))
            stds[i + 1, -1] = float(np.std(vals))

    # --- draw ---
    colors = _bar_colors(baseline_labels, by_baseline)
    fig, ax = plt.subplots(figsize=(max(14, n_groups * 2.2), 7))
    x = np.arange(n_groups, dtype=float)
    width = 0.8 / n_bars

    for i in range(n_bars):
        offset = (i - n_bars / 2 + 0.5) * width
        ax.bar(
            x + offset,
            means[i],
            width,
            yerr=stds[i],
            label=bar_labels[i],
            color=colors[i],
            edgecolor="black",
            linewidth=0.5,
            capsize=3,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=9)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    fig.tight_layout()

    out_path = os.path.join(plot_dir, filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# =========================================================================
# COMPARISON TABLE — same-PSNR matching across baselines
# =========================================================================

N3DV_SEQUENCES: list[str] = [
    "cook_spinach",
    "coffee_martini",
    "cut_roasted_beef",
    "flame_salmon_1",
    "flame_steak",
    "sear_steak",
]

LIVOGS_RD_SUBDIR = "livogs_rd_new"

GPCC_QP_COMBOS: list[tuple[int, int, int]] = [
    # (qp_rest, qp_dc, qp_opacity)
    (40, 4, 16), (40, 4, 34), (40, 4, 40),
    (40, 16, 16), (40, 16, 34), (40, 16, 40),
    (40, 20, 16), (40, 20, 34), (40, 20, 40),
    (40, 24, 16), (40, 24, 34), (40, 24, 40),
    (40, 28, 16), (40, 28, 34), (40, 28, 40),
    (38, 4, 4), (38, 16, 4),
    (34, 4, 4), (34, 16, 4),
    (31, 4, 4), (31, 16, 4),
    (28, 4, 4), (28, 16, 4),
    (38, 4, 16), (38, 16, 16),
    (34, 4, 16), (34, 16, 16),
    (31, 4, 16), (31, 16, 16),
    (28, 4, 16), (28, 16, 16),
    (38, 4, 28), (38, 16, 28),
    (34, 4, 28), (34, 16, 28),
    (31, 4, 28), (31, 16, 28),
    (28, 4, 28), (28, 16, 28),
    (16, 4, 4), (16, 16, 4),
    (4, 4, 4), (4, 16, 4),
    (16, 4, 16), (4, 4, 16),
]


@dataclass(frozen=True)
class TableSeqCfg:
    """Configuration for one sequence in the comparison table."""
    dataset: str
    sequence: str
    model_root: str
    frame_id: int
    group_size: int
    mesongs_tag: str
    gpcc_octree_depths: tuple[int, ...]


def _build_table_cfgs() -> list[TableSeqCfg]:
    cfgs: list[TableSeqCfg] = []
    for seq in EXPERIMENTS:
        cfgs.append(TableSeqCfg(
            dataset="HiFi4G",
            sequence=seq,
            model_root=f"{DATA_PATH}/train_output/{DATASET_NAME}/{seq}",
            frame_id=0,
            group_size=VIDEOGS_GROUP_SIZE,
            mesongs_tag=MESONGS_DEFAULT_CONFIG_TAG,
            gpcc_octree_depths=(8, 9, 10, 11, 12),
        ))
    for seq in N3DV_SEQUENCES:
        cfgs.append(TableSeqCfg(
            dataset="N3DV",
            sequence=seq,
            model_root=f"/synology/rajrup/Queen/pretrained_output/Neural_3D_Video/queen_compressed_{seq}",
            frame_id=1,
            group_size=VIDEOGS_GROUP_SIZE,
            mesongs_tag="d17_nb8_nblk57_cb2048",
            gpcc_octree_depths=(12, 13, 14, 15, 16, 17),
        ))
    return cfgs


@dataclass
class MethodEntry:
    """One method's data for a single sequence in the comparison table."""
    encode_ms: float | None = None
    decode_ms: float | None = None
    psnr_diff: float | None = None       # decomp_psnr - gt_psnr
    compression_ratio: float | None = None  # uncompressed / compressed
    compressed_mb: float | None = None
    decomp_psnr: float | None = None


def _load_eval_metrics(eval_json_path: str, frame_id: int) -> dict[str, float] | None:
    """Load gt_psnr and decomp_psnr from evaluation JSON for a frame."""
    if not os.path.isfile(eval_json_path):
        return None
    try:
        with open(eval_json_path, encoding="utf-8") as f:
            d = json.load(f)
        for fr in d.get("per_frame", []):
            if int(fr["frame"]) == frame_id:
                return {
                    "gt_psnr": float(fr["gt_psnr"]),
                    "decomp_psnr": float(fr["decomp_psnr"]),
                }
    except (OSError, json.JSONDecodeError, KeyError, ValueError):
        pass
    return None


def _load_videogs_entry(cfg: TableSeqCfg) -> MethodEntry:
    """Load VideoGS default (qp=25) metrics and timing for the anchor frame."""
    fid = cfg.frame_id
    group_tag = f"frames_{fid}_{fid + cfg.group_size - 1}_int_1"
    out_dir = os.path.join(cfg.model_root, "compression", "videogs", "qp_25", group_tag)

    entry = MethodEntry()
    bench_path = os.path.join(out_dir, "benchmark_videogs_pipeline.csv")
    if os.path.isfile(bench_path):
        try:
            with open(bench_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if int(row["frame_id"]) == fid:
                        entry.encode_ms = float(row["total_encode_ms"])
                        entry.decode_ms = float(row["total_decode_ms"])
                        gop_avg = int(row["compressed_size_gop_avg_bytes"])
                        uncomp = int(row["uncompressed_size_bytes"])
                        entry.compressed_mb = gop_avg / (1024 * 1024)
                        if gop_avg > 0:
                            entry.compression_ratio = uncomp / gop_avg
                        break
        except (OSError, KeyError, ValueError) as exc:
            print(f"  [WARN] VideoGS benchmark: {exc}")

    metrics = _load_eval_metrics(os.path.join(out_dir, "evaluation", "evaluation_results.json"), fid)
    if metrics:
        entry.decomp_psnr = metrics["decomp_psnr"]
        entry.psnr_diff = metrics["decomp_psnr"] - metrics["gt_psnr"]

    return entry


def _load_mesongs_entry(cfg: TableSeqCfg) -> MethodEntry:
    """Load MesonGS default metrics and timing for the anchor frame."""
    fid = cfg.frame_id
    out_dir = os.path.join(cfg.model_root, "compression", "mesongs", cfg.mesongs_tag, f"frame{fid}")

    entry = MethodEntry()

    bench_path = os.path.join(out_dir, "benchmark_mesongs.csv")
    if os.path.isfile(bench_path):
        try:
            with open(bench_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if int(row["frame_id"]) == fid:
                        enc_col = "total_encode_ms" if "total_encode_ms" in row else "encode_time_ms"
                        dec_col = "total_decode_ms" if "total_decode_ms" in row else "decode_time_ms"
                        entry.encode_ms = float(row[enc_col])
                        entry.decode_ms = float(row[dec_col])
                        comp = int(row["compressed_size_bytes"])
                        uncomp = int(row["uncompressed_size_bytes"])
                        entry.compressed_mb = comp / (1024 * 1024)
                        if comp > 0:
                            entry.compression_ratio = uncomp / comp
                        break
        except (OSError, KeyError, ValueError) as exc:
            print(f"  [WARN] MesonGS benchmark: {exc}")

    metrics = _load_eval_metrics(os.path.join(out_dir, "evaluation", "evaluation_results.json"), fid)
    if metrics:
        entry.decomp_psnr = metrics["decomp_psnr"]
        entry.psnr_diff = metrics["decomp_psnr"] - metrics["gt_psnr"]

    return entry


def _load_gpcc_entry(cfg: TableSeqCfg, target_psnr: float, ref_uncomp_bytes: int | None) -> MethodEntry:
    """Load GPCC Pareto-frontier point closest to *target_psnr* for this sequence.

    Since GPCC benchmark CSVs lack uncompressed_size_bytes, *ref_uncomp_bytes*
    (obtained from another baseline) is used for the compression ratio.
    """
    fid = cfg.frame_id
    gpcc_root = os.path.join(cfg.model_root, "compression", "gpcc")

    candidates: list[dict[str, Any]] = []
    for depth in cfg.gpcc_octree_depths:
        for qp_rest, qp_dc, qp_opacity in GPCC_QP_COMBOS:
            params_tag = f"J{depth}_rest{qp_rest}_dc{qp_dc}_op{qp_opacity}"
            out_dir = os.path.join(gpcc_root, params_tag, f"frame{fid}")

            bench_path = os.path.join(out_dir, "benchmark_gpcc.csv")
            eval_path = os.path.join(out_dir, "evaluation", "evaluation_results.json")

            if not os.path.isfile(bench_path) or not os.path.isfile(eval_path):
                continue

            try:
                bench_data: dict[str, Any] | None = None
                with open(bench_path, newline="", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        if int(row["frame_idx"]) == fid:
                            bench_data = {
                                "encode_ms": float(row["encode_time_s"]) * 1000,
                                "decode_ms": float(row["decode_time_s"]) * 1000,
                                "compressed_bytes": int(row["total_compressed_bytes"]),
                            }
                            break
                if bench_data is None:
                    continue

                metrics = _load_eval_metrics(eval_path, fid)
                if metrics is None:
                    continue

                candidates.append({
                    **bench_data,
                    "decomp_psnr": metrics["decomp_psnr"],
                    "gt_psnr": metrics["gt_psnr"],
                })
            except (OSError, KeyError, ValueError):
                continue

    if not candidates:
        return MethodEntry()

    candidates.sort(key=lambda c: c["compressed_bytes"])
    frontier: list[dict[str, Any]] = [candidates[0]]
    for c in candidates[1:]:
        if c["decomp_psnr"] > frontier[-1]["decomp_psnr"]:
            frontier.append(c)

    best = min(frontier, key=lambda c: abs(c["decomp_psnr"] - target_psnr))

    comp_bytes = best["compressed_bytes"]
    entry = MethodEntry(
        encode_ms=best["encode_ms"],
        decode_ms=best["decode_ms"],
        decomp_psnr=best["decomp_psnr"],
        psnr_diff=best["decomp_psnr"] - best["gt_psnr"],
        compressed_mb=comp_bytes / (1024 * 1024),
    )
    if ref_uncomp_bytes is not None and comp_bytes > 0:
        entry.compression_ratio = ref_uncomp_bytes / comp_bytes

    return entry


def _load_livogs_hull(cfg: TableSeqCfg) -> list[tuple[float, float]]:
    """Load LivoGS R-D hull points as (compressed_mb, decomp_psnr)."""
    hull_csv = os.path.join(
        cfg.model_root, "compression", LIVOGS_RD_SUBDIR, "plots",
        f"acdc_psnr_size_curve_frame{cfg.frame_id}_sweep_hull.csv",
    )
    if not os.path.isfile(hull_csv):
        return []
    pts: list[tuple[float, float]] = []
    try:
        with open(hull_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                pts.append((float(row["compressed_mb"]), float(row["decomp_psnr"])))
    except (OSError, KeyError, ValueError):
        pass
    pts.sort(key=lambda p: p[0])
    return pts


def _load_livogs_sweep(cfg: TableSeqCfg) -> list[dict[str, Any]]:
    """Load LivoGS sweep summary (all evaluated configs)."""
    sweep_csv = os.path.join(
        cfg.model_root, "compression", LIVOGS_RD_SUBDIR,
        f"acdc_hull_sweep_summary_frame{cfg.frame_id}.csv",
    )
    if not os.path.isfile(sweep_csv):
        return []
    rows: list[dict[str, Any]] = []
    try:
        with open(sweep_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(row)
    except (OSError, KeyError, ValueError):
        pass
    return rows


def _find_livogs_samepsnr_entry(
    cfg: TableSeqCfg,
    target_psnr: float,
    hull: list[tuple[float, float]],
    sweep: list[dict[str, Any]],
    ref_uncomp_bytes: int | None,
) -> MethodEntry:
    """Find LivoGS hull point closest to *target_psnr*, then load timing."""
    if not hull:
        return MethodEntry()

    best_mb, best_psnr = min(hull, key=lambda p: abs(p[1] - target_psnr))

    entry = MethodEntry(compressed_mb=best_mb, decomp_psnr=best_psnr)

    if not sweep:
        return entry

    best_sweep = min(
        sweep,
        key=lambda r: (float(r["compressed_mb"]) - best_mb) ** 2
                     + (float(r["decomp_psnr"]) - best_psnr) ** 2,
    )

    label = str(best_sweep.get("label", ""))
    depth = int(best_sweep.get("depth", 0))
    if not label or depth <= 0:
        return entry

    exp_dir = os.path.join(
        cfg.model_root, "compression", LIVOGS_RD_SUBDIR,
        f"frame_{cfg.frame_id}", f"J_{depth}", label,
    )
    bench_path = os.path.join(exp_dir, "benchmark_livogs.csv")
    if os.path.isfile(bench_path):
        try:
            with open(bench_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if int(row["frame_id"]) == cfg.frame_id:
                        entry.encode_ms = float(row["encode_time_ms"])
                        entry.decode_ms = float(row["decode_time_ms"])
                        comp = int(row["compressed_size_bytes"])
                        uncomp = int(row["uncompressed_size_bytes"])
                        entry.compressed_mb = comp / (1024 * 1024)
                        if comp > 0:
                            entry.compression_ratio = uncomp / comp
                        break
        except (OSError, KeyError, ValueError) as exc:
            print(f"  [WARN] LivoGS benchmark {bench_path}: {exc}")

    eval_path = os.path.join(exp_dir, "evaluation", "evaluation_results.json")
    metrics = _load_eval_metrics(eval_path, cfg.frame_id)
    if metrics:
        entry.decomp_psnr = metrics["decomp_psnr"]
        entry.psnr_diff = metrics["decomp_psnr"] - metrics["gt_psnr"]
    elif entry.decomp_psnr is not None:
        pass

    return entry


def _fmt(val: float | None, fmt: str = ".2f") -> str:
    return "" if val is None else f"{val:{fmt}}"


TABLE_METHODS: list[tuple[str, str]] = [
    ("Ours", "GS-NFS (Ours)"),
    ("VideoGS", "V3-2D"),
    ("MesonGS", "MesonGS"),
    ("GPCC", "GPCC"),
]
TABLE_METRICS: list[tuple[str, str]] = [
    ("encode_ms", "Encoding Time (ms)"),
    ("decode_ms", "Decoding Time (ms)"),
    ("psnr_diff", "PSNR (dB)"),
    ("compression_ratio", "Compression Ratio"),
]

TABLE_INTERNAL_COLS = ["dataset", "sequence"]
for _tag, _ in TABLE_METHODS:
    for _metric, _ in TABLE_METRICS:
        TABLE_INTERNAL_COLS.append(f"{_tag}_{_metric}")


def generate_comparison_table(output_dir: str) -> None:
    sep = "=" * 70
    print(f"\n{sep}\nComparison Table — Same-PSNR Matching (GS-NFS matched to V3-2D PSNR)\n{sep}")

    cfgs = _build_table_cfgs()
    table_rows: list[dict[str, Any]] = []

    for cfg in cfgs:
        tag = f"{cfg.dataset}/{cfg.sequence}"
        print(f"\n  Processing {tag} (frame {cfg.frame_id})")

        vgs = _load_videogs_entry(cfg)
        mgs = _load_mesongs_entry(cfg)

        ref_uncomp: int | None = None
        for baseline_entry in [vgs, mgs]:
            if baseline_entry.compression_ratio is not None and baseline_entry.compressed_mb is not None:
                comp_bytes = int(baseline_entry.compressed_mb * 1024 * 1024)
                ref_uncomp = int(baseline_entry.compression_ratio * comp_bytes)
                break

        gpcc_target_psnr = vgs.decomp_psnr if vgs.decomp_psnr is not None else 0.0
        gpcc = _load_gpcc_entry(cfg, gpcc_target_psnr, ref_uncomp)

        hull = _load_livogs_hull(cfg)
        sweep = _load_livogs_sweep(cfg)

        if not hull:
            print(f"    [WARN] No GS-NFS hull for {tag}")

        ours = _find_livogs_samepsnr_entry(
            cfg, vgs.decomp_psnr or 0.0, hull, sweep, ref_uncomp,
        )

        row: dict[str, Any] = {"dataset": cfg.dataset, "sequence": cfg.sequence}
        for method_tag, entry in [
            ("Ours", ours), ("VideoGS", vgs), ("MesonGS", mgs), ("GPCC", gpcc),
        ]:
            row[f"{method_tag}_encode_ms"] = entry.encode_ms
            row[f"{method_tag}_decode_ms"] = entry.decode_ms
            row[f"{method_tag}_psnr_diff"] = entry.psnr_diff
            row[f"{method_tag}_compression_ratio"] = entry.compression_ratio

        table_rows.append(row)

        ours_display = _display_name("Ours")
        print(
            f"    {ours_display}: enc={_fmt(ours.encode_ms)}ms  dec={_fmt(ours.decode_ms)}ms  "
            f"CR={_fmt(ours.compression_ratio)}x  PSNR={_fmt(ours.decomp_psnr)}"
        )
        for bl_key, bl_e in [("VideoGS", vgs), ("MesonGS", mgs), ("GPCC", gpcc)]:
            bl_display = _display_name(bl_key)
            print(
                f"    {bl_display:8s} enc={_fmt(bl_e.encode_ms)}ms  dec={_fmt(bl_e.decode_ms)}ms  "
                f"CR={_fmt(bl_e.compression_ratio)}x  PSNR={_fmt(bl_e.decomp_psnr)}"
            )

    metric_internal_cols = [c for c in TABLE_INTERNAL_COLS if c not in ("dataset", "sequence")]

    def _write_table(path: str, rows: list[dict[str, Any]]) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            method_header = ["", ""]
            for _tag, display in TABLE_METHODS:
                method_header.append(display)
                method_header.extend([""] * (len(TABLE_METRICS) - 1))
            w.writerow(method_header)
            metric_header = ["dataset", "sequence"]
            for _ in TABLE_METHODS:
                for _m_key, m_label in TABLE_METRICS:
                    metric_header.append(m_label)
            w.writerow(metric_header)
            for row in rows:
                data: list[str] = [str(row.get("dataset", "")), str(row.get("sequence", ""))]
                for _tag, _ in TABLE_METHODS:
                    for _m_key, _ in TABLE_METRICS:
                        v = row.get(f"{_tag}_{_m_key}")
                        if v is None:
                            data.append("")
                        elif isinstance(v, float):
                            data.append(f"{v:.4f}")
                        else:
                            data.append(str(v))
                w.writerow(data)
        print(f"  Wrote {len(rows)} rows to: {path}")

    per_seq_path = os.path.join(output_dir, "comparison_table_same_psnr_per_sequence.csv")
    _write_table(per_seq_path, table_rows)

    avg_row: dict[str, Any] = {"dataset": "Average", "sequence": ""}
    for col in metric_internal_cols:
        vals = [r[col] for r in table_rows if isinstance(r.get(col), (int, float))]
        avg_row[col] = sum(vals) / len(vals) if vals else None
    avg_path = os.path.join(output_dir, "comparison_table_same_psnr_average.csv")
    _write_table(avg_path, [avg_row])


def main() -> None:
    sep = "=" * 70
    print(sep)
    print("Baseline Comparison — Collect & Plot")
    print(f"  Sequences:  {len(EXPERIMENTS)}")
    baseline_names = list(BASELINES.keys()) + (["GPCC"] if os.path.isfile(GPCC_DEFAULTS_FILE) else [])
    print(f"  Baselines:  {', '.join(baseline_names)}")
    print(f"  Output:     {OUTPUT_DIR}")
    print(sep)

    print(f"\n{sep}\nStep 1: Collect results\n{sep}")
    rows = collect_all_results()
    print(f"\n  Total results collected: {len(rows)}")

    if not rows:
        print("[ERROR] No results found. Did run_baseline_experiments.py complete?")
        sys.exit(1)

    by_baseline = _group_by(rows, "baseline")
    sorted_baselines = sorted(by_baseline.keys(), key=lambda b: _baseline_sort_key(by_baseline[b]))
    for bl in sorted_baselines:
        bl_rows = by_baseline.get(bl, [])
        if bl_rows:
            avg_psnr = np.mean([r["decomp_psnr"] for r in bl_rows])
            avg_size = np.mean([r["compressed_mb"] for r in bl_rows])
            print(
                f"    {bl:10s}: {len(bl_rows):3d} frames, "
                f"avg PSNR={avg_psnr:.2f} dB, avg size={avg_size:.2f} MB"
            )

    large_csv_path = os.path.join(OUTPUT_DIR, "baseline_results.csv")
    write_large_csv(rows, large_csv_path)

    print(f"\n{sep}\nStep 2: Generate plots\n{sep}")
    plot_dir = os.path.join(OUTPUT_DIR, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    for name in os.listdir(plot_dir):
        if name.endswith(".png"):
            os.remove(os.path.join(plot_dir, name))

    plot_size_by_sequence(rows, plot_dir)
    plot_quality_by_sequence(
        rows, plot_dir, "decomp_psnr", "gt_psnr",
        "PSNR (dB)", "PSNR by Sequence", "psnr_by_sequence.png",
        ylim=PLOT_YLIM.get("PSNR"),
    )
    plot_quality_by_sequence(
        rows, plot_dir, "decomp_ssim", "gt_ssim",
        "SSIM", "SSIM by Sequence", "ssim_by_sequence.png",
        ylim=PLOT_YLIM.get("SSIM"),
    )

    print(f"\n{sep}\nStep 3: Generate comparison table\n{sep}")
    generate_comparison_table(OUTPUT_DIR)

    print(f"\n{sep}")
    print(f"Done! All outputs in: {OUTPUT_DIR}")
    print(sep)


if __name__ == "__main__":
    main()
