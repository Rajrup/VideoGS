#!/usr/bin/env python3
# pyright: reportMissingImports=false

from __future__ import annotations

import csv
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.figure as mfigure
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
PLOT_SEQUENCES: list[str] | None = []
PLOT_VIDEOGS_GROUP_SIZES: list[int] | None = [20]
PLOT_FORCE_RECOLLECT = True

# -- Per-sequence DracoGS sweep overrides -----------------------------------
DRACOGS_SWEEP_OVERRIDES: dict[str, dict[str, Any]] = {
    "4K_Actor1_Greeting": dict(
        dracogs_eg=(0, 8, 10, 12, 14, 16),
        dracogs_eo=(0, 8, 10, 12, 14, 16),
        dracogs_et=(0, 8, 10, 12, 14, 16),
        dracogs_es=(0, 8, 10, 12, 14, 16),
    ),
    "flame_salmon_1": dict(
        dracogs_eg=(0, 8, 10, 12, 14, 16),
        dracogs_eo=(0, 8, 10, 12, 14, 16),
        dracogs_et=(0, 8, 10, 12, 14, 16),
        dracogs_es=(0, 8, 10, 12, 14, 16),
    ),
    "sear_steak": dict(
        dracogs_eg=(0, 8, 10, 12, 14, 16),
        dracogs_eo=(0, 8, 10, 12, 14, 16),
        dracogs_et=(0, 8, 10, 12, 14, 16),
        dracogs_es=(0, 8, 10, 12, 14, 16),
    ),
}

# -- LivoGS hull overlay ---------------------------------------------------
LIVOGS_DATA_PATHS: dict[str, str | None] = {
    "HiFi4G": "/synology/rajrup/VideoGS",
    "N3DV": "/synology/rajrup/Queen",
}
LIVOGS_RD_SUBDIR: str = "livogs_rd_new"

RD_BASELINES = [
    "LivoGS",
    "VideoGS",
    "DracoGS",
    "MesonGS",
    "GPCC"
]
BASELINE_DISPLAY_NAMES: dict[str, str] = {
    "LivoGS": "GS-NFS (Ours)",
    "VideoGS": r"V$^3$-2D",
    "DracoGS": "LTS-Draco",
    "GPCC": "G-PCC",
}

RD_BASELINES_STYLES: dict[str, dict[str, Any]] = {
    "VideoGS": {"color": "#d62728", "marker": "^", "alpha": 0.7, "size": 35},
    "LivoGS": {"color": "#ff7f0e", "marker": "D", "alpha": 0.7, "size": 35},
    "MesonGS": {"color": "#2ca02c", "marker": "s", "alpha": 0.45, "size": 18},
    "DracoGS": {"color": "#1f77b4", "marker": "o", "alpha": 0.45, "size": 18},
    "GPCC": {"color": "#9467bd", "marker": "P", "alpha": 0.45, "size": 18},
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

COLLECTED_CSV_COLUMNS = [
    "dataset",
    "sequence",
    "frame_id",
    "baseline",
    "params",
    "group_size",
    "compressed_size_bytes",
    "compressed_mb",
    "uncompressed_size_bytes",
    "decomp_psnr",
    "decomp_ssim",
    "gt_psnr",
    "gt_ssim",
]


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
            parsed["decomp_ssim"] = _to_float(row.get("decomp_ssim"))
            parsed["gt_psnr"] = _to_float(row.get("gt_psnr"))
            parsed["gt_ssim"] = _to_float(row.get("gt_ssim"))
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


def convex_hull_upper(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Upper convex hull (concave envelope) for rate-distortion points.

    First reduces to the Pareto frontier (non-dominated, monotonically
    increasing), then removes interior points whose quality falls below
    the line connecting their neighbours — keeping only the operating
    points where no convex combination of two others achieves better
    quality at the same rate.
    """
    frontier = pareto_frontier(points)
    if len(frontier) <= 2:
        return frontier
    hull: list[tuple[float, float]] = []
    for p in frontier:
        while len(hull) >= 2:
            o, a = hull[-2], hull[-1]
            # >= 0 means left turn / collinear: middle point is on or
            # below the line from o to p, so drop it.
            cross = (a[0] - o[0]) * (p[1] - o[1]) - (a[1] - o[1]) * (p[0] - o[0])
            if cross >= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    return hull


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


def _load_livogs_hull_points(
    csv_path: str,
    metric_col: str = "decomp_psnr",
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                points.append((float(row["compressed_mb"]), float(row[metric_col])))
            except (KeyError, ValueError):
                continue
    points.sort(key=lambda p: p[0])
    return points


def _load_runner_module() -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_rd_baselines_experiments",
        str(SCRIPTS_DIR / "run_rd_baselines_experiments.py"),
    )
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    return runner


def _frame_span_tag(runner: Any, cfg: Any, frame_id: int) -> str:
    fs, fe, iv = runner._frame_span(cfg, frame_id)
    if cfg.frame_end_exclusive:
        return f"frames_{fs}_{fe - 1}_int_{iv}"
    return f"frames_{fs}_{fe}_int_{iv}"


def _mesongs_legacy_output_folder(
    runner: Any,
    cfg: Any,
    sequence: str,
    frame_id: int,
    depth: int,
    num_bits: int,
    n_block: int,
    cb: int,
) -> str:
    params_tag = f"d{depth}_nb{num_bits}_nblk{n_block}_cb{cb}"
    return str(
        runner._model_root(cfg, sequence)
        / "compression"
        / "mesongs"
        / params_tag
        / _frame_span_tag(runner, cfg, frame_id)
    )


def _dracogs_legacy_output_folder(
    runner: Any,
    cfg: Any,
    sequence: str,
    frame_id: int,
    eg: int,
    eo: int,
    et: int,
    es: int,
) -> str:
    params_tag = f"eg_{eg}_eo_{eo}_et_{et}_es_{es}_cl_{runner.SWEEP_SPACE.dracogs_cl}"
    return str(
        runner._model_root(cfg, sequence)
        / "compression"
        / "dracogs"
        / params_tag
        / _frame_span_tag(runner, cfg, frame_id)
    )


def _load_single_frame_result(
    output_folder: str,
    benchmark_csv_name: str,
    frame_id: int,
    compressed_size_field: str = "compressed_size_bytes",
    frame_id_field: str = "frame_id",
) -> dict[str, Any] | None:
    benchmark_path = os.path.join(output_folder, benchmark_csv_name)
    eval_json_path = os.path.join(output_folder, "evaluation", "evaluation_results.json")

    compressed_bytes: int | None = None
    uncompressed_bytes = 0
    if os.path.isfile(benchmark_path):
        try:
            with open(benchmark_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if int(row[frame_id_field]) == frame_id:
                        comp_raw = row.get(compressed_size_field) or row.get("compressed_size_bytes")
                        if comp_raw is None:
                            break
                        compressed_bytes = int(comp_raw)
                        uncompressed_bytes = int(row.get("uncompressed_size_bytes", 0))
                        break
        except (OSError, KeyError, ValueError):
            pass

    decomp_psnr: float | None = None
    decomp_ssim: float | None = None
    gt_psnr: float | None = None
    gt_ssim: float | None = None
    if os.path.isfile(eval_json_path):
        try:
            with open(eval_json_path, encoding="utf-8") as f:
                eval_data = json.load(f)
            for fr in eval_data.get("per_frame", []):
                if int(fr["frame"]) == frame_id:
                    decomp_psnr = float(fr["decomp_psnr"])
                    decomp_ssim = float(fr["decomp_ssim"])
                    gt_psnr = float(fr["gt_psnr"])
                    gt_ssim = float(fr["gt_ssim"])
                    break
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            pass

    if compressed_bytes is None or decomp_psnr is None:
        return None

    return {
        "compressed_size_bytes": compressed_bytes,
        "compressed_mb": compressed_bytes / (1024 * 1024),
        "uncompressed_size_bytes": uncompressed_bytes,
        "decomp_psnr": decomp_psnr,
        "decomp_ssim": decomp_ssim,
        "gt_psnr": gt_psnr,
        "gt_ssim": gt_ssim,
    }


def _collect_videogs_rows(runner: Any, cfg: Any, sequence: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    frame_ids = runner._frame_ids(cfg)
    frame_tag_re = re.compile(r"^frames_(\d+)_(\d+)_int_(\d+)$")

    def parse_frame_tag(out_path: Path) -> tuple[int | None, int | None, int | None, int | None]:
        m = frame_tag_re.match(out_path.name)
        if m is None:
            return None, None, None, None
        start = int(m.group(1))
        end = int(m.group(2))
        interval = int(m.group(3))
        if interval <= 0 or end < start:
            return start, end, interval, None
        group_size = ((end - start) // interval) + 1
        return start, end, interval, group_size

    def discover_output_folders_for_qp(qp: int, frame_id: int) -> list[tuple[str, int | None]]:
        qp_root = runner._model_root(cfg, sequence) / "compression" / "videogs" / f"qp_{qp}"
        candidates: list[tuple[str, int | None]] = []

        if qp_root.is_dir():
            for out_dir in sorted(glob.glob(str(qp_root / "frames_*_int_*"))):
                out_path = Path(out_dir)
                config_path = out_path / "videogs_config.json"
                detected_frame_start_from_tag, _, _, detected_group_size = parse_frame_tag(out_path)
                if (
                    detected_frame_start_from_tag is not None
                    and detected_frame_start_from_tag != frame_id
                ):
                    continue

                detected_frame_start: int | None = None
                if config_path.is_file():
                    try:
                        with open(config_path, encoding="utf-8") as f:
                            config = json.load(f)
                        raw_start = config.get("frame_start")
                        if raw_start is not None:
                            detected_frame_start = int(raw_start)
                    except (OSError, json.JSONDecodeError, ValueError, TypeError):
                        pass

                if detected_frame_start is not None and detected_frame_start != frame_id:
                    continue
                candidates.append((str(out_path), detected_group_size))

        default_out = runner._videogs_output_folder(cfg, sequence, frame_id, qp)
        default_group_size = runner.SWEEP_SPACE.videogs_group_size
        if not any(out == default_out for out, _ in candidates):
            candidates.append((default_out, default_group_size))

        unique: list[tuple[str, int | None]] = []
        seen: set[str] = set()
        for out, gsize in candidates:
            if out in seen:
                continue
            seen.add(out)
            unique.append((out, gsize))
        return unique

    for frame_id in frame_ids:
        for qp in runner.SWEEP_SPACE.videogs_qps:
            for out, group_size in discover_output_folders_for_qp(qp, frame_id):
                row = _load_single_frame_result(
                    out,
                    "benchmark_videogs_pipeline.csv",
                    frame_id,
                    compressed_size_field="compressed_size_gop_avg_bytes",
                )
                if row is None:
                    continue
                param_suffix = f" g={group_size}" if group_size is not None else ""
                row.update(
                    dataset=cfg.name,
                    sequence=sequence,
                    frame_id=frame_id,
                    baseline="VideoGS",
                    params=f"qp={qp}{param_suffix}",
                    group_size=group_size,
                )
                rows.append(row)
    return rows


def _collect_mesongs_rows(runner: Any, cfg: Any, sequence: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frame_id in runner._frame_ids(cfg):
        for depth in runner.SWEEP_SPACE.mesongs_depths_by_dataset[cfg.name]:
            for num_bits in runner.SWEEP_SPACE.mesongs_num_bits:
                for n_block in runner.SWEEP_SPACE.mesongs_n_blocks:
                    for cb in runner.SWEEP_SPACE.mesongs_codebook_sizes:
                        out = runner._mesongs_output_folder(
                            cfg,
                            sequence,
                            frame_id,
                            depth,
                            num_bits,
                            n_block,
                            cb,
                        )
                        row = _load_single_frame_result(out, "benchmark_mesongs.csv", frame_id)
                        if row is None:
                            legacy_out = _mesongs_legacy_output_folder(
                                runner,
                                cfg,
                                sequence,
                                frame_id,
                                depth,
                                num_bits,
                                n_block,
                                cb,
                            )
                            row = _load_single_frame_result(legacy_out, "benchmark_mesongs.csv", frame_id)
                        if row is None:
                            continue
                        row.update(
                            dataset=cfg.name,
                            sequence=sequence,
                            frame_id=frame_id,
                            baseline="MesonGS",
                            params=f"d={depth} nb={num_bits} nblk={n_block} cb={cb}",
                        )
                        rows.append(row)
    return rows


def _collect_dracogs_rows(runner: Any, cfg: Any, sequence: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frame_id in runner._frame_ids(cfg):
        for eg in runner.SWEEP_SPACE.dracogs_eg:
            for eo in runner.SWEEP_SPACE.dracogs_eo:
                for et in runner.SWEEP_SPACE.dracogs_et:
                    for es in runner.SWEEP_SPACE.dracogs_es:
                        out = runner._dracogs_output_folder(cfg, sequence, frame_id, eg, eo, et, es)
                        row = _load_single_frame_result(out, "benchmark_dracogs.csv", frame_id)
                        if row is None:
                            legacy_out = _dracogs_legacy_output_folder(
                                runner,
                                cfg,
                                sequence,
                                frame_id,
                                eg,
                                eo,
                                et,
                                es,
                            )
                            row = _load_single_frame_result(legacy_out, "benchmark_dracogs.csv", frame_id)
                        if row is None:
                            continue
                        row.update(
                            dataset=cfg.name,
                            sequence=sequence,
                            frame_id=frame_id,
                            baseline="DracoGS",
                            params=f"eg={eg} eo={eo} et={et} es={es}",
                        )
                        rows.append(row)
    return rows


def _collect_gpcc_rows(runner: Any, cfg: Any, sequence: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frame_id in runner._frame_ids(cfg):
        for depth in runner.SWEEP_SPACE.gpcc_octree_depths_by_dataset[cfg.name]:
            for qp_rest, qp_dc, qp_opacity in runner.SWEEP_SPACE.gpcc_qp_combos:
                out = runner._gpcc_output_folder(cfg, sequence, frame_id, depth, qp_opacity, qp_dc, qp_rest)
                row = _load_single_frame_result(
                    out,
                    "benchmark_gpcc.csv",
                    frame_id,
                    compressed_size_field="total_compressed_bytes",
                    frame_id_field="frame_idx",
                )
                if row is None:
                    continue
                row.update(
                    dataset=cfg.name,
                    sequence=sequence,
                    frame_id=frame_id,
                    baseline="GPCC",
                    params=f"J={depth} rest={qp_rest} dc={qp_dc} op={qp_opacity}",
                )
                rows.append(row)
    return rows


_METRIC_INFO: dict[str, dict[str, str]] = {
    "psnr": {
        "decomp_col": "decomp_psnr",
        "gt_col": "gt_psnr",
        "label": "PSNR (dB)",
        "gt_fmt": "GT ({value:.2f} dB)",
    },
    "ssim": {
        "decomp_col": "decomp_ssim",
        "gt_col": "gt_ssim",
        "label": "SSIM",
        "gt_fmt": "GT ({value:.4f})",
    },
}


def _render_rd_plot(
    dataset: str,
    sequence: str,
    frame_id: int,
    rows: list[dict[str, Any]],
    *,
    frontier_only: bool = False,
    metric: str = "psnr",
) -> mfigure.Figure:
    minfo = _METRIC_INFO[metric]
    decomp_col = minfo["decomp_col"]
    gt_col = minfo["gt_col"]
    y_label = minfo["label"]
    gt_fmt = minfo["gt_fmt"]

    fig, ax = plt.subplots(figsize=(10, 7))

    by_baseline: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_baseline[str(row.get("baseline", ""))].append(row)

    gt_candidates = [float(r[gt_col]) for r in rows if r.get(gt_col) is not None]
    gt_val = gt_candidates[0] if gt_candidates else None
    gpcc_max_x: float | None = None
    all_max_x: float | None = None

    for baseline in RD_BASELINES:
        style = RD_BASELINES_STYLES[baseline]
        baseline_rows = by_baseline.get(baseline, [])

        if baseline == "LivoGS" and not baseline_rows:
            hull_csv = _find_livogs_hull_csv(dataset, sequence, frame_id)
            if hull_csv is None:
                continue
            hull_pts = _load_livogs_hull_points(hull_csv, metric_col=decomp_col)
            if not hull_pts:
                continue
            if frontier_only and gt_val is not None:
                hull_pts = [(x, y) for x, y in hull_pts if y <= gt_val]
            if not hull_pts:
                continue
            hx, hy = zip(*hull_pts)
            if frontier_only:
                livogs_max = max(hx)
                if all_max_x is None or livogs_max > all_max_x:
                    all_max_x = livogs_max
            display_name = BASELINE_DISPLAY_NAMES.get("LivoGS", "LivoGS")
            label = display_name if frontier_only else f"{display_name} ({len(hull_pts)} pts)"
            ax.scatter(
                hx, hy,
                color=style["color"],
                marker=style["marker"],
                s=style["size"],
                alpha=style["alpha"],
                label=label,
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
        if frontier_only:
            all_xs = [float(r["compressed_mb"]) for r in baseline_rows if r.get("compressed_mb") is not None]
            if all_xs:
                cur_max = max(all_xs)
                if baseline == "GPCC":
                    gpcc_max_x = cur_max
                if all_max_x is None or cur_max > all_max_x:
                    all_max_x = cur_max
        if frontier_only and gt_val is not None:
            baseline_rows = [
                r for r in baseline_rows
                if r.get(decomp_col) is not None and float(r[decomp_col]) <= gt_val
            ]
            if not baseline_rows:
                continue
        xs = [float(r["compressed_mb"]) for r in baseline_rows if r.get("compressed_mb") is not None]
        ys = [float(r[decomp_col]) for r in baseline_rows if r.get(decomp_col) is not None]
        if not xs or not ys:
            continue

        frontier = convex_hull_upper(list(zip(xs, ys)))
        display_name = BASELINE_DISPLAY_NAMES.get(baseline, baseline)

        if frontier_only:
            if not frontier:
                continue
            plot_xs, plot_ys = zip(*frontier)
            label = display_name
        else:
            plot_xs, plot_ys = xs, ys
            label = f"{display_name} ({len(xs)} pts)"

        ax.scatter(
            plot_xs,
            plot_ys,
            color=style["color"],
            marker=style["marker"],
            s=style["size"],
            alpha=style["alpha"],
            label=label,
            edgecolors="none",
            zorder=3,
        )

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

    if gt_val is not None:
        ax.axhline(
            gt_val,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label=gt_fmt.format(value=gt_val),
            zorder=1,
        )

    if frontier_only:
        xlim_x = gpcc_max_x if gpcc_max_x is not None else all_max_x
        if xlim_x is not None:
            ax.set_xlim(left=0, right=math.ceil(xlim_x))
        ax.set_xlabel("Compressed Size (MB)", fontsize=20)
        ax.set_ylabel(y_label, fontsize=20)
        ax.tick_params(axis="both", labelsize=20)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=20, markerscale=2)
    else:
        ax.set_xlabel("Compressed Size (MB)")
        ax.set_ylabel(y_label)
        ax.set_title(f"R-D Curve ({y_label}) | {dataset} | {sequence} | Frame {frame_id}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=9, markerscale=2)
    fig.tight_layout()
    return fig


def _has_metric_data(rows: list[dict[str, Any]], metric: str) -> bool:
    decomp_col = _METRIC_INFO[metric]["decomp_col"]
    return any(r.get(decomp_col) is not None for r in rows)


def plot_group(
    dataset: str,
    sequence: str,
    frame_id: int,
    rows: list[dict[str, Any]],
    output_root: str,
) -> None:
    dataset_dir = os.path.join(output_root, dataset)
    os.makedirs(dataset_dir, exist_ok=True)

    for metric in _METRIC_INFO:
        if not _has_metric_data(rows, metric):
            continue

        stem = f"rd_baselines_curve_{sequence}_frame{frame_id}_{metric}"

        fig_png = _render_rd_plot(dataset, sequence, frame_id, rows, frontier_only=False, metric=metric)
        out_png = os.path.join(dataset_dir, f"{stem}.png")
        fig_png.savefig(out_png, dpi=150)
        plt.close(fig_png)
        print(f"Saved: {out_png}")

        fig_pdf = _render_rd_plot(dataset, sequence, frame_id, rows, frontier_only=True, metric=metric)
        out_pdf = os.path.join(dataset_dir, f"{stem}.pdf")
        fig_pdf.savefig(out_pdf)
        plt.close(fig_pdf)
        print(f"Saved: {out_pdf}")


@contextmanager
def _dracogs_sweep_override(runner_module: Any, sequence: str):
    overrides = DRACOGS_SWEEP_OVERRIDES.get(sequence)
    if not overrides:
        yield
        return
    from dataclasses import replace as _replace
    original = runner_module.SWEEP_SPACE
    runner_module.SWEEP_SPACE = _replace(original, **overrides)
    try:
        yield
    finally:
        runner_module.SWEEP_SPACE = original


def _group_plot_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("dataset", "")),
            str(row.get("sequence", "")),
            int(row.get("frame_id", 0)),
        )
        groups[key].append(row)
    return groups


def main() -> None:
    csv_path = PLOT_CSV_PATH
    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    print(f"Output dir: {PLOT_OUTPUT_DIR}")

    if PLOT_FORCE_RECOLLECT:
        print("Re-collecting baseline results ...")
        runner = _load_runner_module()
        all_collectors = {
            "videogs": _collect_videogs_rows,
            "mesongs": _collect_mesongs_rows,
            "dracogs": _collect_dracogs_rows,
            "gpcc": _collect_gpcc_rows,
        }
        active_baselines = {b.lower() for b in RD_BASELINES}
        collectors = {k: v for k, v in all_collectors.items() if k in active_baselines}

        allowed_datasets = set(PLOT_DATASETS) if PLOT_DATASETS else None
        allowed_sequences = set(PLOT_SEQUENCES) if PLOT_SEQUENCES else None
        videogs_allowed = set(PLOT_VIDEOGS_GROUP_SIZES) if PLOT_VIDEOGS_GROUP_SIZES is not None else None

        all_rows: list[dict[str, Any]] = []
        for ds_name, cfg in runner.ALL_DATASETS.items():
            if allowed_datasets is not None and ds_name not in allowed_datasets:
                continue
            if cfg.name not in runner.SEQUENCE_SETTINGS:
                continue
            sequences = list(runner.SEQUENCE_SETTINGS[cfg.name])
            if allowed_sequences is not None:
                sequences = [seq for seq in sequences if seq in allowed_sequences]
            for sequence in sequences:
                seq_rows: list[dict[str, Any]] = []
                with _dracogs_sweep_override(runner, sequence):
                    for bl_key, collector in collectors.items():
                        rows = collector(runner, cfg, sequence)
                        seq_rows.extend(rows)
                        if rows:
                            print(f"  Collected {cfg.name} | {sequence} | {bl_key}: {len(rows)} rows")

                # -- plot this sequence immediately --
                plot_rows = seq_rows
                if videogs_allowed is not None:
                    plot_rows = [
                        r
                        for r in plot_rows
                        if str(r.get("baseline", "")) != "VideoGS"
                        or r.get("group_size") in videogs_allowed
                    ]
                all_rows.extend(plot_rows)
                groups = _group_plot_rows(plot_rows)
                for (dataset, seq, frame_id), group_rows in sorted(groups.items()):
                    plot_group(dataset, seq, frame_id, group_rows, PLOT_OUTPUT_DIR)

        # write accumulated CSV
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=COLLECTED_CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for row in all_rows:
                writer.writerow(row)
        print(f"  Collected {len(all_rows)} total rows -> {csv_path}")
    else:
        print(f"Input CSV:  {csv_path}")
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
        groups = _group_plot_rows(rows)
        for (dataset, sequence, frame_id), group_rows in sorted(groups.items()):
            plot_group(dataset, sequence, frame_id, group_rows, PLOT_OUTPUT_DIR)


if __name__ == "__main__":
    main()
