#!/usr/bin/env python3
"""Generate same-PSNR comparison table for GS-NFS vs baseline methods.

This script reads existing experiment artifacts and builds a per-sequence table
with:
  - GS-NFS default latency (encode/decode)
  - Baseline default latency (VideoGS, MesonGS, LTS-Draco)
  - delta-PSNR and compression-diff against GS-NFS matched by anchor PSNR

Matching protocol (per sequence, per baseline):
  1) Read baseline anchor-frame PSNR from its default output.
  2) Read GS-NFS R-D hull for the same anchor frame.
  3) Find hull point with closest PSNR to the baseline anchor PSNR.
  4) Resolve sweep config nearest to that hull point.
  5) Load per-frame data from the matched GS-NFS experiment directory.
  6) Over common frames, compute and average:
     - delta_PSNR = PSNR(GS-NFS matched) - PSNR(baseline)
     - compression_diff = size(baseline) / size(GS-NFS matched)

Outputs are written to scripts/rd_baselines_results/:
  - same_psnr_table.csv
  - same_psnr_table.tex
  - same_psnr_frame_log.txt

Usage:
    python scripts/generate_same_psnr_table.py
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, cast


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "rd_baselines_results"
CSV_OUT = OUTPUT_DIR / "same_psnr_table.csv"
TEX_OUT = OUTPUT_DIR / "same_psnr_table.tex"
LOG_OUT = OUTPUT_DIR / "same_psnr_frame_log.txt"

HIFI4G_SEQS = [
    "4K_Actor1_Greeting",
    "4K_Actor2_Dancing",
    "4K_Actor3_Violin",
    "4K_Actor4_Dancing",
    "4K_Actor5_Oil-paper_Umbrella",
    "4K_Actor6_Changing_Clothes",
    "4K_Actor7_Nunchaku",
]

N3DV_SEQS = [
    "cook_spinach",
    "coffee_martini",
    "cut_roasted_beef",
    "flame_salmon_1",
    "flame_steak",
    "sear_steak",
]

LIVOGS_RD_SUBDIR = "livogs_rd_new"
GPCC_DEFAULTS_FILE = SCRIPT_DIR / "gpcc_defaults.json"


def _load_gpcc_defaults() -> dict[str, dict[str, int]]:
    if not GPCC_DEFAULTS_FILE.is_file():
        return {}
    with open(GPCC_DEFAULTS_FILE, encoding="utf-8") as f:
        return cast(dict[str, dict[str, int]], json.load(f))


def _gpcc_params_tag(params: dict[str, int]) -> str:
    return f"J{params['voxel_depth']}_rest{params['qp_rest']}_dc{params['qp_dc']}_op{params['qp_opacity']}"


@dataclass(frozen=True)
class SeqCfg:
    dataset: str
    sequence: str
    model_root: str
    anchor_frame: int
    max_candidate_frame: int
    mesongs_tag: str
    gpcc_tag: str


@dataclass
class FrameMetric:
    psnr: float
    size_bytes: float
    encode_ms: float | None = None
    decode_ms: float | None = None


@dataclass
class MethodStats:
    encode_latency: float | None = None
    decode_latency: float | None = None
    delta_psnr: float | None = None
    compression_diff: float | None = None


@dataclass
class SequenceRow:
    dataset: str
    sequence: str
    gs_nfs: MethodStats
    videogs: MethodStats
    mesongs: MethodStats
    dracogs: MethodStats
    gpcc: MethodStats


def _build_seq_cfgs() -> list[SeqCfg]:
    gpcc_defaults = _load_gpcc_defaults()
    cfgs: list[SeqCfg] = []
    for seq in HIFI4G_SEQS:
        gpcc_params = gpcc_defaults.get(seq, {})
        cfgs.append(
            SeqCfg(
                dataset="HiFi4G",
                sequence=seq,
                model_root=f"/synology/rajrup/VideoGS/train_output/HiFi4G_Dataset/{seq}",
                anchor_frame=0,
                max_candidate_frame=200,
                mesongs_tag="d12_nb8_nblk57_cb2048",
                gpcc_tag=_gpcc_params_tag(gpcc_params) if gpcc_params else "",
            )
        )
    for seq in N3DV_SEQS:
        gpcc_params = gpcc_defaults.get(seq, {})
        cfgs.append(
            SeqCfg(
                dataset="N3DV",
                sequence=seq,
                model_root=f"/synology/rajrup/Queen/pretrained_output/Neural_3D_Video/queen_compressed_{seq}",
                anchor_frame=1,
                max_candidate_frame=300,
                mesongs_tag="d17_nb8_nblk57_cb2048",
                gpcc_tag=_gpcc_params_tag(gpcc_params) if gpcc_params else "",
            )
        )
    return cfgs


def _read_csv_rows(path: str) -> list[dict[str, str]]:
    if not os.path.isfile(path):
        return []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def _to_int(v: object) -> int | None:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    if isinstance(v, str):
        try:
            return int(v)
        except ValueError:
            return None
    return None


def _to_float(v: object) -> float | None:
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    return None


def _read_eval_per_frame(path: str) -> dict[int, float]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data_obj = cast(object, json.load(f))
    except (OSError, json.JSONDecodeError):
        return {}

    if not isinstance(data_obj, dict):
        return {}
    data = cast(dict[str, object], data_obj)
    per_frame_obj = data.get("per_frame", [])
    if not isinstance(per_frame_obj, list):
        return {}
    per_frame = cast(list[object], per_frame_obj)

    out: dict[int, float] = {}
    for fr_obj in per_frame:
        if not isinstance(fr_obj, dict):
            continue
        fr = cast(dict[str, object], fr_obj)
        frame_obj = fr.get("frame", -1)
        psnr_obj = fr.get("decomp_psnr", 0.0)
        fid = _to_int(frame_obj)
        psnr = _to_float(psnr_obj)
        if fid is None or psnr is None:
            continue
        out[fid] = psnr
    return out


def _candidate_output_folders(
    model_root: str,
    subdir: str,
    output_tag: str,
    frame_id: int | None = None,
) -> list[str]:
    legacy_root = os.path.join(model_root, "compression", subdir, output_tag)
    candidates: list[str] = []
    if frame_id is not None:
        candidates.extend(
            [
                os.path.join(legacy_root, f"frame{frame_id}"),
                os.path.join(legacy_root, "frame_results", f"frame{frame_id}"),
                os.path.join(legacy_root, "frame_results"),
            ]
        )
    candidates.append(legacy_root)
    return candidates


def _discover_frame_dirs(
    cfg: SeqCfg,
    subdir: str,
    output_tag: str,
    frame_prefix: str,
) -> tuple[list[int], dict[int, str]]:
    frame_ids: list[int] = []
    resolved: dict[int, str] = {}
    for fid in range(0, cfg.max_candidate_frame + 1):
        candidates = _candidate_output_folders(cfg.model_root, subdir, output_tag, frame_id=fid)
        hit: str | None = None
        for folder in candidates:
            frame_folder = folder if os.path.basename(folder).startswith(frame_prefix) else os.path.join(folder, f"{frame_prefix}{fid}")
            if os.path.isdir(frame_folder):
                hit = frame_folder
                break
        if hit is not None:
            frame_ids.append(fid)
            resolved[fid] = hit
    return frame_ids, resolved


def _discover_videogs_anchors(cfg: SeqCfg) -> tuple[list[int], dict[int, str]]:
    candidates = [
        os.path.join(cfg.model_root, "compression", "videogs", "qp_25"),
        os.path.join(cfg.model_root, "compression", "videogs", "qp25"),
        os.path.join(cfg.model_root, "compression", "videogs", "qp_25", "frame_results"),
        os.path.join(cfg.model_root, "compression", "videogs", "qp25", "frame_results"),
    ]

    anchors: dict[int, str] = {}
    for root in candidates:
        if not os.path.isdir(root):
            continue
        try:
            entries = os.listdir(root)
        except OSError:
            continue
        for name in entries:
            if not (name.startswith("frames_") and name.endswith("_int_1")):
                continue
            parts = name.split("_")
            if len(parts) < 4:
                continue
            try:
                anchor = int(parts[1])
            except ValueError:
                continue
            if 0 <= anchor <= cfg.max_candidate_frame and anchor not in anchors:
                anchors[anchor] = os.path.join(root, name)

    sorted_anchors = sorted(anchors.keys())
    return sorted_anchors, anchors


def _load_method_frame_data(
    output_folder: str,
    benchmark_csv_name: str,
    size_column: str,
    anchor_only: bool,
    anchor_frame: int | None,
    frame_id_column: str = "frame_id",
) -> dict[int, FrameMetric]:
    benchmark_path = os.path.join(output_folder, benchmark_csv_name)
    eval_path = os.path.join(output_folder, "evaluation", "evaluation_results.json")

    psnr_by_frame = _read_eval_per_frame(eval_path)
    rows = _read_csv_rows(benchmark_path)

    by_frame: dict[int, FrameMetric] = {}
    for row in rows:
        try:
            fid = int(row[frame_id_column])
            size_bytes = float(row[size_column])
        except (KeyError, TypeError, ValueError):
            continue

        if anchor_only:
            is_anchor = str(row.get("is_anchor_frame", "")).strip().lower() in {"1", "true", "yes"}
            if anchor_frame is not None and fid != anchor_frame and not is_anchor:
                continue

        if fid not in psnr_by_frame:
            continue

        enc: float | None = None
        dec: float | None = None
        try:
            enc = float(row["total_encode_ms"])
            dec = float(row["total_decode_ms"])
        except (KeyError, TypeError, ValueError):
            pass

        by_frame[fid] = FrameMetric(psnr=psnr_by_frame[fid], size_bytes=size_bytes, encode_ms=enc, decode_ms=dec)
    return by_frame


def _avg_latency(frames: dict[int, FrameMetric]) -> tuple[float | None, float | None]:
    enc_vals = [v.encode_ms for v in frames.values() if v.encode_ms is not None]
    dec_vals = [v.decode_ms for v in frames.values() if v.decode_ms is not None]
    enc = sum(enc_vals) / len(enc_vals) if enc_vals else None
    dec = sum(dec_vals) / len(dec_vals) if dec_vals else None
    return enc, dec


def _load_hull_points(cfg: SeqCfg) -> list[tuple[float, float]]:
    hull_path = os.path.join(
        cfg.model_root,
        "compression",
        LIVOGS_RD_SUBDIR,
        "plots",
        f"acdc_psnr_size_curve_frame{cfg.anchor_frame}_sweep_hull.csv",
    )
    pts: list[tuple[float, float]] = []
    for row in _read_csv_rows(hull_path):
        try:
            pts.append((float(row["compressed_mb"]), float(row["decomp_psnr"])))
        except (KeyError, TypeError, ValueError):
            continue
    pts.sort(key=lambda p: p[0])
    return pts


def _load_sweep_rows(cfg: SeqCfg) -> list[dict[str, str]]:
    summary_path = os.path.join(
        cfg.model_root,
        "compression",
        LIVOGS_RD_SUBDIR,
        f"acdc_hull_sweep_summary_frame{cfg.anchor_frame}.csv",
    )
    return _read_csv_rows(summary_path)


def _find_matching_sweep_row(
    sweep_rows: list[dict[str, str]],
    target_mb: float,
    target_psnr: float,
) -> dict[str, str] | None:
    if not sweep_rows:
        return None
    try:
        return min(
            sweep_rows,
            key=lambda r: (float(r["compressed_mb"]) - target_mb) ** 2
            + (float(r["decomp_psnr"]) - target_psnr) ** 2,
        )
    except (KeyError, TypeError, ValueError):
        return None


def _find_matched_livogs_exp_dir(cfg: SeqCfg, sweep_row: dict[str, str] | None) -> str | None:
    if not sweep_row:
        return None
    label = str(sweep_row.get("label", "")).strip()
    depth = str(sweep_row.get("depth", "")).strip()
    if not label or not depth:
        return None
    return os.path.join(
        cfg.model_root,
        "compression",
        LIVOGS_RD_SUBDIR,
        f"frame_{cfg.anchor_frame}",
        f"J_{depth}",
        label,
    )


def _load_livogs_default_latency(cfg: SeqCfg) -> dict[int, FrameMetric]:
    latency_csv = os.path.join(cfg.model_root, "latency_benchmark", "livogs", "benchmark_livogs.csv")
    if os.path.isfile(latency_csv):
        out: dict[int, FrameMetric] = {}
        for row in _read_csv_rows(latency_csv):
            try:
                fid = int(row["frame_id"])
                out[fid] = FrameMetric(
                    psnr=0.0,
                    size_bytes=float(row.get("compressed_size_bytes", 0.0)),
                    encode_ms=float(row["total_encode_ms"]),
                    decode_ms=float(row["total_decode_ms"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
        if out:
            return out

    sweep_rows = _load_sweep_rows(cfg)
    default_like = [
        r
        for r in sweep_rows
        if "default" in str(r.get("label", "")).lower() or "klt" in str(r.get("label", "")).lower()
    ]
    candidate_rows = default_like if default_like else sweep_rows[:]

    for row in candidate_rows:
        exp_dir = _find_matched_livogs_exp_dir(cfg, row)
        if not exp_dir:
            continue
        bench_path = os.path.join(exp_dir, "benchmark_livogs.csv")
        if not os.path.isfile(bench_path):
            continue
        frames = _load_method_frame_data(
            exp_dir,
            "benchmark_livogs.csv",
            size_column="compressed_size_bytes",
            anchor_only=False,
            anchor_frame=None,
        )
        if frames:
            return frames
    return {}


def _mean(vals: list[float]) -> float | None:
    return (sum(vals) / len(vals)) if vals else None


def _compute_delta_and_compression(
    baseline: dict[int, FrameMetric],
    livogs_matched: dict[int, FrameMetric],
) -> tuple[float | None, float | None, list[int]]:
    common = sorted(set(baseline.keys()) & set(livogs_matched.keys()))
    delta_samples: list[float] = []
    comp_samples: list[float] = []

    for fid in common:
        b = baseline[fid]
        l = livogs_matched[fid]
        if l.size_bytes <= 0:
            continue
        delta_samples.append(l.psnr - b.psnr)
        comp_samples.append(b.size_bytes / l.size_bytes)

    return _mean(delta_samples), _mean(comp_samples), common


def _fmt_db(v: float | None) -> str:
    return "" if v is None else f"{v:.2f}"


def _fmt_ratio(v: float | None) -> str:
    return "" if v is None else f"{v:.2f}"


def _method_headers() -> list[str]:
    return [
        "",
        "",
        "V^3-2D",
        "",
        "MesonGS",
        "",
        "LTS-Draco",
        "",
        "GPCC",
        "",
    ]


def _metric_headers() -> list[str]:
    return [
        "dataset",
        "sequence",
        "delta_PSNR",
        "compression_ratio",
        "delta_PSNR",
        "compression_ratio",
        "delta_PSNR",
        "compression_ratio",
        "delta_PSNR",
        "compression_ratio",
    ]


def _row_to_list(row: SequenceRow) -> list[str]:
    return [
        row.dataset,
        row.sequence,
        _fmt_db(row.videogs.delta_psnr),
        _fmt_ratio(row.videogs.compression_diff),
        _fmt_db(row.mesongs.delta_psnr),
        _fmt_ratio(row.mesongs.compression_diff),
        _fmt_db(row.dracogs.delta_psnr),
        _fmt_ratio(row.dracogs.compression_diff),
        _fmt_db(row.gpcc.delta_psnr),
        _fmt_ratio(row.gpcc.compression_diff),
    ]


def _aggregate_rows(dataset: str, rows: list[SequenceRow]) -> SequenceRow:
    def _agg(getter: Callable[[SequenceRow], float | None]) -> float | None:
        vals = [v for v in (getter(r) for r in rows) if v is not None]
        if not vals:
            return None
        return float(sum(vals) / len(vals))

    return SequenceRow(
        dataset=dataset,
        sequence="Average",
        gs_nfs=MethodStats(
            encode_latency=_agg(lambda r: r.gs_nfs.encode_latency),
            decode_latency=_agg(lambda r: r.gs_nfs.decode_latency),
        ),
        videogs=MethodStats(
            encode_latency=_agg(lambda r: r.videogs.encode_latency),
            decode_latency=_agg(lambda r: r.videogs.decode_latency),
            delta_psnr=_agg(lambda r: r.videogs.delta_psnr),
            compression_diff=_agg(lambda r: r.videogs.compression_diff),
        ),
        mesongs=MethodStats(
            encode_latency=_agg(lambda r: r.mesongs.encode_latency),
            decode_latency=_agg(lambda r: r.mesongs.decode_latency),
            delta_psnr=_agg(lambda r: r.mesongs.delta_psnr),
            compression_diff=_agg(lambda r: r.mesongs.compression_diff),
        ),
        dracogs=MethodStats(
            encode_latency=_agg(lambda r: r.dracogs.encode_latency),
            decode_latency=_agg(lambda r: r.dracogs.decode_latency),
            delta_psnr=_agg(lambda r: r.dracogs.delta_psnr),
            compression_diff=_agg(lambda r: r.dracogs.compression_diff),
        ),
        gpcc=MethodStats(
            encode_latency=_agg(lambda r: r.gpcc.encode_latency),
            decode_latency=_agg(lambda r: r.gpcc.decode_latency),
            delta_psnr=_agg(lambda r: r.gpcc.delta_psnr),
            compression_diff=_agg(lambda r: r.gpcc.compression_diff),
        ),
    )


def _write_csv(rows: list[SequenceRow], path: Path) -> None:
    os.makedirs(path.parent, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(_method_headers())
        w.writerow(_metric_headers())
        for row in rows:
            w.writerow(_row_to_list(row))
    print(f"  Wrote CSV: {path}")


def _tex_seq_name(dataset: str, sequence: str) -> str:
    if sequence == "Average":
        return "Mean"
    if dataset == "HiFi4G":
        # "4K_Actor1_Greeting" → "Actor1"
        parts = sequence.split("_")
        if len(parts) >= 2 and parts[1].startswith("Actor"):
            return parts[1]
        return sequence.replace("_", "\\_")
    # N3DV: escape underscores, drop trailing "_1" (e.g. flame_salmon_1)
    name = sequence
    if name.endswith("_1"):
        name = name[:-2]
    return name.replace("_", "\\_")


def _write_tex(rows: list[SequenceRow], path: Path) -> None:
    os.makedirs(path.parent, exist_ok=True)
    lines: list[str] = []
    lines.append("\\begin{tabular}{llccccccc}")
    lines.append("\\toprule")
    lines.append(
        "Sequence & \\multicolumn{2}{c}{\\vthree{}-2D} "
        "& \\multicolumn{2}{c}{MesonGS} "
        "& \\multicolumn{2}{c}{LTS-Draco} "
        "& \\multicolumn{2}{c}{GPCC} \\\\"
    )
    lines.append(
        "         & \\dpsnr{} (dB) & \\rcr{} "
        "& \\dpsnr{} (dB) & \\rcr{} "
        "& \\dpsnr{} (dB) & \\rcr{} "
        "& \\dpsnr{} (dB) & \\rcr{} \\\\"
    )
    lines.append("\\midrule")

    for i, row in enumerate(rows):
        if i > 0 and row.dataset == "N3DV" and rows[i - 1].dataset == "HiFi4G":
            lines.append("\\midrule")
        if row.sequence == "Average":
            lines.append("\\hline")

        seq_name = _tex_seq_name(row.dataset, row.sequence)
        cells = [
            seq_name,
            _fmt_db(row.videogs.delta_psnr),
            _fmt_ratio(row.videogs.compression_diff),
            _fmt_db(row.mesongs.delta_psnr),
            _fmt_ratio(row.mesongs.compression_diff),
            _fmt_db(row.dracogs.delta_psnr),
            _fmt_ratio(row.dracogs.compression_diff),
            _fmt_db(row.gpcc.delta_psnr),
            _fmt_ratio(row.gpcc.compression_diff),
        ]
        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    with open(path, "w", encoding="utf-8") as f:
        _ = f.write("\n".join(lines) + "\n")
    print(f"  Wrote LaTeX: {path}")


def _print_console_table(rows: list[SequenceRow]) -> None:
    headers = _metric_headers()
    method = _method_headers()
    data = [_row_to_list(r) for r in rows]
    all_rows = [method, headers] + data

    widths: list[int] = []
    for col in range(len(headers)):
        widths.append(max(len(r[col]) for r in all_rows))

    def _join(row: list[str]) -> str:
        return " | ".join(val.ljust(widths[i]) for i, val in enumerate(row))

    print("\n" + "=" * 120)
    print("Same-PSNR Comparison Table")
    print("=" * 120)
    print(_join(method))
    print(_join(headers))
    print("-" * 120)
    for row in data:
        print(_join(row))
    print("=" * 120)


def _write_log(log_lines: list[str], path: Path) -> None:
    os.makedirs(path.parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        _ = f.write("\n".join(log_lines) + "\n")
    print(f"  Wrote log: {path}")


def _warn(msg: str, log_lines: list[str]) -> None:
    print(f"  [WARN] {msg}")
    log_lines.append(f"[WARN] {msg}")


def _info(msg: str, log_lines: list[str]) -> None:
    print(f"  [INFO] {msg}")
    log_lines.append(f"[INFO] {msg}")


def _collect_baseline_frames(cfg: SeqCfg, log_lines: list[str]) -> tuple[
    dict[int, FrameMetric],
    dict[int, FrameMetric],
    dict[int, FrameMetric],
    dict[int, FrameMetric],
]:
    draco_frames: dict[int, FrameMetric] = {}
    meson_frames: dict[int, FrameMetric] = {}
    video_frames: dict[int, FrameMetric] = {}
    gpcc_frames: dict[int, FrameMetric] = {}

    draco_ids, draco_dirs = _discover_frame_dirs(cfg, "dracogs", "eg_16_eo_16_et_16_es_16_cl_10", "frame")
    _info(f"{cfg.dataset}/{cfg.sequence} DracoGS frames found: {draco_ids}", log_lines)
    for fid, folder in draco_dirs.items():
        d = _load_method_frame_data(folder, "benchmark_dracogs.csv", "compressed_size_bytes", False, None)
        if fid in d:
            draco_frames[fid] = d[fid]

    meson_ids, meson_dirs = _discover_frame_dirs(cfg, "mesongs", cfg.mesongs_tag, "frame")
    _info(f"{cfg.dataset}/{cfg.sequence} MesonGS frames found: {meson_ids}", log_lines)
    for fid, folder in meson_dirs.items():
        d = _load_method_frame_data(folder, "benchmark_mesongs.csv", "compressed_size_bytes", False, None)
        if fid in d:
            meson_frames[fid] = d[fid]

    v_anchors, v_dirs = _discover_videogs_anchors(cfg)
    _info(f"{cfg.dataset}/{cfg.sequence} VideoGS anchors found: {v_anchors}", log_lines)
    for anchor, folder in v_dirs.items():
        d = _load_method_frame_data(
            folder,
            "benchmark_videogs_pipeline.csv",
            "compressed_size_gop_avg_bytes",
            anchor_only=True,
            anchor_frame=anchor,
        )
        if anchor in d:
            video_frames[anchor] = d[anchor]

    if cfg.gpcc_tag:
        gpcc_ids, gpcc_dirs = _discover_frame_dirs(cfg, "gpcc", cfg.gpcc_tag, "frame")
        _info(f"{cfg.dataset}/{cfg.sequence} GPCC frames found: {gpcc_ids}", log_lines)
        for fid, folder in gpcc_dirs.items():
            d = _load_method_frame_data(
                folder, "benchmark_gpcc.csv", "total_compressed_bytes", False, None,
                frame_id_column="frame_idx",
            )
            if fid in d:
                gpcc_frames[fid] = d[fid]
    else:
        _warn(f"{cfg.dataset}/{cfg.sequence} GPCC defaults not available", log_lines)

    return draco_frames, meson_frames, video_frames, gpcc_frames


def _load_livogs_matched_frames(cfg: SeqCfg, target_psnr: float, log_lines: list[str]) -> dict[int, FrameMetric]:
    hull = _load_hull_points(cfg)
    if not hull:
        _warn(f"{cfg.dataset}/{cfg.sequence}: missing GS-NFS hull at anchor frame {cfg.anchor_frame}", log_lines)
        return {}

    target_point = min(hull, key=lambda p: abs(p[1] - target_psnr))
    sweep_rows = _load_sweep_rows(cfg)
    sweep_match = _find_matching_sweep_row(sweep_rows, target_point[0], target_point[1])
    exp_dir = _find_matched_livogs_exp_dir(cfg, sweep_match)
    if not exp_dir:
        _warn(f"{cfg.dataset}/{cfg.sequence}: failed to resolve matched GS-NFS experiment directory", log_lines)
        return {}

    frames = _load_method_frame_data(
        exp_dir,
        "benchmark_livogs.csv",
        "compressed_size_bytes",
        anchor_only=False,
        anchor_frame=None,
    )
    _info(
        f"{cfg.dataset}/{cfg.sequence}: matched GS-NFS for target PSNR {target_psnr:.4f} uses {exp_dir}; "
        + f"frames={sorted(frames.keys())}",
        log_lines,
    )
    if not frames:
        _warn(f"{cfg.dataset}/{cfg.sequence}: matched GS-NFS has no overlapping benchmark/eval frames", log_lines)
    return frames


def _sequence_row(cfg: SeqCfg, log_lines: list[str]) -> SequenceRow:
    tag = f"{cfg.dataset}/{cfg.sequence}"
    print(f"\n{'=' * 70}\nProcessing {tag}\n{'=' * 70}")

    draco_frames, meson_frames, video_frames, gpcc_frames = _collect_baseline_frames(cfg, log_lines)

    gs_default_frames = _load_livogs_default_latency(cfg)
    gs_enc, gs_dec = _avg_latency(gs_default_frames)
    if gs_enc is None or gs_dec is None:
        _warn(f"{tag}: GS-NFS default latency unavailable", log_lines)
    else:
        _info(f"{tag}: GS-NFS default latency frames={sorted(gs_default_frames.keys())}", log_lines)

    def _build_baseline_stats(name: str, baseline_frames: dict[int, FrameMetric]) -> MethodStats:
        b_enc, b_dec = _avg_latency(baseline_frames)
        if b_enc is None or b_dec is None:
            _warn(f"{tag}: {name} latency unavailable", log_lines)

        if cfg.anchor_frame not in baseline_frames:
            _warn(f"{tag}: {name} missing anchor frame {cfg.anchor_frame}", log_lines)
            return MethodStats(encode_latency=b_enc, decode_latency=b_dec)

        matched = _load_livogs_matched_frames(cfg, baseline_frames[cfg.anchor_frame].psnr, log_lines)
        if not matched:
            return MethodStats(encode_latency=b_enc, decode_latency=b_dec)

        delta, comp, common = _compute_delta_and_compression(baseline_frames, matched)
        if not common:
            _warn(f"{tag}: {name} has no common frames with matched GS-NFS", log_lines)
        else:
            _info(f"{tag}: {name} common frames for delta/compression: {common}", log_lines)

        return MethodStats(
            encode_latency=b_enc,
            decode_latency=b_dec,
            delta_psnr=delta,
            compression_diff=comp,
        )

    return SequenceRow(
        dataset=cfg.dataset,
        sequence=cfg.sequence,
        gs_nfs=MethodStats(encode_latency=gs_enc, decode_latency=gs_dec),
        videogs=_build_baseline_stats("VideoGS", video_frames),
        mesongs=_build_baseline_stats("MesonGS", meson_frames),
        dracogs=_build_baseline_stats("DracoGS", draco_frames),
        gpcc=_build_baseline_stats("GPCC", gpcc_frames),
    )


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cfgs = _build_seq_cfgs()
    log_lines: list[str] = []

    seq_rows: list[SequenceRow] = []
    for cfg in cfgs:
        seq_rows.append(_sequence_row(cfg, log_lines))

    hifi_rows = [r for r in seq_rows if r.dataset == "HiFi4G"]
    n3dv_rows = [r for r in seq_rows if r.dataset == "N3DV"]

    ordered_rows: list[SequenceRow] = []
    ordered_rows.extend(hifi_rows)
    ordered_rows.append(_aggregate_rows("HiFi4G", hifi_rows))
    ordered_rows.extend(n3dv_rows)
    ordered_rows.append(_aggregate_rows("N3DV", n3dv_rows))

    _print_console_table(ordered_rows)
    _write_csv(ordered_rows, CSV_OUT)
    _write_tex(ordered_rows, TEX_OUT)
    _write_log(log_lines, LOG_OUT)

    print("\nDone.")


if __name__ == "__main__":
    main()
