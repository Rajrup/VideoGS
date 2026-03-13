#!/usr/bin/env python3
"""Pick LivoGS settings that match VideoGS / MesonGS defaults in rate or PSNR.

For each sequence, reads:
  - VideoGS default (qp=25, group_size=20) per-group metrics
  - MesonGS default per-frame metrics
  - LivoGS R-D sweep hull

Outputs CSV files under  scripts/rd_baselines_results/  with the matched
LivoGS operating points.

Usage:
    python scripts/pick_livogs_settings.py
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "rd_baselines_results"

# ---------------------------------------------------------------------------
# Dataset / sequence configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SeqCfg:
    dataset: str
    sequence: str
    model_root: str          # root that contains  compression/
    frame_id: int
    group_size: int
    mesongs_depth: int
    mesongs_nb: int
    mesongs_nblk: int
    mesongs_cb: int
    livogs_rd_subdir: str    # livogs_rd_new


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


def _build_seq_cfgs() -> list[SeqCfg]:
    cfgs: list[SeqCfg] = []
    for seq in HIFI4G_SEQS:
        cfgs.append(SeqCfg(
            dataset="HiFi4G",
            sequence=seq,
            model_root=f"/synology/rajrup/VideoGS/train_output/HiFi4G_Dataset/{seq}",
            frame_id=0,
            group_size=20,
            mesongs_depth=12,
            mesongs_nb=8,
            mesongs_nblk=57,
            mesongs_cb=2048,
            livogs_rd_subdir="livogs_rd_new",
        ))
    for seq in N3DV_SEQS:
        cfgs.append(SeqCfg(
            dataset="N3DV",
            sequence=seq,
            model_root=f"/synology/rajrup/Queen/pretrained_output/Neural_3D_Video/queen_compressed_{seq}",
            frame_id=1,
            group_size=20,
            mesongs_depth=17,
            mesongs_nb=8,
            mesongs_nblk=57,
            mesongs_cb=2048,
            livogs_rd_subdir="livogs_rd_new",
        ))
    return cfgs


ALL_SEQ_CFGS = _build_seq_cfgs()

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _videogs_group_tag(frame_id: int, group_size: int) -> str:
    return f"frames_{frame_id}_{frame_id + group_size - 1}_int_1"


def load_videogs_default(cfg: SeqCfg) -> dict[str, Any] | None:
    """Load VideoGS qp=25 metrics for the anchor-frame GOP."""
    group_tag = _videogs_group_tag(cfg.frame_id, cfg.group_size)
    group_dir = os.path.join(cfg.model_root, "compression", "videogs", "qp_25", group_tag)

    eval_path = os.path.join(group_dir, "evaluation", "evaluation_results.json")
    bench_path = os.path.join(group_dir, "benchmark_videogs_pipeline.csv")

    # PSNR per frame (all frames in the GOP)
    per_frame_psnr: list[float] = []
    anchor_psnr: float | None = None
    if os.path.isfile(eval_path):
        with open(eval_path, encoding="utf-8") as f:
            d = json.load(f)
        for fr in d.get("per_frame", []):
            per_frame_psnr.append(float(fr["decomp_psnr"]))
            if int(fr["frame"]) == cfg.frame_id:
                anchor_psnr = float(fr["decomp_psnr"])

    # Size (GOP average) — same for every frame in the group
    gop_avg_bytes: int | None = None
    per_frame_size: list[float] = []
    if os.path.isfile(bench_path):
        with open(bench_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                fid = int(row["frame_id"])
                avg = int(row["compressed_size_gop_avg_bytes"])
                per_frame_size.append(avg / (1024 * 1024))
                if fid == cfg.frame_id:
                    gop_avg_bytes = avg

    if anchor_psnr is None or gop_avg_bytes is None:
        return None

    return {
        "anchor_psnr": anchor_psnr,
        "per_frame_psnr": per_frame_psnr,
        "compressed_mb": gop_avg_bytes / (1024 * 1024),
        "per_frame_size_mb": per_frame_size,
        "compressed_size_bytes": gop_avg_bytes,
    }


def load_mesongs_default(cfg: SeqCfg) -> dict[str, Any] | None:
    """Load MesonGS default metrics."""
    params_tag = f"d{cfg.mesongs_depth}_nb{cfg.mesongs_nb}_nblk{cfg.mesongs_nblk}_cb{cfg.mesongs_cb}"
    frame_dir_name = f"frame{cfg.frame_id}"
    out_dir = os.path.join(cfg.model_root, "compression", "mesongs", params_tag, frame_dir_name)

    eval_path = os.path.join(out_dir, "evaluation", "evaluation_results.json")
    bench_path = os.path.join(out_dir, "benchmark_mesongs.csv")

    psnr: float | None = None
    if os.path.isfile(eval_path):
        with open(eval_path, encoding="utf-8") as f:
            d = json.load(f)
        for fr in d.get("per_frame", []):
            if int(fr["frame"]) == cfg.frame_id:
                psnr = float(fr["decomp_psnr"])
                break

    size_bytes: int | None = None
    if os.path.isfile(bench_path):
        with open(bench_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if int(row["frame_id"]) == cfg.frame_id:
                    size_bytes = int(row["compressed_size_bytes"])
                    break

    if psnr is None or size_bytes is None:
        return None

    return {
        "anchor_psnr": psnr,
        "per_frame_psnr": [psnr],
        "compressed_mb": size_bytes / (1024 * 1024),
        "per_frame_size_mb": [size_bytes / (1024 * 1024)],
        "compressed_size_bytes": size_bytes,
    }


@dataclass
class LivoGSPoint:
    compressed_mb: float
    decomp_psnr: float


def load_livogs_hull(cfg: SeqCfg) -> list[LivoGSPoint]:
    """Load LivoGS convex-hull R-D points."""
    hull_csv = os.path.join(
        cfg.model_root, "compression", cfg.livogs_rd_subdir, "plots",
        f"acdc_psnr_size_curve_frame{cfg.frame_id}_sweep_hull.csv",
    )
    if not os.path.isfile(hull_csv):
        return []
    pts: list[LivoGSPoint] = []
    with open(hull_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pts.append(LivoGSPoint(
                compressed_mb=float(row["compressed_mb"]),
                decomp_psnr=float(row["decomp_psnr"]),
            ))
    pts.sort(key=lambda p: p.compressed_mb)
    return pts


def load_livogs_sweep(cfg: SeqCfg) -> list[dict[str, Any]]:
    """Load full LivoGS sweep summary (all evaluated configs)."""
    sweep_csv = os.path.join(
        cfg.model_root, "compression", cfg.livogs_rd_subdir,
        f"acdc_hull_sweep_summary_frame{cfg.frame_id}.csv",
    )
    if not os.path.isfile(sweep_csv):
        return []
    rows: list[dict[str, Any]] = []
    with open(sweep_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Matching logic
# ---------------------------------------------------------------------------

def find_closest_by_rate(hull: list[LivoGSPoint], target_mb: float) -> LivoGSPoint | None:
    if not hull:
        return None
    return min(hull, key=lambda p: abs(p.compressed_mb - target_mb))


def find_closest_by_psnr(hull: list[LivoGSPoint], target_psnr: float) -> LivoGSPoint | None:
    if not hull:
        return None
    return min(hull, key=lambda p: abs(p.decomp_psnr - target_psnr))


def find_sweep_config_for_point(
    sweep: list[dict[str, Any]], target_mb: float, target_psnr: float
) -> dict[str, Any] | None:
    """Find the sweep config row closest to (target_mb, target_psnr)."""
    if not sweep:
        return None
    return min(
        sweep,
        key=lambda r: (float(r["compressed_mb"]) - target_mb) ** 2
                     + (float(r["decomp_psnr"]) - target_psnr) ** 2,
    )


# ---------------------------------------------------------------------------
# CSV output columns
# ---------------------------------------------------------------------------

COMPARISON_COLS = [
    "dataset",
    "sequence",
    "frame_id",
    "comparison",             # "vs_VideoGS" or "vs_MesonGS"
    "method",                 # "VideoGS" / "MesonGS" / "LivoGS-samerate" / "LivoGS-samepsnr"
    "compressed_mb",
    "decomp_psnr",
    "compressed_size_bytes",
    # VideoGS config details (populated for VideoGS rows)
    "videogs_qp",
    "videogs_group_size",
    # MesonGS config details (populated for MesonGS rows)
    "mesongs_depth",
    "mesongs_nb",
    "mesongs_nblk",
    "mesongs_cb",
    # LivoGS config details (populated for LivoGS rows)
    "livogs_label",
    "livogs_depth",
    "livogs_seed_idx",
    "livogs_qp_dc",
    "livogs_qp_ac",
    "livogs_qp_quats",
    "livogs_qp_scales",
    "livogs_qp_opacity",
    "livogs_beta",
]


def _row(
    cfg: SeqCfg,
    comparison: str,
    method: str,
    compressed_mb: float,
    decomp_psnr: float,
    compressed_size_bytes: int | None = None,
    videogs_params: dict[str, Any] | None = None,
    mesongs_params: dict[str, Any] | None = None,
    livogs_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    r: dict[str, Any] = {
        "dataset": cfg.dataset,
        "sequence": cfg.sequence,
        "frame_id": cfg.frame_id,
        "comparison": comparison,
        "method": method,
        "compressed_mb": f"{compressed_mb:.6f}",
        "decomp_psnr": f"{decomp_psnr:.6f}",
        "compressed_size_bytes": compressed_size_bytes or "",
        "videogs_qp": "",
        "videogs_group_size": "",
        "mesongs_depth": "",
        "mesongs_nb": "",
        "mesongs_nblk": "",
        "mesongs_cb": "",
        "livogs_label": "",
        "livogs_depth": "",
        "livogs_seed_idx": "",
        "livogs_qp_dc": "",
        "livogs_qp_ac": "",
        "livogs_qp_quats": "",
        "livogs_qp_scales": "",
        "livogs_qp_opacity": "",
        "livogs_beta": "",
    }
    if videogs_params:
        r["videogs_qp"] = videogs_params.get("qp", "")
        r["videogs_group_size"] = videogs_params.get("group_size", "")
    if mesongs_params:
        r["mesongs_depth"] = mesongs_params.get("depth", "")
        r["mesongs_nb"] = mesongs_params.get("nb", "")
        r["mesongs_nblk"] = mesongs_params.get("nblk", "")
        r["mesongs_cb"] = mesongs_params.get("cb", "")
    if livogs_config:
        r["livogs_label"] = livogs_config.get("label", "")
        r["livogs_depth"] = livogs_config.get("depth", "")
        r["livogs_seed_idx"] = livogs_config.get("seed_idx", "")
        r["livogs_qp_dc"] = livogs_config.get("qp_dc", "")
        r["livogs_qp_ac"] = livogs_config.get("qp_ac", "")
        r["livogs_qp_quats"] = livogs_config.get("qp_quats", "")
        r["livogs_qp_scales"] = livogs_config.get("qp_scales", "")
        r["livogs_qp_opacity"] = livogs_config.get("qp_opacity", "")
        r["livogs_beta"] = livogs_config.get("beta", "")
    return r


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    skipped: list[str] = []

    for cfg in ALL_SEQ_CFGS:
        tag = f"{cfg.dataset}/{cfg.sequence}"
        print(f"\n{'='*60}")
        print(f"Processing {tag}  (frame {cfg.frame_id})")
        print(f"{'='*60}")

        # Load baselines
        vgs = load_videogs_default(cfg)
        mgs = load_mesongs_default(cfg)
        hull = load_livogs_hull(cfg)
        sweep = load_livogs_sweep(cfg)

        if vgs is None:
            print(f"  WARNING: VideoGS default not found for {tag}")
            skipped.append(f"{tag} (VideoGS)")
        if mgs is None:
            print(f"  WARNING: MesonGS default not found for {tag}")
            skipped.append(f"{tag} (MesonGS)")
        if not hull:
            print(f"  WARNING: LivoGS hull not found for {tag}")
            skipped.append(f"{tag} (LivoGS hull)")
            continue

        print(f"  LivoGS hull: {len(hull)} points, "
              f"rate [{hull[0].compressed_mb:.4f}, {hull[-1].compressed_mb:.4f}] MB, "
              f"PSNR [{hull[0].decomp_psnr:.4f}, {hull[-1].decomp_psnr:.4f}] dB")
        print(f"  LivoGS sweep: {len(sweep)} configs")

        # --- VideoGS comparison ---
        if vgs:
            vgs_rate = vgs["compressed_mb"]
            vgs_psnr = vgs["anchor_psnr"]
            print(f"\n  VideoGS default: rate={vgs_rate:.4f} MB  PSNR={vgs_psnr:.4f} dB")

            all_rows.append(_row(cfg, "vs_VideoGS", "VideoGS",
                                 vgs_rate, vgs_psnr, vgs["compressed_size_bytes"],
                                 videogs_params={"qp": 25, "group_size": cfg.group_size}))

            livogs_samerate = find_closest_by_rate(hull, vgs_rate)
            if livogs_samerate:
                sr_config = find_sweep_config_for_point(
                    sweep, livogs_samerate.compressed_mb, livogs_samerate.decomp_psnr)
                print(f"  LivoGS same-rate:  rate={livogs_samerate.compressed_mb:.4f} MB  "
                      f"PSNR={livogs_samerate.decomp_psnr:.4f} dB  "
                      f"(Δrate={livogs_samerate.compressed_mb - vgs_rate:+.4f})")
                all_rows.append(_row(cfg, "vs_VideoGS", "LivoGS-samerate",
                                     livogs_samerate.compressed_mb,
                                     livogs_samerate.decomp_psnr,
                                     livogs_config=sr_config))

            livogs_samepsnr = find_closest_by_psnr(hull, vgs_psnr)
            if livogs_samepsnr:
                sp_config = find_sweep_config_for_point(
                    sweep, livogs_samepsnr.compressed_mb, livogs_samepsnr.decomp_psnr)
                print(f"  LivoGS same-PSNR: rate={livogs_samepsnr.compressed_mb:.4f} MB  "
                      f"PSNR={livogs_samepsnr.decomp_psnr:.4f} dB  "
                      f"(ΔPSNR={livogs_samepsnr.decomp_psnr - vgs_psnr:+.4f})")
                all_rows.append(_row(cfg, "vs_VideoGS", "LivoGS-samepsnr",
                                     livogs_samepsnr.compressed_mb,
                                     livogs_samepsnr.decomp_psnr,
                                     livogs_config=sp_config))

        # --- MesonGS comparison ---
        if mgs:
            mgs_rate = mgs["compressed_mb"]
            mgs_psnr = mgs["anchor_psnr"]
            params_str = (f"d={cfg.mesongs_depth} nb={cfg.mesongs_nb} "
                          f"nblk={cfg.mesongs_nblk} cb={cfg.mesongs_cb}")
            print(f"\n  MesonGS default ({params_str}): "
                  f"rate={mgs_rate:.4f} MB  PSNR={mgs_psnr:.4f} dB")

            all_rows.append(_row(cfg, "vs_MesonGS", "MesonGS",
                                 mgs_rate, mgs_psnr, mgs["compressed_size_bytes"],
                                 mesongs_params={
                                     "depth": cfg.mesongs_depth,
                                     "nb": cfg.mesongs_nb,
                                     "nblk": cfg.mesongs_nblk,
                                     "cb": cfg.mesongs_cb,
                                 }))

            livogs_samerate = find_closest_by_rate(hull, mgs_rate)
            if livogs_samerate:
                sr_config = find_sweep_config_for_point(
                    sweep, livogs_samerate.compressed_mb, livogs_samerate.decomp_psnr)
                print(f"  LivoGS same-rate:  rate={livogs_samerate.compressed_mb:.4f} MB  "
                      f"PSNR={livogs_samerate.decomp_psnr:.4f} dB  "
                      f"(Δrate={livogs_samerate.compressed_mb - mgs_rate:+.4f})")
                all_rows.append(_row(cfg, "vs_MesonGS", "LivoGS-samerate",
                                     livogs_samerate.compressed_mb,
                                     livogs_samerate.decomp_psnr,
                                     livogs_config=sr_config))

            livogs_samepsnr = find_closest_by_psnr(hull, mgs_psnr)
            if livogs_samepsnr:
                sp_config = find_sweep_config_for_point(
                    sweep, livogs_samepsnr.compressed_mb, livogs_samepsnr.decomp_psnr)
                print(f"  LivoGS same-PSNR: rate={livogs_samepsnr.compressed_mb:.4f} MB  "
                      f"PSNR={livogs_samepsnr.decomp_psnr:.4f} dB  "
                      f"(ΔPSNR={livogs_samepsnr.decomp_psnr - mgs_psnr:+.4f})")
                all_rows.append(_row(cfg, "vs_MesonGS", "LivoGS-samepsnr",
                                     livogs_samepsnr.compressed_mb,
                                     livogs_samepsnr.decomp_psnr,
                                     livogs_config=sp_config))

    # --- Write output CSVs ---
    # All comparisons
    all_csv = OUTPUT_DIR / "livogs_matched_settings_all.csv"
    _write_csv(all_csv, all_rows)

    # Per-comparison splits
    for comp in ("vs_VideoGS", "vs_MesonGS"):
        subset = [r for r in all_rows if r["comparison"] == comp]
        if subset:
            out = OUTPUT_DIR / f"livogs_matched_settings_{comp}.csv"
            _write_csv(out, subset)

    # Per-dataset splits
    for ds in ("HiFi4G", "N3DV"):
        subset = [r for r in all_rows if r["dataset"] == ds]
        if subset:
            out = OUTPUT_DIR / f"livogs_matched_settings_{ds}.csv"
            _write_csv(out, subset)

    if skipped:
        print(f"\nSkipped ({len(skipped)}):")
        for s in skipped:
            print(f"  - {s}")

    print("\nDone.")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    os.makedirs(path.parent, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COMPARISON_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} rows -> {path}")


if __name__ == "__main__":
    main()
