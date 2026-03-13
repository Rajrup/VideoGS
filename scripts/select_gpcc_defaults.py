#!/usr/bin/env python3
"""Select per-sequence GPCC default parameters from R-D sweep results.

For each sequence (HiFi4G and N3DV), scans the GPCC R-D experiment outputs
(produced by run_rd_baselines_experiments.py) and selects the smallest-size
config whose quality is within 0.2 dB of the best available config
(depth=max, QP=(4,4,4)) and whose decomp PSNR does not exceed the GT PSNR.

Writes the selected parameters to a JSON file that run_baseline_experiments.py
can read as GPCC defaults.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PSNR_TOLERANCE = 0.2
PARAM_TAG_RE = re.compile(r"^J(\d+)_rest(\d+)_dc(\d+)_op(\d+)$")

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_JSON = SCRIPT_DIR / "gpcc_defaults.json"

THREAD_WORKERS = 32


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    data_path: str
    dataset_name: str
    sequences: tuple[str, ...]
    frame_id: int

    def model_root(self, sequence: str) -> Path:
        if self.name == "N3DV":
            return (
                Path(self.data_path) / "pretrained_output" / self.dataset_name
                / f"queen_compressed_{sequence}"
            )
        return Path(self.data_path) / "train_output" / self.dataset_name / sequence

    def gpcc_root(self, sequence: str) -> Path:
        return self.model_root(sequence) / "compression" / "gpcc"


HIFI4G = DatasetConfig(
    name="HiFi4G",
    data_path="/synology/rajrup/VideoGS",
    dataset_name="HiFi4G_Dataset",
    sequences=(
        "4K_Actor1_Greeting",
        "4K_Actor2_Dancing",
        "4K_Actor3_Violin",
        "4K_Actor4_Dancing",
        "4K_Actor5_Oil-paper_Umbrella",
        "4K_Actor6_Changing_Clothes",
        "4K_Actor7_Nunchaku",
    ),
    frame_id=0,
)

N3DV = DatasetConfig(
    name="N3DV",
    data_path="/synology/rajrup/Queen",
    dataset_name="Neural_3D_Video",
    sequences=(
        "cook_spinach",
        "coffee_martini",
        "cut_roasted_beef",
        "flame_salmon_1",
        "flame_steak",
        "sear_steak",
    ),
    frame_id=1,
)

ALL_DATASETS: list[DatasetConfig] = [HIFI4G, N3DV]


@dataclass
class GpccPoint:
    voxel_depth: int
    qp_rest: int
    qp_dc: int
    qp_opacity: int
    total_compressed_bytes: int
    decomp_psnr: float
    gt_psnr: float


def _load_point(config_dir: Path, frame_id: int) -> GpccPoint | None:
    m = PARAM_TAG_RE.match(config_dir.name)
    if not m:
        return None

    depth, qp_rest, qp_dc, qp_opacity = (
        int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)),
    )
    frame_dir = config_dir / f"frame{frame_id}"

    compressed_bytes: int | None = None
    try:
        with open(frame_dir / "benchmark_gpcc.csv", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if int(row["frame_idx"]) == frame_id:
                    compressed_bytes = int(row["total_compressed_bytes"])
                    break
    except (OSError, KeyError, ValueError):
        return None
    if compressed_bytes is None:
        return None

    decomp_psnr: float | None = None
    gt_psnr: float | None = None
    try:
        with open(
            frame_dir / "evaluation" / "evaluation_results.json", encoding="utf-8",
        ) as f:
            data = json.load(f)
        for fr in data.get("per_frame", []):
            if int(fr["frame"]) == frame_id:
                decomp_psnr = float(fr["decomp_psnr"])
                gt_psnr = float(fr["gt_psnr"])
                break
    except (OSError, json.JSONDecodeError, KeyError, ValueError):
        return None
    if decomp_psnr is None or gt_psnr is None:
        return None

    return GpccPoint(
        voxel_depth=depth,
        qp_rest=qp_rest,
        qp_dc=qp_dc,
        qp_opacity=qp_opacity,
        total_compressed_bytes=compressed_bytes,
        decomp_psnr=decomp_psnr,
        gt_psnr=gt_psnr,
    )


def _collect_points(gpcc_root: Path, frame_id: int) -> list[GpccPoint]:
    if not gpcc_root.is_dir():
        return []

    config_dirs = sorted(
        (gpcc_root / entry.name for entry in os.scandir(gpcc_root) if entry.is_dir()),
        key=lambda p: p.name,
    )
    if not config_dirs:
        return []

    points: list[GpccPoint] = []
    with ThreadPoolExecutor(max_workers=THREAD_WORKERS) as executor:
        futures = {
            executor.submit(_load_point, d, frame_id): d for d in config_dirs
        }
        for future in as_completed(futures):
            pt = future.result()
            if pt is not None:
                points.append(pt)
    return points


def _find_reference(points: list[GpccPoint]) -> GpccPoint | None:
    candidates = [
        p for p in points
        if p.qp_rest == 4 and p.qp_dc == 4 and p.qp_opacity == 4
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.voxel_depth)


def select_default(
    dataset_name: str,
    sequence: str,
    gpcc_root: Path,
    frame_id: int,
) -> dict[str, Any] | None:
    points = _collect_points(gpcc_root, frame_id)
    if not points:
        print(f"  [SKIP] {dataset_name}/{sequence}: no GPCC results found")
        return None

    ref = _find_reference(points)
    if ref is None:
        print(f"  [SKIP] {dataset_name}/{sequence}: no reference point (depth=max, QP=(4,4,4))")
        return None

    gt_psnr = ref.gt_psnr
    below_gt = [p for p in points if p.decomp_psnr < gt_psnr]

    if not below_gt:
        print(
            f"  [SKIP] {dataset_name}/{sequence}: all configs exceed GT PSNR "
            f"(ref={ref.decomp_psnr:.2f}, gt={gt_psnr:.2f})"
        )
        return None

    if ref.decomp_psnr >= gt_psnr:
        best = max(below_gt, key=lambda p: p.decomp_psnr)
    else:
        psnr_threshold = ref.decomp_psnr - PSNR_TOLERANCE
        candidates = [p for p in below_gt if p.decomp_psnr >= psnr_threshold]
        if not candidates:
            print(
                f"  [SKIP] {dataset_name}/{sequence}: no candidates within {PSNR_TOLERANCE} dB of reference "
                f"(ref={ref.decomp_psnr:.2f}, gt={gt_psnr:.2f})"
            )
            return None
        best = min(candidates, key=lambda p: p.total_compressed_bytes)

    print(
        f"  {dataset_name}/{sequence}: J={best.voxel_depth} rest={best.qp_rest} "
        f"dc={best.qp_dc} op={best.qp_opacity} | "
        f"size={best.total_compressed_bytes / (1024 * 1024):.2f} MB | "
        f"PSNR={best.decomp_psnr:.2f} (ref={ref.decomp_psnr:.2f}, gt={gt_psnr:.2f})"
    )

    return {
        "voxel_depth": best.voxel_depth,
        "qp_rest": best.qp_rest,
        "qp_dc": best.qp_dc,
        "qp_opacity": best.qp_opacity,
    }


def main() -> None:
    print("=" * 70)
    print("Select GPCC Default Parameters from R-D Sweep")
    print(f"  Tolerance: {PSNR_TOLERANCE} dB")
    print(f"  Datasets:  {', '.join(d.name for d in ALL_DATASETS)}")
    print(f"  Output:    {OUTPUT_JSON}")
    print("=" * 70)

    defaults: dict[str, dict[str, Any]] = {}
    for ds in ALL_DATASETS:
        print(f"\n--- {ds.name} (frame {ds.frame_id}) ---")
        for seq in ds.sequences:
            result = select_default(ds.name, seq, ds.gpcc_root(seq), ds.frame_id)
            if result is not None:
                defaults[seq] = result

    if not defaults:
        print("\n[ERROR] No defaults selected for any sequence.")
        sys.exit(1)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(defaults, f, indent=2)
    print(f"\nWrote {len(defaults)} sequence defaults to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
