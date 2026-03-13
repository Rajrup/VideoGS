#!/usr/bin/env python3
"""Temp script: measure GPCC encode/decode time using Pareto-selected configs.

Config selection replicates _load_gpcc_entry from
collect_and_plot_default_baselines.py (Pareto frontier matched to VideoGS qp=25
PSNR).

Timing excludes boundary file I/O:
  - Encode: excludes input PLY load, metadata JSON write, size-stat calculation.
  - Decode: excludes metadata JSON read, output PLY save, temp-dir cleanup.
Includes GPU warmup before any timed measurement.

Usage (videogs conda env with LiVoGS CUDA modules):
    python scripts/tmp_gpcc_timing.py
"""

from __future__ import annotations

import concurrent.futures
import csv
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from gpcc_baseline.compress_decompress_pipeline import (
    _build_decode_command,
    _build_encode_command,
    _read_attribute_ply,
    _resolve_input_ply,
    _run_tmc3_command,
    _voxelize_and_merge,
    _write_color_ply,
    _write_reflectance_ply,
    adaptive_normalize,
    convert_rgb2yuv,
    convert_yuv2rgb,
    load_videogs_ply,
    morton_order_sort,
    numpy_to_native,
    save_videogs_ply_from_arrays,
    uchar_to_float,
    uint16_to_float,
)

try:
    import torch
except ImportError:
    torch = None

HIFI4G_DATA_PATH = "/synology/rajrup/VideoGS"
N3DV_DATA_PATH = "/synology/rajrup/Queen"
VIDEOGS_GROUP_SIZE = 20

TMC3_PATH = "/home/haodongw/workspace/mpeg-pcc-tmc13/build/tmc3/tmc3"

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


@dataclass
class SeqConfig:
    name: str
    dataset: str
    model_root: str
    input_dir: str
    frame_ids: tuple[int, ...]
    gpcc_octree_depths: tuple[int, ...]


SEQUENCES: list[SeqConfig] = [
    SeqConfig(
        name="4K_Actor1_Greeting",
        dataset="HiFi4G",
        model_root=f"{HIFI4G_DATA_PATH}/train_output/HiFi4G_Dataset/4K_Actor1_Greeting",
        input_dir=f"{HIFI4G_DATA_PATH}/train_output/HiFi4G_Dataset/4K_Actor1_Greeting/checkpoint",
        frame_ids=(0,),
        gpcc_octree_depths=(12),
    ),
    SeqConfig(
        name="flame_salmon_1",
        dataset="N3DV",
        model_root=f"{N3DV_DATA_PATH}/pretrained_output/Neural_3D_Video/queen_compressed_flame_salmon_1",
        input_dir=f"{N3DV_DATA_PATH}/pretrained_output/Neural_3D_Video/queen_compressed_flame_salmon_1",
        frame_ids=(1,),
        gpcc_octree_depths=(17),
    ),
    SeqConfig(
        name="sear_steak",
        dataset="N3DV",
        model_root=f"{N3DV_DATA_PATH}/pretrained_output/Neural_3D_Video/queen_compressed_sear_steak",
        input_dir=f"{N3DV_DATA_PATH}/pretrained_output/Neural_3D_Video/queen_compressed_sear_steak",
        frame_ids=(1,),
        gpcc_octree_depths=(17),
    ),
]


def _cuda_sync() -> None:
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def load_videogs_target_psnr(model_root: str, frame_id: int) -> float | None:
    """Load VideoGS (qp=25) decomp_psnr — used as the GPCC matching target."""
    group_tag = f"frames_{frame_id}_{frame_id + VIDEOGS_GROUP_SIZE - 1}_int_1"
    eval_path = os.path.join(
        model_root, "compression", "videogs", "qp_25", group_tag,
        "evaluation", "evaluation_results.json",
    )
    if not os.path.isfile(eval_path):
        return None
    try:
        with open(eval_path, encoding="utf-8") as f:
            data = json.load(f)
        for fr in data.get("per_frame", []):
            if int(fr["frame"]) == frame_id:
                return float(fr["decomp_psnr"])
    except (OSError, json.JSONDecodeError, KeyError, ValueError):
        pass
    return None


def select_gpcc_pareto_config(
    model_root: str,
    frame_id: int,
    depths: tuple[int, ...],
    target_psnr: float,
) -> dict[str, Any] | None:
    """Replicate _load_gpcc_entry Pareto-frontier selection."""
    gpcc_root = os.path.join(model_root, "compression", "gpcc")
    candidates: list[dict[str, Any]] = []

    for depth in depths:
        for qp_rest, qp_dc, qp_opacity in GPCC_QP_COMBOS:
            params_tag = f"J{depth}_rest{qp_rest}_dc{qp_dc}_op{qp_opacity}"
            out_dir = os.path.join(gpcc_root, params_tag, f"frame{frame_id}")

            bench_path = os.path.join(out_dir, "benchmark_gpcc.csv")
            eval_path = os.path.join(out_dir, "evaluation", "evaluation_results.json")
            if not os.path.isfile(bench_path) or not os.path.isfile(eval_path):
                continue

            try:
                bench_data: dict[str, Any] | None = None
                with open(bench_path, newline="", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        if int(row["frame_idx"]) == frame_id:
                            bench_data = {
                                "encode_ms": float(row["encode_time_s"]) * 1000,
                                "decode_ms": float(row["decode_time_s"]) * 1000,
                                "compressed_bytes": int(row["total_compressed_bytes"]),
                            }
                            break
                if bench_data is None:
                    continue

                metrics: dict[str, float] | None = None
                with open(eval_path, encoding="utf-8") as f:
                    eval_data = json.load(f)
                for fr in eval_data.get("per_frame", []):
                    if int(fr["frame"]) == frame_id:
                        metrics = {
                            "decomp_psnr": float(fr["decomp_psnr"]),
                            "gt_psnr": float(fr["gt_psnr"]),
                        }
                        break
                if metrics is None:
                    continue

                candidates.append({
                    **bench_data,
                    "decomp_psnr": metrics["decomp_psnr"],
                    "gt_psnr": metrics["gt_psnr"],
                    "depth": depth,
                    "qp_rest": qp_rest,
                    "qp_dc": qp_dc,
                    "qp_opacity": qp_opacity,
                })
            except (OSError, KeyError, ValueError):
                continue

    if not candidates:
        return None

    # Pareto frontier: sorted by size, keep only points with increasing PSNR
    candidates.sort(key=lambda c: c["compressed_bytes"])
    frontier: list[dict[str, Any]] = [candidates[0]]
    for c in candidates[1:]:
        if c["decomp_psnr"] > frontier[-1]["decomp_psnr"]:
            frontier.append(c)

    # Pick the frontier point closest to the target PSNR
    return min(frontier, key=lambda c: abs(c["decomp_psnr"] - target_psnr))


def gpu_warmup(ply_path: str) -> dict[str, Any]:
    """Load PLY to GPU + run a short voxelization to warm up CUDA context and kernels."""
    print("  GPU warmup: loading PLY + voxelization ...")
    device = "cuda:0" if (torch is not None and torch.cuda.is_available()) else "cpu"
    params = load_videogs_ply(ply_path, device=device)
    _voxelize_and_merge(params, voxel_depth=8, device=0)
    _cuda_sync()
    print("  GPU warmup done.\n")
    return params


def timed_encode_gpcc(
    params: dict[str, Any],
    output_dir: str,
    qp_config: dict[str, int],
    tmc3_path: str,
    voxel_depth: int,
) -> dict[str, Any]:
    """encode_gpcc with timing that excludes input PLY load, metadata write, and size stats."""
    os.makedirs(output_dir, exist_ok=True)
    temp_encode_dir = os.path.join(output_dir, "temp_encode")
    attr_ply_dir = os.path.join(temp_encode_dir, "attribute_ply")
    os.makedirs(attr_ply_dir, exist_ok=True)
    compressed_dirs = {
        "opacity": os.path.join(temp_encode_dir, "opacity_compressed"),
        "dc": os.path.join(temp_encode_dir, "dc_compressed"),
        "rest": os.path.join(temp_encode_dir, "rest_compressed"),
        "scale": os.path.join(temp_encode_dir, "scale_compressed"),
        "rot": os.path.join(temp_encode_dir, "rot_compressed"),
    }
    for d in compressed_dirs.values():
        os.makedirs(d, exist_ok=True)

    device_id = 0

    _cuda_sync()
    t0 = time.perf_counter()

    merged = _voxelize_and_merge(params, voxel_depth=voxel_depth, device=device_id)
    _cuda_sync()

    xyz = merged["voxel_xyz"]
    colors = merged["colors"]
    n_rest_channels = int(colors.shape[1] - 3)
    opacity = merged["opacity"]
    scales = merged["scales"]
    quats = merged["quats"]

    metadata: dict[str, Any] = {
        "Geometry": {"vmin": merged["vmin"], "voxel_size": merged["voxel_size"], "voxel_depth": voxel_depth},
        "Attribute": {},
        "files": {"opacity": [], "dc": [], "rest": [], "scale": [], "rot": []},
    }
    compression_jobs: list[tuple[str, str, str, int]] = []

    opacity_u16, mn, mx = adaptive_normalize(opacity, np.uint16)
    metadata["Attribute"]["opacity"] = {"min": mn, "max": mx}
    opacity_ply = os.path.join(attr_ply_dir, "opacity.ply")
    _write_reflectance_ply(opacity_ply, xyz, opacity_u16)
    opacity_bin = os.path.join(compressed_dirs["opacity"], "opacity.bin")
    metadata["files"]["opacity"].append("opacity.bin")
    compression_jobs.append(("opacity", opacity_ply, opacity_bin, qp_config["qp_opacity"]))

    dc_rgb = colors[:, :3]
    dc_yuv = convert_rgb2yuv(dc_rgb)
    dc_u8 = []
    for i in range(3):
        arr_u8, mn, mx = adaptive_normalize(dc_yuv[:, i], np.uint8)
        dc_u8.append(arr_u8)
        metadata["Attribute"][f"f_dc_{i}"] = {"min": mn, "max": mx}
    dc_ply = os.path.join(attr_ply_dir, "dc.ply")
    _write_color_ply(dc_ply, xyz, dc_u8[0], dc_u8[1], dc_u8[2])
    dc_bin = os.path.join(compressed_dirs["dc"], "dc.bin")
    metadata["files"]["dc"].append("dc.bin")
    compression_jobs.append(("dc", dc_ply, dc_bin, qp_config["qp_dc"]))

    rest_rgb = colors[:, 3 : 3 + n_rest_channels]
    for i in range(0, n_rest_channels, 3):
        rest_triplet_yuv = convert_rgb2yuv(rest_rgb[:, i : i + 3])
        c_u8 = []
        for c in range(3):
            arr_u8, mn, mx = adaptive_normalize(rest_triplet_yuv[:, c], np.uint8)
            c_u8.append(arr_u8)
            metadata["Attribute"][f"f_rest_{i + c}"] = {"min": mn, "max": mx}
        name = f"rest_{i:02d}_{i+1:02d}_{i+2:02d}"
        rest_ply = os.path.join(attr_ply_dir, f"{name}.ply")
        _write_color_ply(rest_ply, xyz, c_u8[0], c_u8[1], c_u8[2])
        rest_bin = os.path.join(compressed_dirs["rest"], f"{name}.bin")
        metadata["files"]["rest"].append(f"{name}.bin")
        compression_jobs.append(("rest", rest_ply, rest_bin, qp_config["qp_rest"]))

    for i in range(3):
        scale_u16, mn, mx = adaptive_normalize(scales[:, i], np.uint16)
        metadata["Attribute"][f"scale_{i}"] = {"min": mn, "max": mx}
        name = f"scale_{i}"
        scale_ply = os.path.join(attr_ply_dir, f"{name}.ply")
        _write_reflectance_ply(scale_ply, xyz, scale_u16)
        scale_bin = os.path.join(compressed_dirs["scale"], f"{name}.bin")
        metadata["files"]["scale"].append(f"{name}.bin")
        compression_jobs.append(("scale", scale_ply, scale_bin, 4))

    for i in range(4):
        rot_u16, mn, mx = adaptive_normalize(quats[:, i], np.uint16)
        metadata["Attribute"][f"rot_{i}"] = {"min": mn, "max": mx}
        name = f"rot_{i}"
        rot_ply = os.path.join(attr_ply_dir, f"{name}.ply")
        _write_reflectance_ply(rot_ply, xyz, rot_u16)
        rot_bin = os.path.join(compressed_dirs["rot"], f"{name}.bin")
        metadata["files"]["rot"].append(f"{name}.bin")
        compression_jobs.append(("rot", rot_ply, rot_bin, 4))

    futures = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for attr_type, input_ply, output_bin, qp in compression_jobs:
            command = _build_encode_command(
                tmc3_path=tmc3_path, input_ply=input_ply, output_bin=output_bin,
                attr_type=attr_type, qp=qp, voxel_depth=voxel_depth,
            )
            futures.append(executor.submit(_run_tmc3_command, command))
        for future in concurrent.futures.as_completed(futures):
            future.result()

    encode_time = time.perf_counter() - t0

    shutil.rmtree(attr_ply_dir)

    metadata_json_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=numpy_to_native)

    total_bytes = 0
    for cat in compressed_dirs:
        for n in metadata["files"][cat]:
            total_bytes += os.path.getsize(os.path.join(compressed_dirs[cat], n))

    return {
        "total_compressed_bytes": total_bytes,
        "num_points_input": int(params["means"].shape[0]),
        "num_points_voxelized": int(xyz.shape[0]),
        "encode_time_s": encode_time,
        "compressed_dir": temp_encode_dir,
        "metadata_json_path": metadata_json_path,
    }


def timed_decode_gpcc(
    compressed_dir: str,
    output_ply_path: str,
    metadata_json_path: str,
    tmc3_path: str,
) -> dict[str, Any]:
    """decode_gpcc with timing that excludes metadata read, output PLY save, and cleanup."""
    with open(metadata_json_path, "r", encoding="utf-8") as f:
        metadata: dict[str, Any] = json.load(f)

    temp_decode_dir = os.path.join(compressed_dir, "temp_decode")
    if os.path.exists(temp_decode_dir):
        shutil.rmtree(temp_decode_dir, ignore_errors=True)
    os.makedirs(temp_decode_dir, exist_ok=True)

    category_dirs = {
        "opacity": os.path.join(compressed_dir, "opacity_compressed"),
        "dc": os.path.join(compressed_dir, "dc_compressed"),
        "rest": os.path.join(compressed_dir, "rest_compressed"),
        "scale": os.path.join(compressed_dir, "scale_compressed"),
        "rot": os.path.join(compressed_dir, "rot_compressed"),
    }

    decomp_ply_paths: dict[str, list[str]] = {
        "opacity": [], "dc": [], "rest": [], "scale": [], "rot": [],
    }
    decode_jobs: list[list[str]] = []
    for category in ["opacity", "dc", "rest", "scale", "rot"]:
        out_cat_dir = os.path.join(temp_decode_dir, f"{category}_decompressed")
        os.makedirs(out_cat_dir, exist_ok=True)
        for bin_name in metadata["files"][category]:
            input_bin = os.path.join(category_dirs[category], bin_name)
            output_ply = os.path.join(out_cat_dir, os.path.splitext(bin_name)[0] + ".ply")
            decode_jobs.append(_build_decode_command(tmc3_path, input_bin, output_ply))
            decomp_ply_paths[category].append(output_ply)

    t0 = time.perf_counter()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(_run_tmc3_command, cmd) for cmd in decode_jobs]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    first_ply = decomp_ply_paths["opacity"][0]
    ref_points, _, _ = _read_attribute_ply(first_ply)
    points_ref, _ = morton_order_sort(ref_points)
    n_points = points_ref.shape[0]

    geom_meta = metadata.get("Geometry", {})
    vmin = np.asarray(geom_meta.get("vmin", [0.0, 0.0, 0.0]), dtype=np.float32)
    voxel_size = float(geom_meta.get("voxel_size", 1.0))
    points_world = (points_ref.astype(np.float32) + 0.5) * voxel_size + vmin.reshape(1, 3)

    def _read_sorted_reflectance(ply_path: str) -> np.ndarray:
        points, _, refl = _read_attribute_ply(ply_path)
        if refl is None:
            raise ValueError(f"Missing reflectance: {ply_path}")
        _, idx = morton_order_sort(points)
        return refl[idx].astype(np.float32)

    def _read_sorted_colors(ply_path: str) -> np.ndarray:
        points, clrs, _ = _read_attribute_ply(ply_path)
        if clrs is None:
            raise ValueError(f"Missing color: {ply_path}")
        _, idx = morton_order_sort(points)
        return clrs[idx].astype(np.float32)

    opacity_denorm = uint16_to_float(
        _read_sorted_reflectance(decomp_ply_paths["opacity"][0]),
        metadata["Attribute"]["opacity"]["min"],
        metadata["Attribute"]["opacity"]["max"],
    )

    dc_sorted = _read_sorted_colors(decomp_ply_paths["dc"][0])
    y0 = uchar_to_float(dc_sorted[:, 0], metadata["Attribute"]["f_dc_0"]["min"], metadata["Attribute"]["f_dc_0"]["max"])
    y1 = uchar_to_float(dc_sorted[:, 1], metadata["Attribute"]["f_dc_1"]["min"], metadata["Attribute"]["f_dc_1"]["max"])
    y2 = uchar_to_float(dc_sorted[:, 2], metadata["Attribute"]["f_dc_2"]["min"], metadata["Attribute"]["f_dc_2"]["max"])
    dc_rgb = convert_yuv2rgb(np.stack([y0, y1, y2], axis=1)).astype(np.float32)

    rest_attr_indices = sorted(
        int(k.split("_")[-1]) for k in metadata["Attribute"] if k.startswith("f_rest_")
    )
    rest_rgb_channels: list[np.ndarray] = []
    for rest_idx, rest_ply in enumerate(decomp_ply_paths["rest"]):
        rest_sorted = _read_sorted_colors(rest_ply)
        idx_base = rest_idx * 3
        k0 = f"f_rest_{rest_attr_indices[idx_base]}"
        k1 = f"f_rest_{rest_attr_indices[idx_base + 1]}"
        k2 = f"f_rest_{rest_attr_indices[idx_base + 2]}"
        r0 = uchar_to_float(rest_sorted[:, 0], metadata["Attribute"][k0]["min"], metadata["Attribute"][k0]["max"])
        r1 = uchar_to_float(rest_sorted[:, 1], metadata["Attribute"][k1]["min"], metadata["Attribute"][k1]["max"])
        r2 = uchar_to_float(rest_sorted[:, 2], metadata["Attribute"][k2]["min"], metadata["Attribute"][k2]["max"])
        triplet = convert_yuv2rgb(np.stack([r0, r1, r2], axis=1)).astype(np.float32)
        rest_rgb_channels.extend([triplet[:, 0], triplet[:, 1], triplet[:, 2]])

    scales_arr = np.zeros((n_points, 3), dtype=np.float32)
    for i, scale_ply in enumerate(decomp_ply_paths["scale"]):
        scales_arr[:, i] = uint16_to_float(
            _read_sorted_reflectance(scale_ply),
            metadata["Attribute"][f"scale_{i}"]["min"],
            metadata["Attribute"][f"scale_{i}"]["max"],
        )

    quats_arr = np.zeros((n_points, 4), dtype=np.float32)
    for rot_ply in decomp_ply_paths["rot"]:
        stem = os.path.splitext(os.path.basename(rot_ply))[0]
        comp_name = stem.replace("_dec", "")
        comp_idx = int(comp_name.split("_")[-1])
        quats_arr[:, comp_idx] = uint16_to_float(
            _read_sorted_reflectance(rot_ply),
            metadata["Attribute"][f"rot_{comp_idx}"]["min"],
            metadata["Attribute"][f"rot_{comp_idx}"]["max"],
        )

    quat_norm = np.linalg.norm(quats_arr, axis=1, keepdims=True)
    quats_arr = quats_arr / np.maximum(quat_norm, 1e-12)

    if rest_rgb_channels:
        rest_rgb = np.stack(rest_rgb_channels, axis=1).astype(np.float32)
        colors_out = np.concatenate([dc_rgb, rest_rgb], axis=1).astype(np.float32)
    else:
        colors_out = dc_rgb.astype(np.float32)

    decode_time = time.perf_counter() - t0

    save_videogs_ply_from_arrays(
        means=points_world,
        colors=colors_out,
        opacity=opacity_denorm.astype(np.float32),
        scales=scales_arr.astype(np.float32),
        quats=quats_arr.astype(np.float32),
        output_path=output_ply_path,
    )

    shutil.rmtree(temp_decode_dir, ignore_errors=True)
    return {"decode_time_s": decode_time, "num_points_output": int(n_points)}


def main() -> None:
    sep = "=" * 70
    print(sep)
    print("GPCC Baseline  --  Encoding / Decoding Time Measurement")
    print("  (file I/O excluded from timing, GPU warmed up)")
    print(f"  Config selection: Pareto frontier matched to VideoGS (qp=25) PSNR")
    print(f"  TMC3 path: {TMC3_PATH}")
    print(sep)

    first_ply_path: str | None = None
    for seq in SEQUENCES:
        for fid in seq.frame_ids:
            resolved = _resolve_input_ply(seq.input_dir, fid)
            if resolved is not None:
                first_ply_path = resolved
                break
        if first_ply_path is not None:
            break
    if first_ply_path is None:
        print("[ERROR] Could not resolve any input PLY for warmup — aborting")
        return

    warmup_params = gpu_warmup(first_ply_path)
    warmup_used = False

    results: list[dict[str, Any]] = []

    for seq in SEQUENCES:
        print(f"\n{'_' * 60}")
        print(f"Sequence: {seq.name}  ({seq.dataset}),  frames={list(seq.frame_ids)}")
        print(f"  Model root:    {seq.model_root}")
        print(f"  Octree depths: {seq.gpcc_octree_depths}")

        for frame_id in seq.frame_ids:
            print(f"\n  --- frame {frame_id} ---")

            target_psnr = load_videogs_target_psnr(seq.model_root, frame_id)
            if target_psnr is None:
                print("  [ERROR] Could not load VideoGS target PSNR -- skipping")
                continue
            print(f"  VideoGS target PSNR: {target_psnr:.2f} dB")

            selected = select_gpcc_pareto_config(
                seq.model_root, frame_id, seq.gpcc_octree_depths, target_psnr,
            )
            if selected is None:
                print("  [ERROR] No valid GPCC candidates found -- skipping")
                continue

            depth = selected["depth"]
            qp_rest = selected["qp_rest"]
            qp_dc = selected["qp_dc"]
            qp_opacity = selected["qp_opacity"]

            print(f"  Selected config (Pareto-matched to VideoGS PSNR):")
            print(f"    depth (J)  = {depth}")
            print(f"    qp_rest    = {qp_rest}")
            print(f"    qp_dc      = {qp_dc}")
            print(f"    qp_opacity = {qp_opacity}")
            print(
                f"    -> PSNR = {selected['decomp_psnr']:.2f} dB,  "
                f"size = {selected['compressed_bytes']} B "
                f"({selected['compressed_bytes'] / (1024 * 1024):.2f} MB)"
            )
            print(
                f"    -> Previous timing (from CSV): "
                f"enc = {selected['encode_ms']:.1f} ms,  "
                f"dec = {selected['decode_ms']:.1f} ms"
            )

            ply_path = _resolve_input_ply(seq.input_dir, frame_id)
            if ply_path is None:
                print("  [ERROR] Could not resolve input PLY -- skipping")
                continue
            print(f"  Input PLY: {ply_path}")

            print("  Pre-loading PLY to GPU (excluded from timing) ...")
            if not warmup_used and ply_path == first_ply_path:
                params = warmup_params
                warmup_used = True
            else:
                device = "cuda:0" if (torch is not None and torch.cuda.is_available()) else "cpu"
                params = load_videogs_ply(ply_path, device=device)
                _cuda_sync()

            tmp_dir = tempfile.mkdtemp(prefix=f"gpcc_timing_{seq.name}_f{frame_id}_")
            output_dir = os.path.join(tmp_dir, "compressed")
            output_ply_path = os.path.join(tmp_dir, "decoded", "point_cloud.ply")

            qp_config = {
                "qp_rest": qp_rest,
                "qp_dc": qp_dc,
                "qp_opacity": qp_opacity,
            }

            try:
                print("\n  Running encode (timed: voxelize + attr prep + TMC3 encode) ...")
                enc = timed_encode_gpcc(
                    params=params,
                    output_dir=output_dir,
                    qp_config=qp_config,
                    tmc3_path=TMC3_PATH,
                    voxel_depth=depth,
                )
                encode_s = enc["encode_time_s"]
                comp_bytes = int(enc["total_compressed_bytes"])
                print(f"  Encode time: {encode_s:.3f} s  ({encode_s * 1000:.1f} ms)")
                print(
                    f"  Compressed size: {comp_bytes} B  "
                    f"({comp_bytes / (1024 * 1024):.2f} MB)"
                )

                print("  Running decode (timed: TMC3 decode + attr reconstruction) ...")
                dec = timed_decode_gpcc(
                    compressed_dir=enc["compressed_dir"],
                    output_ply_path=output_ply_path,
                    metadata_json_path=enc["metadata_json_path"],
                    tmc3_path=TMC3_PATH,
                )
                decode_s = dec["decode_time_s"]
                print(f"  Decode time: {decode_s:.3f} s  ({decode_s * 1000:.1f} ms)")

                results.append({
                    "sequence": seq.name,
                    "dataset": seq.dataset,
                    "frame_id": frame_id,
                    "depth": depth,
                    "qp_rest": qp_rest,
                    "qp_dc": qp_dc,
                    "qp_opacity": qp_opacity,
                    "encode_s": encode_s,
                    "decode_s": decode_s,
                    "compressed_bytes": comp_bytes,
                    "decomp_psnr": selected["decomp_psnr"],
                })

            except Exception as exc:
                print(f"  [ERROR] {exc}")
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\n\n{sep}")
    print("SUMMARY")
    print(sep)

    if not results:
        print("No successful measurements.")
        return

    header = (
        f"{'Sequence':<28s} {'Frame':>5s} {'Depth':>5s} {'qp_rest':>7s} {'qp_dc':>5s} "
        f"{'qp_op':>5s} {'Enc(ms)':>9s} {'Dec(ms)':>9s} "
        f"{'Size(MB)':>9s} {'PSNR':>7s}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['sequence']:<28s} "
            f"{r['frame_id']:>5d} "
            f"{r['depth']:>5d} {r['qp_rest']:>7d} {r['qp_dc']:>5d} {r['qp_opacity']:>5d} "
            f"{r['encode_s'] * 1000:>9.1f} {r['decode_s'] * 1000:>9.1f} "
            f"{r['compressed_bytes'] / (1024 * 1024):>9.2f} "
            f"{r['decomp_psnr']:>7.2f}"
        )

    avg_enc = sum(r["encode_s"] for r in results) / len(results)
    avg_dec = sum(r["decode_s"] for r in results) / len(results)
    print(f"\nAverage encode: {avg_enc * 1000:.1f} ms")
    print(f"Average decode: {avg_dec * 1000:.1f} ms")
    print(sep)


if __name__ == "__main__":
    main()
