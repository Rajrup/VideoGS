#!/usr/bin/env python3
"""
MesonGS Compression + Decompression for VideoGS-trained Gaussian Splat Models.

For each frame:
  1. Load PLY from VideoGS checkpoint into MesonGS GaussianModel
  2. Compute importance via cal_imp() (renders from train cameras)
  3. Optional: prune low-importance Gaussians
  4. encode_mesongs(): Octree → VQ → RAHT → Block Quantize → LZ77
  5. decode_mesongs(): LZ77 → Dequant → iRAHT → VQ lookup → Octree decode
  6. Convert Euler angles → quaternions, save as VideoGS-compatible PLY

Must be run from the MesonGS directory in the mesongs conda environment.
"""

import os
import sys
import csv
import json
import time
import argparse
import re
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from typing import Any

# --- sys.path setup: MesonGS root must be on path (for raht_torch etc.) ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_VIDEOGS_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_MESONGS_ROOT = os.path.join(_VIDEOGS_ROOT, "MesonGS")
if _MESONGS_ROOT not in sys.path:
    sys.path.insert(0, _MESONGS_ROOT)

from scene import GaussianModel
from scene.cameras import Camera
from scene.colmap_loader import rotmat2qvec, qvec2rotmat
from scene.dataset_readers import CameraInfo, getNerfppNorm
from arguments import OptimizationParams
from utils.general_utils import PILtoTorch
from utils.graphics_utils import focal2fov
from mesongs import cal_imp, prune_mask
from compression.compress_decompress import encode_mesongs, decode_mesongs
from compression.utils import euler_to_quaternion

# ---------------------------------------------------------------------------
# HiFi4G Config (same structure as universal_config in mesongs.py)
# ---------------------------------------------------------------------------
DEFAULT_DEPTH = 12
DEFAULT_NUM_BITS = 8
DEFAULT_N_BLOCK = 57
DEFAULT_CODEBOOK_SIZE = 2048
DEFAULT_PRUNE_PERCENT = 0.0

_FRAME_SPAN_TAG_RE = re.compile(r"^frames_\d+_\d+_int_\d+$")
_FRAME_DIR_TAG_RE = re.compile(r"^frame\d+$")

hifi4g_config = {
    'prune': {
        '4K_Actor1_Greeting': DEFAULT_PRUNE_PERCENT,
        '4K_Actor2_Dancing': DEFAULT_PRUNE_PERCENT,
        '4K_Actor3_Violin': DEFAULT_PRUNE_PERCENT,
        '4K_Actor4_Dancing': DEFAULT_PRUNE_PERCENT,
        '4K_Actor5_Oil-paper_Umbrella': DEFAULT_PRUNE_PERCENT,
        '4K_Actor6_Changing_Clothes': DEFAULT_PRUNE_PERCENT,
        '4K_Actor7_Nunchaku': DEFAULT_PRUNE_PERCENT,
    },
    'depth': {
        '4K_Actor1_Greeting': DEFAULT_DEPTH,
        '4K_Actor2_Dancing': DEFAULT_DEPTH,
        '4K_Actor3_Violin': DEFAULT_DEPTH,
        '4K_Actor4_Dancing': DEFAULT_DEPTH,
        '4K_Actor5_Oil-paper_Umbrella': DEFAULT_DEPTH,
        '4K_Actor6_Changing_Clothes': DEFAULT_DEPTH,
        '4K_Actor7_Nunchaku': DEFAULT_DEPTH,
    },
    'n_block': {
        '4K_Actor1_Greeting': DEFAULT_N_BLOCK,
        '4K_Actor2_Dancing': DEFAULT_N_BLOCK,
        '4K_Actor3_Violin': DEFAULT_N_BLOCK,
        '4K_Actor4_Dancing': DEFAULT_N_BLOCK,
        '4K_Actor5_Oil-paper_Umbrella': DEFAULT_N_BLOCK,
        '4K_Actor6_Changing_Clothes': DEFAULT_N_BLOCK,
        '4K_Actor7_Nunchaku': DEFAULT_N_BLOCK,
    },
    'cb': {
        '4K_Actor1_Greeting': DEFAULT_CODEBOOK_SIZE,
        '4K_Actor2_Dancing': DEFAULT_CODEBOOK_SIZE,
        '4K_Actor3_Violin': DEFAULT_CODEBOOK_SIZE,
        '4K_Actor4_Dancing': DEFAULT_CODEBOOK_SIZE,
        '4K_Actor5_Oil-paper_Umbrella': DEFAULT_CODEBOOK_SIZE,
        '4K_Actor6_Changing_Clothes': DEFAULT_CODEBOOK_SIZE,
        '4K_Actor7_Nunchaku': DEFAULT_CODEBOOK_SIZE,
    },
    'num_bits': {
        '4K_Actor1_Greeting': DEFAULT_NUM_BITS,
        '4K_Actor2_Dancing': DEFAULT_NUM_BITS,
        '4K_Actor3_Violin': DEFAULT_NUM_BITS,
        '4K_Actor4_Dancing': DEFAULT_NUM_BITS,
        '4K_Actor5_Oil-paper_Umbrella': DEFAULT_NUM_BITS,
        '4K_Actor6_Changing_Clothes': DEFAULT_NUM_BITS,
        '4K_Actor7_Nunchaku': DEFAULT_NUM_BITS,
    }
}

# ---------------------------------------------------------------------------
# Camera Loading (from HiFi4G preprocessed transforms.json)
# ---------------------------------------------------------------------------

def load_train_cameras(dataset_path, first_frame, resolution, llffhold=8):
    """Load train cameras from the first frame's transforms.json.

    Train cameras = all cameras where idx % llffhold != 0.
    Transform parsing follows evaluate_decompress.py exactly.
    Returns (cameras, cam_infos_for_norm) where cam_infos_for_norm
    is a list of CameraInfo for getNerfppNorm.
    """
    frame_path = os.path.join(dataset_path, str(first_frame))
    with open(os.path.join(frame_path, "transforms.json")) as f:
        contents = json.load(f)
    frames = contents["frames"]

    all_indices = list(range(len(frames)))
    train_indices = [idx for idx in all_indices if idx % llffhold != 0]

    print(f"Total cameras: {len(frames)}, Train cameras (llffhold={llffhold}): {len(train_indices)}")

    flip_mat = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])

    cameras = []
    cam_infos = []

    for uid, idx in enumerate(train_indices):
        entry = frames[idx]
        cam_name = entry["file_path"]

        matrix = np.linalg.inv(np.matmul(np.array(entry["transform_matrix"]), flip_mat))
        R = np.transpose(qvec2rotmat(-rotmat2qvec(matrix[:3, :3])))
        T = matrix[:3, 3]

        image_path = os.path.join(frame_path, cam_name)
        image_pil = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image_pil.size

        fx = entry["fl_x"]
        fy = entry["fl_y"]
        FovY = focal2fov(fy, orig_h)
        FovX = focal2fov(fx, orig_w)

        if resolution in [1, 2, 4, 8]:
            target_res = (round(orig_w / resolution), round(orig_h / resolution))
        else:
            target_res = (orig_w, orig_h)

        resized_image = PILtoTorch(image_pil, target_res)
        gt_image = resized_image[:3, ...]

        cam = Camera(
            colmap_id=idx, R=R, T=T,
            FoVx=FovX, FoVy=FovY,
            image=gt_image, gt_alpha_mask=None,
            image_name=Path(cam_name).stem, uid=uid,
        )
        cameras.append(cam)

        cam_infos.append(CameraInfo(
            uid=uid, R=R, T=T, FovY=FovY, FovX=FovX,
            image=image_pil, image_path=image_path,
            image_name=Path(cam_name).stem, width=orig_w, height=orig_h
        ))

    return cameras, cam_infos

# ---------------------------------------------------------------------------
# PLY utilities
# ---------------------------------------------------------------------------

def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder) if "iteration_" in fname]
    return max(saved_iters)


def compute_uncompressed_size(gaussians, sh_degree=3):
    """Compute uncompressed size in bytes from GaussianModel attributes (float32)."""
    N = gaussians.get_xyz.shape[0]
    n_sh_rest = 3 * ((sh_degree + 1) ** 2 - 1)  # 45 for sh_degree=3
    n_floats_per_point = 3 + 3 + n_sh_rest + 1 + 3 + 4  # xyz + f_dc + f_rest + opacity + scale + rot
    return N * n_floats_per_point * 4  # float32 = 4 bytes


def save_decoded_ply(decoded_gaussians, output_path):
    """Convert decoded model (Euler angles) to quaternions and save as PLY.

    Uses MesonGS's build_rotation_from_euler for the rotation matrix,
    then our rotation_matrix_to_quaternion for the quaternion.
    Finally calls the author's save_ply().
    """
    with torch.no_grad():
        quats = euler_to_quaternion(decoded_gaussians._euler.detach())
        decoded_gaussians._rotation = nn.Parameter(quats, requires_grad=False)
    decoded_gaussians.save_ply(output_path)


def resolve_config_root(output_folder):
    output_path = Path(output_folder)
    if _FRAME_SPAN_TAG_RE.match(output_path.name) or _FRAME_DIR_TAG_RE.match(output_path.name):
        return output_path.parent
    return output_path


def write_single_frame_benchmark_csv(csv_path: Path, benchmark_row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "frame_id",
            "total_encode_ms",
            "total_decode_ms",
            "original_points",
            "after_prune_points",
            "after_octree_points",
            "decoded_points",
            "uncompressed_size_bytes",
            "compressed_size_bytes",
        ])
        w.writerow([
            int(benchmark_row["frame"]),
            f"{float(benchmark_row['total_encode_ms']):.2f}",
            f"{float(benchmark_row['total_decode_ms']):.2f}",
            int(benchmark_row["original_points"]),
            int(benchmark_row["after_prune_points"]),
            int(benchmark_row["after_octree_points"]),
            int(benchmark_row["decoded_points"]),
            int(benchmark_row["uncompressed_size_bytes"]),
            int(benchmark_row["compressed_size_bytes"]),
        ])


def write_single_frame_config_json(config_path: Path, config_template: dict[str, Any], frame_id: int) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    out = dict(config_template)
    out["frame_list"] = [int(frame_id)]
    out["frame_id"] = int(frame_id)
    with config_path.open("w") as f:
        json.dump(out, f, indent=4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MesonGS compress + decompress for VideoGS-trained models"
    )
    parser.add_argument("--ply_path", type=str, required=True,
                        help="Path to checkpoint dir containing frame folders (0, 1, ...)")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to preprocessed dataset (containing frame folders with transforms.json)")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Folder for benchmark CSV and metadata")
    parser.add_argument("--output_ply_folder", type=str, default=None,
                        help="Folder for decompressed PLY output (omit to skip saving)")
    parser.add_argument("--frame_start", type=int, default=0)
    parser.add_argument("--frame_end", type=int, default=200)
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--frame_ids", type=str, default=None,
                        help="Comma-separated frame IDs (overrides --frame_start/--frame_end/--interval)")
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--resolution", type=int, default=2,
                        help="Resolution scale for camera images (1, 2, 4, 8)")
    parser.add_argument("--scene_name", type=str, required=True,
                        help="HiFi4G sequence name (e.g. 4K_Actor1_Greeting)")
    parser.add_argument("--save_bitstreams", action="store_true",
                        help="Write .npz bitstreams and .zip to disk per frame")
    parser.add_argument("--white_background", action="store_true")

    # MesonGS hyperparameters (override config defaults)
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH, help="Octree depth (default: from config)")
    parser.add_argument("--n_block", type=int, default=DEFAULT_N_BLOCK, help="Block quantization count (default: from config)")
    parser.add_argument("--codebook_size", type=int, default=DEFAULT_CODEBOOK_SIZE, help="VQ codebook size (default: from config)")
    parser.add_argument("--prune", action="store_true", help="Enable pruning before compression")
    parser.add_argument("--prune_percent", type=float, default=DEFAULT_PRUNE_PERCENT,
                        help="Prune fraction (default: from config)")
    
    # MesonGS defaults that rarely change
    parser.add_argument("--oct_merge", type=str, default="mean", choices=["mean", "imp", "rand"])
    parser.add_argument("--batch_size", type=int, default=262144)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--num_bits", type=int, default=DEFAULT_NUM_BITS)

    args = parser.parse_args()

    config_root = resolve_config_root(args.output_folder)
    # --- Load config (from hifi4g_config, overridable by CLI) ---
    scene = args.scene_name
    if scene not in hifi4g_config['depth']:
        print(f"Warning: scene '{scene}' not in hifi4g_config, using defaults")

    depth = args.depth if args.depth is not None else hifi4g_config['depth'].get(scene, DEFAULT_DEPTH)
    n_block = args.n_block if args.n_block is not None else hifi4g_config['n_block'].get(scene, DEFAULT_N_BLOCK)
    codebook_size = args.codebook_size if args.codebook_size is not None else hifi4g_config['cb'].get(scene, DEFAULT_CODEBOOK_SIZE)
    prune_percent = args.prune_percent if args.prune_percent is not None else hifi4g_config['prune'].get(scene, DEFAULT_PRUNE_PERCENT)
    num_bits = args.num_bits if args.num_bits is not None else hifi4g_config['num_bits'].get(scene, DEFAULT_NUM_BITS)

    # --- Build dataset_args (SimpleNamespace matching what encode/decode_mesongs expects) ---
    from types import SimpleNamespace
    dataset_args = SimpleNamespace(
        sh_degree=args.sh_degree,
        depth=depth,
        num_bits=num_bits,
        oct_merge=args.oct_merge,
        raht=True,
        per_block_quant=True,
        per_channel_quant=False,
        n_block=n_block,
        codebook_size=codebook_size,
        batch_size=args.batch_size,
        steps=args.steps,
        percent=prune_percent,
        white_background=args.white_background,
    )

    pipe_args = SimpleNamespace(
        convert_SHs_python=True,
        compute_cov3D_python=False,
        scene_imp=scene,
        debug=False,
    )

    os.makedirs(args.output_folder, exist_ok=True)
    if args.output_ply_folder is not None:
        os.makedirs(args.output_ply_folder, exist_ok=True)

    # --- Print configuration ---
    print("=" * 70)
    print("MesonGS Compress + Decompress Pipeline")
    print("=" * 70)
    print(f"  PLY path:           {args.ply_path}")
    print(f"  Dataset path:       {args.dataset_path}")
    print(f"  Output folder:      {args.output_folder}")
    print(f"  Output PLY folder:  {args.output_ply_folder or '(skip)'}")
    if args.frame_ids is not None:
        frame_list = sorted(int(x.strip()) for x in args.frame_ids.split(","))
        print(f"  Frames:             {frame_list}")
    else:
        frame_list = list(range(args.frame_start, args.frame_end, args.interval))
        print(f"  Frames:             {args.frame_start} to {args.frame_end} (interval={args.interval})")
    print(f"  Scene:              {scene}")
    print(f"  SH degree:          {args.sh_degree}")
    print(f"  Resolution:         {args.resolution}")
    print(f"  Octree depth:       {depth}")
    print(f"  N block:            {n_block}")
    print(f"  Codebook size:      {codebook_size}")
    print(f"  Pruning:            {args.prune} (percent={prune_percent})")
    print(f"  Oct merge:          {args.oct_merge}")
    print(f"  VQ steps:           {args.steps}")
    print(f"  Save bitstreams:    {args.save_bitstreams}")
    print("=" * 70)

    # --- Load train cameras once (from first frame) ---
    print("\nLoading train cameras...")
    first_frame = frame_list[0]
    train_cameras, cam_infos = load_train_cameras(
        args.dataset_path, first_frame, args.resolution
    )
    nerf_norm = getNerfppNorm(cam_infos)
    cameras_extent = nerf_norm["radius"]
    print(f"Loaded {len(train_cameras)} train cameras, cameras_extent={cameras_extent:.4f}")

    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # --- Setup OptimizationParams for training_setup (needed if --prune) ---
    opt_parser = argparse.ArgumentParser()
    op = OptimizationParams(opt_parser)
    opt_args = op.extract(opt_parser.parse_args([]))

    # --- Per-frame loop ---
    benchmark_rows = []

    for frame in tqdm(frame_list, desc="Frames"):

        # --- 1. Load PLY ---
        ckpt_path = os.path.join(args.ply_path, str(frame), "point_cloud")
        if not os.path.exists(ckpt_path):
            print(f"Warning: Checkpoint not found: {ckpt_path}, skipping frame {frame}")
            continue
        max_iter = searchForMaxIteration(ckpt_path)
        ply_file_path = os.path.join(ckpt_path, f"iteration_{max_iter}", "point_cloud.ply")

        gaussians = GaussianModel(args.sh_degree, depth=depth, num_bits=num_bits)
        gaussians.load_ply(ply_file_path, og_number_points=-1, spatial_lr_scale=cameras_extent)
        N_original = gaussians.get_xyz.shape[0]
        uncompressed_size_bytes = compute_uncompressed_size(gaussians, args.sh_degree)

        # --- 2. Setup optimizer if pruning ---
        if args.prune:
            gaussians.training_setup(opt_args)

        # --- 3. Importance calculation ---
        torch.cuda.synchronize()
        t_enc_start = time.perf_counter()
        with torch.no_grad():
            imp = cal_imp(gaussians, train_cameras, pipe_args, background)

        # --- 4. Optional pruning ---
        N_after_prune = N_original
        if args.prune:
            pmask = prune_mask(dataset_args.percent, imp)
            imp = imp[torch.logical_not(pmask)]
            gaussians.prune_points(pmask)
            N_after_prune = gaussians.get_xyz.shape[0]

        # --- 5. Encode (timed) ---
        bitstreams = encode_mesongs(
            gaussians, dataset_args, imp,
            output_dir=os.path.join(args.output_folder, f"frame_{frame}") if args.save_bitstreams else "",
            save_to_disk=args.save_bitstreams,
        )

        torch.cuda.synchronize()
        t_enc_end = time.perf_counter()
        encode_time_ms = (t_enc_end - t_enc_start) * 1000

        compressed_size_bytes = sum(len(v) for v in bitstreams.values())
        N_after_octree = gaussians.get_xyz.shape[0]

        # --- 6. Decode (timed) ---
        torch.cuda.synchronize()
        t_dec_start = time.perf_counter()

        decoded_gaussians = decode_mesongs(bitstreams, dataset_args)

        torch.cuda.synchronize()
        t_dec_end = time.perf_counter()
        decode_time_ms = (t_dec_end - t_dec_start) * 1000

        N_decoded = decoded_gaussians.get_xyz.shape[0]

        # --- 7. Save PLY (Euler → quaternion → save_ply) ---
        if args.output_ply_folder is not None:
            frame_ply_folder = os.path.join(args.output_ply_folder, str(frame), "point_cloud")
            os.makedirs(frame_ply_folder, exist_ok=True)
            ply_out_path = os.path.join(frame_ply_folder, "point_cloud.ply")
            save_decoded_ply(decoded_gaussians, ply_out_path)

            canonical_frame_ply_folder = config_root / f"frame{frame}" / "decompressed_ply" / str(frame) / "point_cloud"
            canonical_frame_ply_folder.mkdir(parents=True, exist_ok=True)
            canonical_ply_out_path = canonical_frame_ply_folder / "point_cloud.ply"
            if str(canonical_ply_out_path) != ply_out_path:
                save_decoded_ply(decoded_gaussians, str(canonical_ply_out_path))

        benchmark_rows.append({
            "frame": frame,
            "total_encode_ms": encode_time_ms,
            "total_decode_ms": decode_time_ms,
            "original_points": N_original,
            "after_prune_points": N_after_prune,
            "after_octree_points": N_after_octree,
            "decoded_points": N_decoded,
            "uncompressed_size_bytes": uncompressed_size_bytes,
            "compressed_size_bytes": compressed_size_bytes,
        })

        tqdm.write(
            f"  Frame {frame}: N={N_original}"
            f"{'→' + str(N_after_prune) + ' pruned' if args.prune else ''}"
            f"→{N_after_octree} octree→{N_decoded} decoded, "
            f"enc={encode_time_ms:.2f} ms, dec={decode_time_ms:.2f} ms, "
            f"uncomp={uncompressed_size_bytes / 1024 / 1024:.2f} MB, "
            f"comp={compressed_size_bytes / 1024 / 1024:.2f} MB, "
            f"ratio={uncompressed_size_bytes / compressed_size_bytes:.2f}x"
        )

        del gaussians, decoded_gaussians, bitstreams, imp
        torch.cuda.empty_cache()

    # --- Benchmark CSV and summary ---
    if benchmark_rows:
        csv_path = os.path.join(args.output_folder, "benchmark_mesongs.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame_id", "total_encode_ms", "total_decode_ms",
                         "original_points", "after_prune_points",
                         "after_octree_points", "decoded_points",
                         "uncompressed_size_bytes", "compressed_size_bytes"])
            for r in benchmark_rows:
                w.writerow([
                    r["frame"],
                    f"{r['total_encode_ms']:.2f}",
                    f"{r['total_decode_ms']:.2f}",
                    r["original_points"],
                    r["after_prune_points"],
                    r["after_octree_points"],
                    r["decoded_points"],
                    r["uncompressed_size_bytes"],
                    r["compressed_size_bytes"],
                ])

        n = len(benchmark_rows)
        total_enc_ms = sum(r["total_encode_ms"] for r in benchmark_rows)
        total_dec_ms = sum(r["total_decode_ms"] for r in benchmark_rows)
        total_uncomp = sum(r["uncompressed_size_bytes"] for r in benchmark_rows)
        total_comp = sum(r["compressed_size_bytes"] for r in benchmark_rows)
        total_orig_points = sum(r["original_points"] for r in benchmark_rows)
        total_octree_points = sum(r["after_octree_points"] for r in benchmark_rows)
        total_decoded_points = sum(r["decoded_points"] for r in benchmark_rows)

        config_out = {
            "scene_name": scene,
            "depth": depth,
            "n_block": n_block,
            "codebook_size": codebook_size,
            "prune": args.prune,
            "prune_percent": prune_percent,
            "oct_merge": args.oct_merge,
            "vq_steps": args.steps,
            "sh_degree": args.sh_degree,
            "resolution": args.resolution,
            "frame_list": [int(f) for f in frame_list],
        }
        with open(os.path.join(args.output_folder, "mesongs_config.json"), "w") as f:
            json.dump(config_out, f, indent=4)

        for row in benchmark_rows:
            frame_id = int(row["frame"])
            frame_root = config_root / f"frame{frame_id}"
            write_single_frame_benchmark_csv(frame_root / "benchmark_mesongs.csv", row)
            write_single_frame_config_json(frame_root / "mesongs_config.json", config_out, frame_id)

        print("\n" + "=" * 70)
        print("Benchmark Summary (MesonGS compress + decompress)")
        print("=" * 70)
        print(f"  Frames processed:          {n}")
        print(f"  Total encode time:         {total_enc_ms / 1000:.2f} s  (avg {total_enc_ms / n:.2f} ms/frame)")
        print(f"  Total decode time:         {total_dec_ms / 1000:.2f} s  (avg {total_dec_ms / n:.2f} ms/frame)")
        print(f"  Total uncompressed size:   {total_uncomp / 1024 / 1024:.2f} MB  (avg {total_uncomp / n / 1024 / 1024:.2f} MB/frame)")
        print(f"  Total compressed size:     {total_comp / 1024 / 1024:.2f} MB  (avg {total_comp / n / 1024 / 1024:.2f} MB/frame)")
        print(f"  Compression ratio:         {total_uncomp / total_comp:.2f}x")
        print(f"  Avg point flow:            {total_orig_points / n:.0f} → {total_octree_points / n:.0f} octree → {total_decoded_points / n:.0f} decoded")
        print(f"  CSV: {csv_path}")
        print(f"  Canonical frame root:      {config_root}")
        if args.output_ply_folder is not None:
            print(f"  Canonical PLY layout:      {config_root}/frame*/decompressed_ply")
        print("=" * 70)
    else:
        print("No frames were processed.")

    print("Done.")
