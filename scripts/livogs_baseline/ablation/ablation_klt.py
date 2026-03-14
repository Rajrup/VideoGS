#!/usr/bin/env python3
"""
KLT Color Space Ablation: RGB vs YUV vs KLT15.

For each frame, runs LiVoGS compress + decompress with three color space
transforms and records compressed size, per-attribute breakdown, and timing.

Variants:
  klt  -- DC to YUV (BT.709), AC to YUV then KLT15 (current default)
  yuv  -- DC and AC to YUV (BT.601)
  rgb  -- DC and AC stay in RGB (no color transform)

Supports both QUEEN (Neural_3D_Video) and VideoGS (HiFi4G) PLY formats
via the --format flag.
"""

import os
import sys
import csv
import time
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ---------------------------------------------------------------------------
# sys.path setup (same as ablation_rlgr.py / compress_decompress_pipeline.py)
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.dirname(_THIS_DIR)                       # livogs_baseline/
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPTS_DIR))
_LIVOGS_COMPRESSION = os.path.join(_PROJECT_ROOT, "LiVoGS", "compression")
for p in (_PROJECT_ROOT, _LIVOGS_COMPRESSION):
    if p not in sys.path:
        sys.path.insert(0, p)

from compress_decompress import encode_livogs, decode_livogs

sys.path.insert(0, _SCRIPTS_DIR)
from scripts.livogs_baseline import compress_decompress_pipeline as pipeline

searchForMaxIteration = pipeline.searchForMaxIteration
load_queen_ply = getattr(pipeline, "load_queen_ply", None)
find_queen_ply_path = getattr(pipeline, "find_queen_ply_path", None)
save_queen_ply = getattr(pipeline, "save_queen_ply", None)

# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------
VARIANTS = ["klt", "yuv", "rgb"]
VARIANT_LABELS = {
    "klt": "KLT (DC->YUV, AC->YUV+KLT15)",
    "yuv": "YUV (DC+AC->YUV)",
    "rgb": "RGB (no transform)",
}

FORMAT_DEFAULTS = {
    "queen": {
        "J": 17,
    },
    "videogs": {
        "J": 12,
    },
}

# ---------------------------------------------------------------------------
# VideoGS PLY I/O (adapted from VideoGS/scripts/livogs_baseline/)
# Defined inline to avoid cross-repo imports.
# ---------------------------------------------------------------------------

def find_videogs_ply_path(ply_root, frame_id):
    """Find PLY for a VideoGS frame: {root}/{id}/point_cloud/iteration_{max}/point_cloud.ply"""
    ckpt_path = os.path.join(ply_root, str(frame_id), "point_cloud")
    if not os.path.exists(ckpt_path):
        return None
    try:
        max_iter = searchForMaxIteration(ckpt_path)
    except (ValueError, FileNotFoundError):
        return None
    ply_file = os.path.join(ckpt_path, f"iteration_{max_iter}", "point_cloud.ply")
    return ply_file if os.path.exists(ply_file) else None


def load_videogs_ply(ply_path, device="cuda"):
    """Load a VideoGS-trained PLY and return LiVoGS-compatible param dict."""
    from plyfile import PlyData

    plydata = PlyData.read(ply_path)
    vertex = plydata["vertex"]

    means = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1)
    sh_dc = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1)

    rest_names = sorted(
        [p.name for p in vertex.properties if p.name.startswith("f_rest_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    if rest_names:
        sh_rest = np.stack([vertex[name] for name in rest_names], axis=1)
    else:
        sh_rest = np.zeros((len(vertex), 0), dtype=np.float32)
    colors = np.concatenate([sh_dc, sh_rest], axis=1)

    opacities = np.asarray(vertex["opacity"])
    scales = np.stack([vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=1)
    quats = np.stack(
        [vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]], axis=1,
    )

    params = {
        "means": torch.from_numpy(means.copy()).float().to(device),
        "quats": torch.from_numpy(quats.copy()).float().to(device),
        "scales": torch.from_numpy(scales.copy()).float().to(device),
        "opacities": torch.from_numpy(opacities.copy()).float().to(device),
        "colors": torch.from_numpy(colors.copy()).float().to(device),
    }
    uncompressed_size_bytes = sum(v.numel() * v.element_size() for v in params.values())

    params["quats"] = F.normalize(params["quats"], p=2, dim=1)
    if params["opacities"].min() < 0 or params["opacities"].max() > 1:
        params["opacities"] = torch.sigmoid(params["opacities"])
    if params["scales"].min() < 0:
        params["scales"] = torch.exp(params["scales"])

    return params, uncompressed_size_bytes


def save_videogs_ply(params, output_path, sh_degree=3, eps=1e-6):
    """Save reconstructed params to VideoGS-compatible PLY."""
    means = params["means"].detach().cpu().float().numpy()
    quats = params["quats"].detach().cpu().float().numpy()
    scales = params["scales"].detach().cpu().float().numpy()
    opacities = params["opacities"].detach().cpu().float().numpy()
    colors = params["colors"].detach().cpu().float().numpy()

    N = means.shape[0]
    opacities_c = np.clip(opacities, eps, 1.0 - eps)
    opacities_logit = np.log(opacities_c / (1.0 - opacities_c))
    scales_log = np.log(np.clip(scales, eps, None))

    attr_names = ["x", "y", "z", "nx", "ny", "nz"]
    for i in range(3):
        attr_names.append(f"f_dc_{i}")
    n_rest = colors.shape[1] - 3
    for i in range(n_rest):
        attr_names.append(f"f_rest_{i}")
    attr_names.append("opacity")
    for i in range(3):
        attr_names.append(f"scale_{i}")
    for i in range(4):
        attr_names.append(f"rot_{i}")

    normals = np.zeros((N, 3), dtype=np.float32)
    data = np.concatenate(
        [means, normals, colors, opacities_logit.reshape(-1, 1), scales_log, quats],
        axis=1,
    ).astype(np.float32)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {N}\n".encode())
        for name in attr_names:
            f.write(f"property float {name}\n".encode())
        f.write(b"end_header\n")
        f.write(data.tobytes())


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _per_channel_column_names(n_channels):
    """Generate CSV column names for per-dimension compressed bytes.

    Channel layout: quats(0:4), scales(4:7), opacity(7), sh(8:).
    """
    names = []
    for i in range(4):
        names.append(f"quats_dim{i}_compressed_bytes")
    for i in range(3):
        names.append(f"scales_dim{i}_compressed_bytes")
    names.append("opacity_dim0_compressed_bytes")
    num_sh = n_channels - 8
    for i in range(num_sh):
        names.append(f"sh_dim{i}_compressed_bytes")
    assert len(names) == n_channels
    return names


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="KLT color space ablation: RGB vs YUV vs KLT15",
    )
    p.add_argument("--ply_path", type=str, required=True,
                    help="Path to model checkpoint directory")
    p.add_argument("--output_folder", type=str, required=True,
                    help="Folder for benchmark CSV, config, and decompressed PLYs")
    p.add_argument("--format", type=str, default="queen",
                    choices=["queen", "videogs"],
                    help="PLY format: queen (N3DV) or videogs (HiFi4G)")
    p.add_argument("--frame_ids", type=str, default=None,
                    help="Comma-separated frame IDs to process (e.g. '0' or '1,50,100')")
    p.add_argument("--frame_start", type=int, default=None,
                    help="Start frame (inclusive). Ignored if --frame_ids is set.")
    p.add_argument("--frame_end", type=int, default=None,
                    help="End frame (inclusive). Ignored if --frame_ids is set.")
    p.add_argument("--interval", type=int, default=1)
    p.add_argument("--sh_degree", type=int, default=2,
                    help="SH degree (2 for QUEEN/N3DV, 3 for VideoGS/HiFi4G)")
    p.add_argument("--J", type=int, default=None,
                    help="Octree depth (default: 17 for queen/N3DV, 12 for videogs/HiFi4G)")
    p.add_argument("--quantize_step", type=float, default=0.0001,
                    help="Uniform quantization step for all attributes")
    p.add_argument("--nvcomp_algorithm", type=str, default="ANS",
                    choices=["None", "LZ4", "Snappy", "GDeflate", "Deflate",
                             "zStandard", "Cascaded", "Bitcomp", "ANS"])
    p.add_argument("--rlgr_block_size", type=int, default=4096)
    p.add_argument("--save_ply", action="store_true",
                    help="Save decompressed PLYs for quality evaluation")
    p.add_argument("--device", type=str, default="cuda:0")
    args = p.parse_args()

    if args.J is None:
        args.J = FORMAT_DEFAULTS[args.format]["J"]

    if args.format == "queen" and (
        load_queen_ply is None or find_queen_ply_path is None or save_queen_ply is None
    ):
        raise RuntimeError(
            "queen format requested, but Queen PLY helpers are unavailable in "
            "scripts/livogs_baseline/compress_decompress_pipeline.py"
        )

    nvcomp_algorithm = None if args.nvcomp_algorithm == "None" else args.nvcomp_algorithm

    # Build quantize_step dict (uniform QP for all attributes)
    qs = args.quantize_step
    quantize_step = {
        "quats": qs,
        "scales": qs,
        "opacity": qs,
        "sh_dc": qs,
        "sh_rest": [qs] * (3 * ((args.sh_degree + 1) ** 2 - 1)),
    }

    device = args.device
    device_id = int(device.split(":")[1]) if device.startswith("cuda:") else 0

    # Determine frame list
    if args.frame_ids is not None:
        frames = [int(x.strip()) for x in args.frame_ids.split(",")]
    elif args.frame_start is not None and args.frame_end is not None:
        frames = list(range(args.frame_start, args.frame_end + 1, args.interval))
    else:
        # Default: frame 1 for queen, frame 0 for videogs
        frames = [1] if args.format == "queen" else [0]

    # ---------------------------------------------------------------------------
    # PLY I/O dispatch
    # ---------------------------------------------------------------------------
    queen_load = load_queen_ply
    queen_find = find_queen_ply_path
    queen_save = save_queen_ply

    def _load_frame(frame_id):
        if args.format == "queen":
            if queen_load is None or queen_find is None:
                raise RuntimeError("queen PLY helpers are unavailable")
            frame_str = str(frame_id).zfill(4)
            frame_dir = os.path.join(args.ply_path, "frames", frame_str)
            ply_file = queen_find(frame_dir)
            if ply_file is None:
                return None, 0
            params, uncomp = queen_load(ply_file, device=device)
            return params, uncomp

        ply_file = find_videogs_ply_path(args.ply_path, frame_id)
        if ply_file is None:
            return None, 0
        params, uncomp = load_videogs_ply(ply_file, device=device)
        return params, uncomp

    def _save_decoded(decoded_params, variant, frame_id):
        if not args.save_ply:
            return
        if args.format == "queen":
            if queen_save is None:
                raise RuntimeError("queen PLY helpers are unavailable")
            frame_str = str(frame_id).zfill(4)
            ply_dir = os.path.join(
                args.output_folder, variant, "decompressed_ply", "frames", frame_str,
            )
            os.makedirs(ply_dir, exist_ok=True)
            queen_save(
                decoded_params, os.path.join(ply_dir, "point_cloud.ply"), args.sh_degree,
            )
            return

        ply_dir = os.path.join(
            args.output_folder, variant, "decompressed_ply",
            str(frame_id), "point_cloud",
        )
        os.makedirs(ply_dir, exist_ok=True)
        save_videogs_ply(
            decoded_params, os.path.join(ply_dir, "point_cloud.ply"), args.sh_degree,
        )

    os.makedirs(args.output_folder, exist_ok=True)

    NUM_WARMUP = 3
    NUM_TIMING_ITERS = 5

    print("=" * 70)
    print("KLT Color Space Ablation Study")
    print("=" * 70)
    print(f"  PLY path:         {args.ply_path}")
    print(f"  Format:           {args.format}")
    print(f"  Output folder:    {args.output_folder}")
    print(f"  Frames:           {frames}")
    print(f"  J={args.J}, qstep={qs}, SH degree={args.sh_degree}")
    print(f"  nvcomp:           {nvcomp_algorithm or 'None'}")
    print(f"  Save PLYs:        {args.save_ply}")
    print(f"  Variants:         {VARIANTS}")
    print(f"  Warmup rounds:    {NUM_WARMUP}")
    print(f"  Timing iters:     {NUM_TIMING_ITERS} (median)")
    print("=" * 70)

    print(f"Warming up GPU ({NUM_WARMUP} rounds)...")
    params0, _ = _load_frame(frames[0])
    if params0 is None:
        raise ValueError(f"PLY not found for warmup frame {frames[0]}")

    for wi in range(NUM_WARMUP):
        for variant in VARIANTS:
            torch.cuda.synchronize(device_id)
            cs = encode_livogs(
                {k: v.clone() for k, v in params0.items()},
                J=args.J, device=device, device_id=device_id,
                sh_color_space=variant, quantize_step=quantize_step,
                rlgr_block_size=args.rlgr_block_size,
                nvcomp_algorithm=nvcomp_algorithm,
            )
            decode_livogs(cs, device=device, device_id=device_id)
            torch.cuda.synchronize(device_id)
            del cs
        print(f"  Round {wi + 1}/{NUM_WARMUP}: OK")
    del params0
    torch.cuda.empty_cache()
    print("Warmup done.\n")

    all_rows = []

    for frame in tqdm(frames, desc="Frames"):
        params, uncompressed_size_bytes = _load_frame(frame)
        if params is None:
            tqdm.write(f"  WARNING: PLY not found for frame {frame}, skipping")
            continue

        num_original = params["means"].shape[0]

        for variant in VARIANTS:
            encode_times = []
            decode_times = []

            for _ti in range(NUM_TIMING_ITERS):
                params_copy = {k: v.clone() for k, v in params.items()}

                torch.cuda.synchronize(device_id)
                t_enc_start = time.perf_counter()
                compressed_state = encode_livogs(
                    params_copy, J=args.J, device=device, device_id=device_id,
                    sh_color_space=variant, quantize_step=quantize_step,
                    rlgr_block_size=args.rlgr_block_size,
                    nvcomp_algorithm=nvcomp_algorithm,
                )
                torch.cuda.synchronize(device_id)
                encode_times.append((time.perf_counter() - t_enc_start) * 1000)

                t_dec_start = time.perf_counter()
                decoded_params = decode_livogs(
                    compressed_state, device=device, device_id=device_id,
                )
                torch.cuda.synchronize(device_id)
                decode_times.append((time.perf_counter() - t_dec_start) * 1000)

                if _ti < NUM_TIMING_ITERS - 1:
                    del compressed_state, decoded_params, params_copy
                    torch.cuda.empty_cache()

            encode_time_ms = float(np.median(encode_times))
            decode_time_ms = float(np.median(decode_times))

            nvox = compressed_state["Nvox"]
            compressed_size_bytes = compressed_state["total_compressed_bytes"]
            position_compressed_bytes = compressed_state["position_compressed_bytes"]
            attribute_compressed_bytes = compressed_state["attribute_compressed_bytes"]
            per_channel_compressed_bytes = compressed_state["per_channel_compressed_bytes"]

            _save_decoded(decoded_params, variant, frame)

            all_rows.append({
                "frame_id": frame,
                "variant": variant,
                "original_points": num_original,
                "voxelized_points": nvox,
                "uncompressed_size_bytes": uncompressed_size_bytes,
                "compressed_size_bytes": compressed_size_bytes,
                "position_compressed_bytes": position_compressed_bytes,
                "attribute_compressed_bytes": attribute_compressed_bytes,
                "encode_time_ms": encode_time_ms,
                "decode_time_ms": decode_time_ms,
                "per_channel_compressed_bytes": per_channel_compressed_bytes,
            })

            del compressed_state, decoded_params, params_copy
            torch.cuda.empty_cache()

        # Per-frame summary
        frame_rows = [r for r in all_rows if r["frame_id"] == frame]
        sizes_str = " | ".join(
            f"{r['variant']}={r['compressed_size_bytes'] / 1024:.1f}KB"
            for r in frame_rows
        )
        tqdm.write(f"  Frame {frame}: N={num_original} | {sizes_str}")

        del params
        torch.cuda.empty_cache()

    if not all_rows:
        print("No frames were processed.")
        return

    # --- Write CSV ---
    csv_path = os.path.join(args.output_folder, "ablation_klt.csv")
    n_ch = len(all_rows[0]["per_channel_compressed_bytes"])
    per_ch_cols = _per_channel_column_names(n_ch)

    base_columns = [
        "frame_id", "variant",
        "original_points", "voxelized_points",
        "uncompressed_size_bytes", "compressed_size_bytes",
        "position_compressed_bytes", "attribute_compressed_bytes",
        "encode_time_ms", "decode_time_ms",
    ]
    csv_columns = base_columns + per_ch_cols

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(csv_columns)
        for row in all_rows:
            w.writerow([
                row["frame_id"],
                row["variant"],
                row["original_points"],
                row["voxelized_points"],
                row["uncompressed_size_bytes"],
                row["compressed_size_bytes"],
                row["position_compressed_bytes"],
                row["attribute_compressed_bytes"],
                f"{row['encode_time_ms']:.2f}",
                f"{row['decode_time_ms']:.2f}",
                *row["per_channel_compressed_bytes"],
            ])
    print(f"\nCSV saved: {csv_path}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("Summary (mean across all frames)")
    print("=" * 70)

    for variant in VARIANTS:
        rows = [r for r in all_rows if r["variant"] == variant]
        if not rows:
            continue
        avg_comp = np.mean([r["compressed_size_bytes"] for r in rows])
        avg_pos = np.mean([r["position_compressed_bytes"] for r in rows])
        avg_attr = np.mean([r["attribute_compressed_bytes"] for r in rows])
        avg_enc = np.mean([r["encode_time_ms"] for r in rows])
        avg_dec = np.mean([r["decode_time_ms"] for r in rows])
        per_ch = [r["per_channel_compressed_bytes"] for r in rows]
        avg_quats = np.mean([sum(ch[0:4]) for ch in per_ch])
        avg_scales = np.mean([sum(ch[4:7]) for ch in per_ch])
        avg_opacity = np.mean([ch[7] for ch in per_ch])
        avg_sh_dc = np.mean([sum(ch[8:11]) for ch in per_ch])
        avg_sh_rest = np.mean([sum(ch[11:]) for ch in per_ch])

        print(f"\n  {variant:>5s} ({VARIANT_LABELS[variant]}):")
        print(f"    Total:     {avg_comp / 1024 / 1024:.4f} MB")
        print(f"    Position:  {avg_pos / 1024 / 1024:.4f} MB")
        print(f"    Attribute: {avg_attr / 1024 / 1024:.4f} MB")
        print(f"      quats:   {avg_quats / 1024 / 1024:.4f} MB")
        print(f"      scales:  {avg_scales / 1024 / 1024:.4f} MB")
        print(f"      opacity: {avg_opacity / 1024 / 1024:.4f} MB")
        print(f"      sh_dc:   {avg_sh_dc / 1024 / 1024:.4f} MB")
        print(f"      sh_rest: {avg_sh_rest / 1024 / 1024:.4f} MB")
        print(f"    Enc time:  {avg_enc:.2f} ms")
        print(f"    Dec time:  {avg_dec:.2f} ms")

    # --- Relative comparison ---
    klt_rows = [r for r in all_rows if r["variant"] == "klt"]
    if klt_rows:
        klt_avg = np.mean([r["compressed_size_bytes"] for r in klt_rows])
        print("\n" + "-" * 70)
        print("Relative to KLT baseline:")
        for variant in VARIANTS:
            vrows = [r for r in all_rows if r["variant"] == variant]
            if not vrows:
                continue
            v_avg = np.mean([r["compressed_size_bytes"] for r in vrows])
            diff_pct = (v_avg - klt_avg) / klt_avg * 100
            print(f"  {variant:>5s}: {diff_pct:+.2f}%  "
                  f"({v_avg / 1024 / 1024:.4f} vs {klt_avg / 1024 / 1024:.4f} MB)")

    print("=" * 70)

    # --- Save config ---
    config = {
        "ply_path": args.ply_path,
        "format": args.format,
        "frames": frames,
        "J": args.J,
        "sh_degree": args.sh_degree,
        "quantize_step": quantize_step,
        "nvcomp_algorithm": nvcomp_algorithm or "None",
        "rlgr_block_size": args.rlgr_block_size,
        "save_ply": args.save_ply,
        "variants": VARIANTS,
    }
    config_path = os.path.join(args.output_folder, "ablation_klt_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved: {config_path}")


if __name__ == "__main__":
    main()
