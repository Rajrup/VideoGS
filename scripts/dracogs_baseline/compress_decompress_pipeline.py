#!/usr/bin/env python3
"""
DracoGS Compression + Decompression for VideoGS-trained Gaussian Splat Models.

For each frame:
  1. Read PLY from VideoGS checkpoint (own PLY reader)
  2. Encode via DracoGS (in-memory Draco bitstream)
  3. Decode via DracoGS (in-memory)
  4. Save decoded result as VideoGS-compatible PLY (own PLY writer)

Must be run in the videogs conda environment.
"""

import os
import sys
import csv
import json
import time
import argparse
from tqdm import tqdm

# --- sys.path setup: DracoGS build + compression dirs ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_VIDEOGS_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_DRACOGS_ROOT = os.path.join(_VIDEOGS_ROOT, "DracoGS")
_DRACOGS_BUILD = os.path.join(_DRACOGS_ROOT, "build", "compression")
_DRACOGS_COMP = os.path.join(_DRACOGS_ROOT, "compression")

for p in (_DRACOGS_BUILD, _DRACOGS_COMP):
    if p not in sys.path:
        sys.path.insert(0, p)

from compression_decompression import encode_dracogs, decode_dracogs

if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
from utils import read_gs_ply, save_gs_ply


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder) if "iteration_" in fname]
    return max(saved_iters)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DracoGS compress + decompress for VideoGS-trained models"
    )
    parser.add_argument("--ply_path", type=str, required=True,
                        help="Path to checkpoint dir containing frame folders (0, 1, ...)")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Folder for benchmark CSV and metadata")
    parser.add_argument("--output_ply_folder", type=str, required=True,
                        help="Folder for decompressed PLY output")
    parser.add_argument("--frame_start", type=int, default=0)
    parser.add_argument("--frame_end", type=int, default=200)
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--scene_name", type=str, required=True,
                        help="HiFi4G sequence name (e.g. 4K_Actor1_Greeting)")

    # DracoGS quantization parameters (0=lossless, higher=more bits=better quality)
    parser.add_argument("--qp", type=int, default=16, help="Quantization bits for position (0-30)")
    parser.add_argument("--qfd", type=int, default=16, help="Quantization bits for SH DC (0-30)")
    parser.add_argument("--qfr1", type=int, default=16, help="Quantization bits for SH band 1 (0-30)")
    parser.add_argument("--qfr2", type=int, default=16, help="Quantization bits for SH band 2 (0-30)")
    parser.add_argument("--qfr3", type=int, default=16, help="Quantization bits for SH band 3 (0-30)")
    parser.add_argument("--qo", type=int, default=16, help="Quantization bits for opacity (0-30)")
    parser.add_argument("--qs", type=int, default=16, help="Quantization bits for scale (0-30)")
    parser.add_argument("--qr", type=int, default=16, help="Quantization bits for rotation (0-30)")
    parser.add_argument("--cl", type=int, default=7, help="Compression level (0-10)")

    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(args.output_ply_folder, exist_ok=True)

    # --- Print configuration ---
    print("=" * 70)
    print("DracoGS Compress + Decompress Pipeline")
    print("=" * 70)
    print(f"  PLY path:           {args.ply_path}")
    print(f"  Output folder:      {args.output_folder}")
    print(f"  Output PLY folder:  {args.output_ply_folder}")
    print(f"  Frames:             {args.frame_start} to {args.frame_end} (interval={args.interval})")
    print(f"  Scene:              {args.scene_name}")
    print(f"  SH degree:          {args.sh_degree}")
    print(f"  Quantization:       qp={args.qp} qfd={args.qfd} qfr1={args.qfr1} qfr2={args.qfr2} qfr3={args.qfr3} qo={args.qo} qs={args.qs} qr={args.qr}")
    print(f"  Compression level:  {args.cl}")
    print("=" * 70)

    # --- Per-frame loop ---
    benchmark_rows = []

    for frame in tqdm(range(args.frame_start, args.frame_end, args.interval), desc="Frames"):

        # --- 1. Locate and read PLY ---
        ckpt_path = os.path.join(args.ply_path, str(frame), "point_cloud")
        if not os.path.exists(ckpt_path):
            print(f"Warning: Checkpoint not found: {ckpt_path}, skipping frame {frame}")
            continue
        max_iter = searchForMaxIteration(ckpt_path)
        ply_file_path = os.path.join(ckpt_path, f"iteration_{max_iter}", "point_cloud.ply")

        gs_data, uncompressed_size_bytes = read_gs_ply(ply_file_path, sh_degree=args.sh_degree)
        N_original = gs_data["positions"].shape[0]

        # --- 2. Encode (timed) ---
        t_enc_start = time.perf_counter()
        bitstream = encode_dracogs(
            gs_data,
            qp=args.qp, qfd=args.qfd,
            qfr1=args.qfr1, qfr2=args.qfr2, qfr3=args.qfr3,
            qo=args.qo, qs=args.qs, qr=args.qr,
            cl=args.cl,
        )
        t_enc_end = time.perf_counter()
        encode_time_ms = (t_enc_end - t_enc_start) * 1000
        compressed_size_bytes = len(bitstream)

        # --- 3. Decode (timed) ---
        t_dec_start = time.perf_counter()
        gs_decoded = decode_dracogs(bitstream)
        t_dec_end = time.perf_counter()
        decode_time_ms = (t_dec_end - t_dec_start) * 1000
        N_decoded = gs_decoded["positions"].shape[0]

        # --- 4. Save PLY ---
        frame_ply_folder = os.path.join(args.output_ply_folder, str(frame), "point_cloud")
        os.makedirs(frame_ply_folder, exist_ok=True)
        ply_out_path = os.path.join(frame_ply_folder, "point_cloud.ply")
        save_gs_ply(gs_decoded, ply_out_path)

        benchmark_rows.append({
            "frame": frame,
            "encode_time_ms": encode_time_ms,
            "decode_time_ms": decode_time_ms,
            "original_points": N_original,
            "decoded_points": N_decoded,
            "uncompressed_size_bytes": uncompressed_size_bytes,
            "compressed_size_bytes": compressed_size_bytes,
        })

        tqdm.write(
            f"  Frame {frame}: N={N_original}→{N_decoded}, "
            f"enc={encode_time_ms:.2f} ms, dec={decode_time_ms:.2f} ms, "
            f"uncomp={uncompressed_size_bytes / 1024 / 1024:.2f} MB, "
            f"comp={compressed_size_bytes / 1024 / 1024:.2f} MB, "
            f"ratio={uncompressed_size_bytes / compressed_size_bytes:.2f}x"
        )

        del gs_data, gs_decoded, bitstream

    # --- Benchmark CSV and summary ---
    if benchmark_rows:
        csv_path = os.path.join(args.output_folder, "benchmark_dracogs.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame_id", "encode_time_ms", "decode_time_ms",
                         "original_points", "decoded_points",
                         "uncompressed_size_bytes", "compressed_size_bytes"])
            for r in benchmark_rows:
                w.writerow([
                    r["frame"],
                    f"{r['encode_time_ms']:.2f}",
                    f"{r['decode_time_ms']:.2f}",
                    r["original_points"],
                    r["decoded_points"],
                    r["uncompressed_size_bytes"],
                    r["compressed_size_bytes"],
                ])

        n = len(benchmark_rows)
        total_enc_ms = sum(r["encode_time_ms"] for r in benchmark_rows)
        total_dec_ms = sum(r["decode_time_ms"] for r in benchmark_rows)
        total_uncomp = sum(r["uncompressed_size_bytes"] for r in benchmark_rows)
        total_comp = sum(r["compressed_size_bytes"] for r in benchmark_rows)

        config_out = {
            "scene_name": args.scene_name,
            "sh_degree": args.sh_degree,
            "qp": args.qp,
            "qfd": args.qfd,
            "qfr1": args.qfr1,
            "qfr2": args.qfr2,
            "qfr3": args.qfr3,
            "qo": args.qo,
            "qs": args.qs,
            "qr": args.qr,
            "cl": args.cl,
            "frame_start": args.frame_start,
            "frame_end": args.frame_end,
            "interval": args.interval,
        }
        with open(os.path.join(args.output_folder, "dracogs_config.json"), "w") as f:
            json.dump(config_out, f, indent=4)

        print("\n" + "=" * 70)
        print("Benchmark Summary (DracoGS compress + decompress)")
        print("=" * 70)
        print(f"  Frames processed:          {n}")
        print(f"  Total encode time:         {total_enc_ms / 1000:.2f} s  (avg {total_enc_ms / n:.2f} ms/frame)")
        print(f"  Total decode time:         {total_dec_ms / 1000:.2f} s  (avg {total_dec_ms / n:.2f} ms/frame)")
        print(f"  Total uncompressed size:   {total_uncomp / 1024 / 1024:.2f} MB  (avg {total_uncomp / n / 1024 / 1024:.2f} MB/frame)")
        print(f"  Total compressed size:     {total_comp / 1024 / 1024:.2f} MB  (avg {total_comp / n / 1024 / 1024:.2f} MB/frame)")
        print(f"  Compression ratio:         {total_uncomp / total_comp:.2f}x")
        print(f"  CSV: {csv_path}")
        print("=" * 70)
    else:
        print("No frames were processed.")

    print("Done.")
