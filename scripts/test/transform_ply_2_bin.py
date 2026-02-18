#!/usr/bin/env python3
"""
Convert decompressed VideoGS PLY models to per-attribute .bin files.

For each frame, reads point_cloud.ply and writes:
  means3d.bin     — [x₀,y₀,z₀, x₁,y₁,z₁, ...] float32
  scales3d.bin    — [sx₀,sy₀,sz₀, sx₁,sy₁,sz₁, ...] float32
  quats3d.bin     — [w₀,x₀,y₀,z₀, w₁,x₁,y₁,z₁, ...] float32
  colors3d.bin    — [r₀,g₀,b₀, r₁,g₁,b₁, ...] float32  (SH0 only)
  opacities3d.bin — [o₀,o₁,o₂, ...] float32

N can be recovered as len(means3d.bin) / (3 * 4).
"""

import os
import argparse
import numpy as np
from plyfile import PlyData
from tqdm import tqdm


def convert_frame(ply_path, out_dir):
    plydata = PlyData.read(ply_path)
    v = plydata['vertex']

    means = np.stack([v['x'], v['y'], v['z']], axis=1).astype(np.float32)
    scales = np.stack([v['scale_0'], v['scale_1'], v['scale_2']], axis=1).astype(np.float32)
    quats = np.stack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']], axis=1).astype(np.float32)
    colors = np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], axis=1).astype(np.float32)
    opacities = np.asarray(v['opacity']).astype(np.float32)

    os.makedirs(out_dir, exist_ok=True)
    means.tofile(os.path.join(out_dir, "means3d.bin"))
    scales.tofile(os.path.join(out_dir, "scales3d.bin"))
    quats.tofile(os.path.join(out_dir, "quats3d.bin"))
    colors.tofile(os.path.join(out_dir, "colors3d.bin"))
    opacities.tofile(os.path.join(out_dir, "opacities3d.bin"))


def main():
    parser = argparse.ArgumentParser(
        description="Convert decompressed PLY to per-attribute .bin files"
    )
    parser.add_argument("--input_folder", type=str, required=True,
                        help="Base videogs_compression folder (e.g. .../videogs_compression)")
    parser.add_argument("--qp", type=int, required=True, help="QP value (e.g. 22)")
    parser.add_argument("--frame_start", type=int, default=0, help="First frame (inclusive, default: 0)")
    parser.add_argument("--frame_end", type=int, default=200, help="Last frame (exclusive, default: 200)")
    args = parser.parse_args()

    ply_root = os.path.join(args.input_folder, f"qp_{args.qp}", "decompressed_ply")
    bin_root = os.path.join(args.input_folder, f"qp_{args.qp}", "decompressed_bin")

    if not os.path.isdir(ply_root):
        raise SystemExit(f"PLY folder not found: {ply_root}")

    frame_dirs = [
        str(f) for f in range(args.frame_start, args.frame_end)
        if os.path.isdir(os.path.join(ply_root, str(f)))
    ]

    for frame_id in tqdm(frame_dirs, desc="Converting PLY → bin"):
        ply_path = os.path.join(ply_root, frame_id, "point_cloud", "point_cloud.ply")
        if not os.path.isfile(ply_path):
            print(f"Warning: PLY not found: {ply_path}, skipping")
            continue
        out_dir = os.path.join(bin_root, frame_id)
        convert_frame(ply_path, out_dir)

    print(f"Done. Output: {bin_root}")


if __name__ == "__main__":
    main()
