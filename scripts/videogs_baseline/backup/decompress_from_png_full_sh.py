import os
import csv
import time
import numpy as np
import json
import argparse
import sys
from tqdm import tqdm

# --- Setup sys.path for VideoGS imports ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_VIDEOGS_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_VIDEOGS_COMPRESSION = os.path.join(_VIDEOGS_ROOT, "compression")

if _VIDEOGS_COMPRESSION not in sys.path:
    sys.path.insert(0, _VIDEOGS_COMPRESSION)

from compress_decompress import decode_videogs_png, dequantize_videogs_image

def save_ply(data, output_file, sh_degree):
    n, k = data.shape
    
    attribute_names = []
    attribute_names.append('x')
    attribute_names.append('y')
    attribute_names.append('z')
    attribute_names.append('nx')
    attribute_names.append('ny')
    attribute_names.append('nz')
    for i in range(3):
        attribute_names.append('f_dc_' + str(i))
    
    # Calculate number of rest coefficients
    # Total attributes = 3 (pos) + 3 (norm) + 3 (dc) + n_rest + 1 (op) + 3 (scale) + 4 (rot)
    # k = 17 + n_rest
    n_rest = k - 17
    
    for i in range(n_rest):
        attribute_names.append('f_rest_' + str(i))
        
    attribute_names.append('opacity')
    for i in range(3):
        attribute_names.append('scale_' + str(i))
    for i in range(4):
        attribute_names.append('rot_' + str(i))

    assert k == len(attribute_names), f"Shape mismatch: data has {k} cols, expected {len(attribute_names)}"

    with open(output_file, 'wb') as ply_file:
        ply_file.write(b"ply\n")
        ply_file.write(b"format binary_little_endian 1.0\n")
        ply_file.write(b"element vertex %d\n" % n)
        
        for attribute_name in attribute_names:
            ply_file.write(b"property float %s\n" % attribute_name.encode())
        
        ply_file.write(b"end_header\n")
        
        for i in range(n):
            vertex_data = data[i].astype(np.float32).tobytes()
            ply_file.write(vertex_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compressed_folder", type=str, required=True, help="Folder containing min_max.json and group folders")
    parser.add_argument("--output_ply_folder", type=str, required=True)
    parser.add_argument("--sh_degree", type=int, default=3)
    args = parser.parse_args()

    if not os.path.exists(args.output_ply_folder):
        os.makedirs(args.output_ply_folder)

    # Load Metadata
    with open(os.path.join(args.compressed_folder, "min_max.json"), "r") as f:
        min_max_info = json.load(f)
    
    with open(os.path.join(args.compressed_folder, "group_info.json"), "r") as f:
        group_info = json.load(f)

    benchmark_rows = []
    # Iterate Groups
    for group_id, info in tqdm(group_info.items(), desc="Decompressing Groups"):
        frame_start, frame_end = info['name_index']
        group_folder = os.path.join(args.compressed_folder, f"group{group_id}")
        
        for frame in tqdm(range(frame_start, frame_end + 1), desc=f"Group {group_id}", leave=False):
            
            # 1. Decode PNGs
            # Estimate num_attributes: 3(pos) + 3(norm) + 3(dc) + 45(rest) + 1(op) + 3(scale) + 4(rot) = 62
            # Max index = 62 + 3 - 1 = 64
            # Time decode + dequantize only (exclude save_ply)
            t0 = time.perf_counter()
            images = decode_videogs_png(group_folder, frame, num_attributes=65)
            if not images:
                print(f"No images found for frame {frame}")
                continue
            ply_data = dequantize_videogs_image(images, frame, min_max_info)
            t1 = time.perf_counter()
            time_ms = (t1 - t0) * 1000
            benchmark_rows.append({"frame": frame, "time_ms": time_ms})

            # Save PLY (not timed)
            frame_ply_folder = os.path.join(args.output_ply_folder, f"{frame}", "point_cloud")
            os.makedirs(frame_ply_folder, exist_ok=True)
            ply_out_path = os.path.join(frame_ply_folder, "point_cloud.ply")
            save_ply(ply_data, ply_out_path, args.sh_degree)

    # Benchmark CSV and summary
    if benchmark_rows:
        csv_path = os.path.join(args.output_ply_folder, "benchmark_decompress_from_png.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame_id", "time_ms"])
            for r in benchmark_rows:
                w.writerow([r["frame"], f"{r['time_ms']:.2f}"])
        total_ms = sum(r["time_ms"] for r in benchmark_rows)
        n = len(benchmark_rows)
        print("\n" + "=" * 60)
        print("Benchmark Summary (decompress PNG to PLY, excl. save)")
        print("=" * 60)
        print(f"  Frames:           {n}")
        print(f"  Total time:       {total_ms / 1000:.2f} s")
        print(f"  Avg time/frame:   {total_ms / n:.2f} ms")
        print(f"  CSV: {csv_path}")
        print("=" * 60)

