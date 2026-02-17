import os
import csv
import time
import numpy as np
import cv2
from plyfile import PlyData
import json
import argparse
from tqdm import tqdm

def normalize_uint8(data):
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val == min_val:
        return np.zeros_like(data, dtype=np.uint8), min_val, max_val
    normalized = (data - min_val) / (max_val - min_val) * 255.0
    return normalized.astype(np.uint8), min_val, max_val

def normalize_uint16(data):
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val == min_val:
        return np.zeros_like(data, dtype=np.uint16), min_val, max_val
    normalized = (data - min_val) / (max_val - min_val) * (2 ** 16 - 1)
    return normalized.astype(np.uint16), min_val, max_val

def get_ply_matrix(file_path):
    plydata = PlyData.read(file_path)
    num_vertices = len(plydata['vertex'])
    num_attributes = len(plydata['vertex'].properties)
    data_matrix = np.zeros((num_vertices, num_attributes), dtype=float)
    for i, name in enumerate(plydata['vertex'].data.dtype.names):
        data_matrix[:, i] = plydata['vertex'].data[name]
    return data_matrix

def calculate_image_size(num_points):
    image_size = 8
    while image_size * image_size < num_points:
        image_size += 8
    return image_size

def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder) if "iteration_" in fname]
    return max(saved_iters)

def quantize_videogs_image(current_data, image_size):
    num_attributes = current_data.shape[1]
    images = {}
    min_max_info = {}
    
    for i in range(num_attributes):
        # Position attributes (0, 1, 2) -> uint16 split
        if i < 3:
            attribute_data, min_val, max_val = normalize_uint16(current_data[:, i])
            min_max_info[f'{i}_min'] = float(min_val)
            min_max_info[f'{i}_max'] = float(max_val)
            
            attribute_data_reshaped = attribute_data.reshape(-1, 1)
            image_odd = np.zeros((image_size * image_size, 1), dtype=np.uint8)
            image_even = np.zeros((image_size * image_size, 1), dtype=np.uint8)
            
            # Even = Low Byte, Odd = High Byte
            image_even[:attribute_data_reshaped.shape[0], :] += (attribute_data_reshaped & 0xff)
            image_odd[:attribute_data_reshaped.shape[0], :] += (attribute_data_reshaped >> 8)
            
            images[f"{2*i}"] = image_even.reshape((image_size, image_size))
            images[f"{2*i+1}"] = image_odd.reshape((image_size, image_size))
            
        else:
            attribute_data, min_val, max_val = normalize_uint8(current_data[:, i])
            min_max_info[f'{i}_min'] = float(min_val)
            min_max_info[f'{i}_max'] = float(max_val)
            
            attribute_data_reshaped = attribute_data.reshape(-1, 1)
            image = np.zeros((image_size * image_size, 1), dtype=np.uint8)
            image[:attribute_data_reshaped.shape[0], :] = attribute_data_reshaped
            
            # Offset index by +3 to match VideoGS convention (normals start at 6, etc.)
            # But wait, if we are just compressing generic attributes, we should just map i -> output_index
            # VideoGS convention: 
            # i=0 (x) -> 0, 1
            # i=1 (y) -> 2, 3
            # i=2 (z) -> 4, 5
            # i=3 (nx) -> 6
            # ...
            images[f"{i+3}"] = image.reshape((image_size, image_size))
            
    return images, min_max_info

def encode_videogs_png(images, output_path, frame_idx):
    for key, img in images.items():
        cv2.imwrite(os.path.join(output_path, f"{frame_idx}_{key}.png"), img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_start", type=int, default=0)
    parser.add_argument("--frame_end", type=int, default=200)
    parser.add_argument("--group_size", type=int, default=20)
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--ply_path", type=str, required=True, help="Path to training output containing frame folders (0, 1, ...)")
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--sh_degree", type=int, default=3)
    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    min_max_json = {}
    viewer_min_max_json = {}
    group_info_json = {}
    benchmark_rows = []  # frame_id, time_ms, compressed_size_bytes, num_points

    # Calculate number of groups
    num_frames = args.frame_end - args.frame_start
    num_groups = (num_frames + args.group_size - 1) // args.group_size

    for group in tqdm(range(num_groups), desc="Compressing Groups"):
        frame_start = group * args.group_size + args.frame_start
        frame_end = min((group + 1) * args.group_size - 1 + args.frame_start, args.frame_end - 1)
        
        if frame_start >= args.frame_end:
            break

        group_info_json[str(group)] = {}
        group_info_json[str(group)]['frame_index'] = [group * args.group_size, (group + 1) * args.group_size - 1]
        group_info_json[str(group)]['name_index'] = [frame_start, frame_end]

        output_path = os.path.join(args.output_folder, f"group{group}")
        os.makedirs(output_path, exist_ok=True)

        for frame in tqdm(range(frame_start, frame_end + 1, args.interval), desc=f"Group {group} Frames", leave=False):
            
            # Find checkpoint path
            ckpt_path = os.path.join(args.ply_path, str(frame), "point_cloud")
            if not os.path.exists(ckpt_path):
                print(f"Warning: Checkpoint path not found: {ckpt_path}")
                continue
                
            max_iter = searchForMaxIteration(ckpt_path)
            ply_file_path = os.path.join(ckpt_path, f"iteration_{max_iter}", "point_cloud.ply")
            
            # Read PLY (not timed)
            current_data = get_ply_matrix(ply_file_path)
            num_points = current_data.shape[0]

            # Time only quantize + encode to PNG
            t0 = time.perf_counter()
            image_size = calculate_image_size(num_points=num_points)
            min_max_json[f'{frame}_num'] = num_points
            viewer_min_max_json[frame] = {}
            viewer_min_max_json[frame]['num'] = num_points
            viewer_min_max_json[frame]['info'] = []
            images, frame_min_max = quantize_videogs_image(current_data, image_size)
            encode_videogs_png(images, output_path, frame)
            t1 = time.perf_counter()
            time_ms = (t1 - t0) * 1000

            # Compressed size for this frame (sum of PNGs for this frame)
            frame_size = 0
            for fname in os.listdir(output_path):
                if fname.startswith(f"{frame}_") and fname.endswith(".png"):
                    frame_size += os.path.getsize(os.path.join(output_path, fname))

            benchmark_rows.append({"frame": frame, "time_ms": time_ms, "compressed_size_bytes": frame_size, "num_points": num_points})

            # Update global min_max with frame info
            for k, v in frame_min_max.items():
                min_max_json[f'{frame}_{k}'] = v
                viewer_min_max_json[frame]['info'].append(v)

    # Save Metadata
    with open(os.path.join(args.output_folder, "min_max.json"), "w") as f:
        json.dump(min_max_json, f, indent=4)

    with open(os.path.join(args.output_folder, "viewer_min_max.json"), "w") as f:
        json.dump(viewer_min_max_json, f, indent=4)

    with open(os.path.join(args.output_folder, "group_info.json"), "w") as f:
        json.dump(group_info_json, f, indent=4)

    # Benchmark CSV and summary
    if benchmark_rows:
        csv_path = os.path.join(args.output_folder, "benchmark_compress_to_png.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame_id", "time_ms", "compressed_size_bytes", "num_points"])
            for r in benchmark_rows:
                w.writerow([r["frame"], f"{r['time_ms']:.2f}", r["compressed_size_bytes"], r["num_points"]])
        total_time_ms = sum(r["time_ms"] for r in benchmark_rows)
        total_size = sum(r["compressed_size_bytes"] for r in benchmark_rows)
        n = len(benchmark_rows)
        print("\n" + "=" * 60)
        print("Benchmark Summary (compress to PNG)")
        print("=" * 60)
        print(f"  Frames processed:       {n}")
        print(f"  Total time (excl PLY):  {total_time_ms / 1000:.2f} s")
        print(f"  Avg time per frame:     {total_time_ms / n:.2f} ms")
        print(f"  Total PNG size:         {total_size / 1024 / 1024:.2f} MB")
        print(f"  Avg size per frame:     {total_size / n / 1024 / 1024:.2f} MB")
        print(f"  CSV: {csv_path}")
        print("=" * 60)
        
    print("Compression Complete.")
