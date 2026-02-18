"""
Plot compressed size per frame: PNG size and video size (avg per frame from group).
Requires: benchmark_compress_to_png.csv, benchmark_compress_png_2_video.csv under input_folder/qp_<qp>/
"""
import os
import argparse
import csv
import matplotlib.pyplot as plt

DEFAULT_INPUT_FOLDER = "/synology/rajrup/VideoGS/train_output/HiFi4G_Dataset/4K_Actor2_Dancing/videogs_compression"
DEFAULT_OUTPUT_FOLDER = os.path.dirname(__file__)

def load_png_csv(path):
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "frame": int(row["frame_id"]),
                "size_bytes": int(row["compressed_size_bytes"]),
            })
    return sorted(rows, key=lambda x: x["frame"])

def load_video_csv(path):
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "group_id": int(row["group_id"]),
                "group_compressed_size_bytes": int(row["group_compressed_size_bytes"]),
                "num_frames": int(row["num_frames"]),
            })
    return sorted(rows, key=lambda x: x["group_id"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default=DEFAULT_INPUT_FOLDER,
                        help="Base folder for videogs_compression (default: %(default)s)")
    parser.add_argument("--qp", type=int, required=True, help="QP value (e.g. 22)")
    parser.add_argument("--output_folder", type=str, default=DEFAULT_OUTPUT_FOLDER, help="Override: output PNG path")
    args = parser.parse_args()

    qp_dir = os.path.join(args.input_folder, f"qp_{args.qp}")
    png_csv = os.path.join(qp_dir, "compressed_png", "benchmark_compress_to_png.csv")
    video_csv = os.path.join(qp_dir, "compressed_video", "benchmark_compress_png_2_video.csv")
    out_dir = os.path.join(args.output_folder, "plots", "videogs_compression", f"qp_{args.qp}")
    out_path = os.path.join(out_dir, "compressed_size.png")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isfile(png_csv):
        raise SystemExit(f"Required file not found: {png_csv}")
    if not os.path.isfile(video_csv):
        raise SystemExit(f"Required file not found: {video_csv}")

    png_rows = load_png_csv(png_csv)
    video_rows = load_video_csv(video_csv)
    if not video_rows:
        raise SystemExit("No video benchmark rows")

    frame_to_group = []
    for row in video_rows:
        for _ in range(row["num_frames"]):
            frame_to_group.append(row["group_id"])

    frame_ids = [r["frame"] for r in png_rows]
    png_sizes_mb = [r["size_bytes"] / 1024 / 1024 for r in png_rows]
    video_sizes_mb = []
    for r in png_rows:
        fid = r["frame"]
        if fid < len(frame_to_group):
            gid = frame_to_group[fid]
            vr = next(x for x in video_rows if x["group_id"] == gid)
            video_sizes_mb.append(vr["group_compressed_size_bytes"] / vr["num_frames"] / 1024 / 1024)
        else:
            video_sizes_mb.append(0)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(frame_ids))
    ax.plot(x, png_sizes_mb, "o-", label="PNG size", color="steelblue", markersize=4)
    ax.plot(x, video_sizes_mb, "s-", label="MP4 size (avg/frame)", color="coral", markersize=4)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Size (MB)")
    ax.set_title(f"VideoGS compressed size per frame [QP={args.qp}]")
    tick_every = 5
    ax.set_xticks(x[::tick_every])
    ax.set_xticklabels(frame_ids[::tick_every], rotation=90)
    ax.legend()
    ax.grid(True, alpha=0.3)

    avg_png = sum(png_sizes_mb) / len(png_sizes_mb)
    avg_mp4 = sum(video_sizes_mb) / len(video_sizes_mb)
    ax.annotate(
        f"avg PNG={avg_png:.2f} MB, avg MP4={avg_mp4:.2f} MB/frame",
        xy=(0.02, 0.95), xycoords="axes fraction",
        fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
