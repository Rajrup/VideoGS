"""
Plot LiVoGS compressed size and voxelized point count per frame.
Requires: benchmark_livogs.csv under input_folder/<config_name>/
"""
import os
import argparse
import csv
import matplotlib.pyplot as plt

DEFAULT_INPUT_FOLDER = "/synology/rajrup/VideoGS/train_output/HiFi4G_Dataset/4K_Actor2_Dancing/livogs_compression"
DEFAULT_OUTPUT_FOLDER = os.path.dirname(__file__)


def load_benchmark_csv(path):
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "frame": int(row["frame_id"]),
                "compressed_size_bytes": int(row["compressed_size_bytes"]),
                "original_points": int(row["original_points"]),
                "voxelized_points": int(row["voxelized_points"]),
            })
    return sorted(rows, key=lambda x: x["frame"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default=DEFAULT_INPUT_FOLDER,
                        help="Base folder for livogs_compression (default: %(default)s)")
    parser.add_argument("--j", type=int, required=True, help="Octree depth J (e.g. 15)")
    parser.add_argument("--qstep", type=str, required=True, help="Quantization step (e.g. 0.0001)")
    parser.add_argument("--sh_color_space", type=str, required=True, help="Color space (e.g. klt)")
    parser.add_argument("--output_folder", type=str, default=DEFAULT_OUTPUT_FOLDER,
                        help="Override: output folder for plot PNG")
    args = parser.parse_args()

    config_name = f"J_{args.j}_qstep_{args.qstep}_{args.sh_color_space}"
    config_dir = os.path.join(args.input_folder, config_name)
    csv_path = os.path.join(config_dir, "benchmark_livogs.csv")
    out_dir = os.path.join(args.output_folder, "plots", "livogs_compression", config_name)
    out_path = os.path.join(out_dir, "compressed_size.png")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isfile(csv_path):
        raise SystemExit(f"Required file not found: {csv_path}")

    rows = load_benchmark_csv(csv_path)
    frame_ids = [r["frame"] for r in rows]
    sizes_mb = [r["compressed_size_bytes"] / 1024 / 1024 for r in rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(frame_ids))
    tick_every = max(1, len(frame_ids) // 40)

    ax.plot(x, sizes_mb, "o-", label="LiVoGS compressed", color="coral", markersize=3)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Size (MB)")
    ax.set_title(f"LiVoGS compressed size per frame [{config_name}]")
    ax.set_ylim(bottom=0)
    ax.set_xticks(list(x)[::tick_every])
    ax.set_xticklabels(frame_ids[::tick_every], rotation=90)
    ax.legend()
    ax.grid(True, alpha=0.3)

    avg_mb = sum(sizes_mb) / len(sizes_mb)
    ax.annotate(
        f"avg = {avg_mb:.2f} MB/frame",
        xy=(0.02, 0.95), xycoords="axes fraction",
        fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
