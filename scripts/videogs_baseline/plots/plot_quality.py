"""
Plot GT vs decompressed model quality per frame (PSNR and SSIM).
Requires: evaluation_results.csv under input_folder/qp_<qp>/evaluation_renders/
"""
import os
import argparse
import csv
import matplotlib.pyplot as plt

DEFAULT_INPUT_FOLDER = "/synology/rajrup/VideoGS/train_output/HiFi4G_Dataset/4K_Actor2_Dancing/videogs_compression"
DEFAULT_OUTPUT_FOLDER = os.path.dirname(__file__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default=DEFAULT_INPUT_FOLDER,
                        help="Base folder for videogs_compression (default: %(default)s)")
    parser.add_argument("--qp", type=int, required=True, help="QP value (e.g. 22)")
    parser.add_argument("--output_folder", type=str, default=DEFAULT_OUTPUT_FOLDER, help="Override: output PNG path")
    args = parser.parse_args()

    qp_dir = os.path.join(args.input_folder, f"qp_{args.qp}")
    evaluation_csv = os.path.join(qp_dir, "evaluation_renders", "evaluation_results.csv")
    out_dir = os.path.join(args.output_folder, "plots", "videogs_compression", f"qp_{args.qp}")
    out_path = os.path.join(out_dir, "quality.png")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isfile(evaluation_csv):
        raise SystemExit(f"Required file not found: {evaluation_csv}")

    frames, gt_psnr, decomp_psnr, gt_ssim, decomp_ssim = [], [], [], [], []
    with open(evaluation_csv) as f:
        r = csv.DictReader(f)
        for row in r:
            if row["frame_id"] == "avg":
                continue
            frames.append(int(row["frame_id"]))
            gt_psnr.append(float(row["gt_psnr"]))
            decomp_psnr.append(float(row["decomp_psnr"]))
            gt_ssim.append(float(row["gt_ssim"]))
            decomp_ssim.append(float(row["decomp_ssim"]))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    x = range(len(frames))
    tick_every = 5
    ax1.set_xticks(x[::tick_every])
    ax1.set_xticklabels(frames[::tick_every], rotation=90)
    ax1.plot(x, gt_psnr, "o-", label="GT model", color="green", markersize=4)
    ax1.plot(x, decomp_psnr, "s-", label="Decompressed model", color="coral", markersize=4)
    ax1.set_ylabel("PSNR")
    ax1.set_title("Quality per frame: GT vs Decompressed")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(x, gt_ssim, "o-", label="GT model", color="green", markersize=4)
    ax2.plot(x, decomp_ssim, "s-", label="Decompressed model", color="coral", markersize=4)
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("SSIM")
    plt.setp(ax2.get_xticklabels(), rotation=90)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
