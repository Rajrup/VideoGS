#!/usr/bin/env python3
"""GPU worker for LiVoGS compression experiments.

Performs compress + decompress + (optional) quality evaluation for a single
QP configuration.  Designed to be spawned by ``run_rd_pipeline.py`` with
``CUDA_VISIBLE_DEVICES`` set per job.

Replaces the old ``evaluate_livogs_compression.py`` which was a thin subprocess
wrapper adding an unnecessary process layer.  This worker calls ``codec`` and
``evaluate_decompress`` directly — no nested subprocesses.

Standalone usage::

    CUDA_VISIBLE_DEVICES=0 python scripts/livogs_baseline/rd_pipeline/worker.py \
        --dataset_name HiFi4G_Dataset --sequence_name 4K_Actor1_Greeting \\
        --frame_id 0 --j 15 --qp_config_json path/to/qp.json
"""

import argparse
import json
import os
import shutil
import sys

# -- config + sibling imports (before any CUDA imports) -----------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

config.setup_livogs_imports()

# Add scripts/ so we can import evaluate_decompress
if config.SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, config.SCRIPTS_DIR)

import codec
from evaluate_decompress import evaluate_decompression_quality


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LiVoGS compression worker (compress + decompress + eval)",
    )
    parser.add_argument("--data_path",      default=config.DATA_PATH)
    parser.add_argument("--dataset_name",   required=True)
    parser.add_argument("--sequence_name",  required=True)
    parser.add_argument("--resolution",     type=int, default=config.RESOLUTION)
    parser.add_argument("--frame_id",       type=int, default=None,
                        help="Single frame to process (overrides --frame_start/--frame_end)")
    parser.add_argument("--frame_start",    type=int, default=0)
    parser.add_argument("--frame_end",      type=int, default=1)
    parser.add_argument("--interval",       type=int, default=1)
    parser.add_argument("--sh_degree",      type=int, default=config.SH_DEGREE)
    parser.add_argument("--j",              type=int, default=config.J,
                        help="Octree depth for voxelization")
    parser.add_argument("--quantize_step",  type=float, default=0.0001,
                        help="Uniform quantization step (ignored when --qp_config_json set)")
    parser.add_argument("--sh_color_space", default=config.SH_COLOR_SPACE,
                        choices=["rgb", "yuv", "klt"])
    parser.add_argument("--rlgr_block_size", type=int, default=config.RLGR_BLOCK_SIZE)
    parser.add_argument("--qp_config_json", default=None,
                        help="Path to QP config JSON (overrides --quantize_step)")
    parser.add_argument("--device",         default="cuda:0",
                        help="Torch device for codec (use cuda:0 with CUDA_VISIBLE_DEVICES pinning)")
    parser.add_argument("--disable_image_and_ply_saving", action="store_true",
                        help="Fast mode: skip PLY save and quality evaluation")
    args = parser.parse_args()

    # --- Resolve frame range -------------------------------------------------
    if args.frame_id is not None:
        args.frame_start = args.frame_id
        args.frame_end   = args.frame_id + 1

    # --- Resolve QP config and output paths ----------------------------------
    qp_label = None
    qp_data  = None
    frame_id = args.frame_id  # may be None in standard mode

    if args.qp_config_json is not None:
        with open(args.qp_config_json) as f:
            qp_data = json.load(f)
        qp_label = qp_data["label"]
        if frame_id is None:
            frame_id         = qp_data.get("frame_id")
            args.frame_start = frame_id
            args.frame_end   = frame_id + 1

    # Build output folder using shared path conventions
    if qp_label is not None:
        output_folder = config.experiment_dir(
            args.data_path, args.dataset_name, args.sequence_name,
            frame_id, args.j, qp_label,
        )
    else:
        output_folder = config.standard_output_dir(
            args.data_path, args.dataset_name, args.sequence_name,
            args.j, args.quantize_step, args.sh_color_space,
        )

    gt_model_path = config.checkpoint_dir(
        args.data_path, args.dataset_name, args.sequence_name,
    )
    dataset_path = config.processed_dataset_dir(
        args.data_path, args.dataset_name, args.sequence_name,
    )

    sep = "=" * 70
    print(sep)
    print("LiVoGS Worker")
    print(f"  Dataset:       {args.dataset_name}")
    print(f"  Sequence:      {args.sequence_name}")
    print(f"  GT model path: {gt_model_path}")
    print(f"  Output folder: {output_folder}")
    print(f"  Frames:        {args.frame_start}..{args.frame_end} (interval={args.interval})")
    print(f"  Device:        {args.device}")
    if qp_label and qp_data is not None:
        print(f"  QP label:      {qp_label}  (beta={qp_data['beta']}, baseline_qp={qp_data['baseline_qp']})")
    else:
        print(f"  J:             {args.j}  |  quantize_step: {args.quantize_step}  |  color space: {args.sh_color_space}")
    print(sep)

    # Copy QP config JSON into output for provenance
    os.makedirs(output_folder, exist_ok=True)
    if args.qp_config_json is not None:
        shutil.copy(args.qp_config_json, os.path.join(output_folder, "qp_config.json"))

    # --- Build quantize_step dict --------------------------------------------
    if args.qp_config_json is not None and qp_data is not None:
        quantize_step = qp_data["quantize_config"]
    else:
        qs = args.quantize_step
        quantize_step = {
            "quats": qs, "scales": qs, "opacity": qs,
            "sh_dc": qs,
            "sh_rest": [qs] * (3 * ((args.sh_degree + 1) ** 2 - 1)),
        }

    # --- Step 1: Compress + Decompress (direct call) -------------------------
    print(f"\n{sep}\nStep 1: LiVoGS Compress + Decompress\n{sep}")
    codec.compress_decompress(
        ply_path=gt_model_path,
        output_folder=output_folder,
        output_ply_folder=os.path.join(output_folder, "decompressed_ply"),
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        interval=args.interval,
        sh_degree=args.sh_degree,
        J=args.j,
        quantize_step=quantize_step,
        sh_color_space=args.sh_color_space,
        rlgr_block_size=args.rlgr_block_size,
        device=args.device,
        skip_save_ply=args.disable_image_and_ply_saving,
    )

    # --- Step 2: Evaluate Decompression Quality (direct call) ----------------
    if args.disable_image_and_ply_saving:
        print("[INFO] --disable_image_and_ply_saving: skipping quality evaluation.")
    else:
        print(f"\n{sep}\nStep 2: Evaluate Decompression Quality\n{sep}")
        eval_args = argparse.Namespace(
            gt_ply_path=gt_model_path,
            decompressed_ply_path=os.path.join(output_folder, "decompressed_ply"),
            dataset_path=dataset_path,
            output_render_path=os.path.join(output_folder, "evaluation"),
            save_renders=True,
            sh_degree=args.sh_degree,
            resolution=args.resolution,
            white_background=True,
            no_white_background=False,
            llffhold=8,
            frame_start=args.frame_start,
            frame_end=args.frame_end,
            interval=args.interval,
        )
        evaluate_decompression_quality(eval_args)

    print(f"\n{sep}")
    print(f"Done!  Results in: {output_folder}")
    print(sep)


if __name__ == "__main__":
    main()
