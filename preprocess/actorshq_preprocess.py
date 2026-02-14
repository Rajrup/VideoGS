"""
ActorsHQ (HumanRF) preprocessing for VideoGS.

Converts per-frame COLMAP output and multiview images into the VideoGS format:
  output/<frame_id>/images/<view_idx>.png
  output/<frame_id>/transforms.json

Optionally composites images with masks to produce white background.
"""

import os
import re
import sys
import json
import argparse
import tempfile
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm

# Preprocess dir; colmap2k.py lives here and is invoked as subprocess
PREPROCESS_DIR = os.path.dirname(os.path.realpath(__file__))
DEFAULT_AABB_SCALE = 2


def run_colmap2k_subprocess(sparse_path, out_json_path, keep_colmap_coords=True, aabb_scale=DEFAULT_AABB_SCALE):
    """Run colmap2k via subprocess; uses a temp copy of sparse so source is not modified."""
    tmp = tempfile.mkdtemp(prefix="actorshq_colmap2k_")
    try:
        for f in ("cameras.bin", "images.bin"):
            src = os.path.join(sparse_path, f)
            if not os.path.isfile(src):
                raise FileNotFoundError(f"Missing {f} in {sparse_path}")
            shutil.copy2(src, os.path.join(tmp, f))
        # colmap2k.py main calls bin2txt(--text) itself, so temp only needs .bin files
        cmd = (
            f'cd "{PREPROCESS_DIR}" && {sys.executable} colmap2k.py '
            f'--text "{tmp}" --out "{out_json_path}" --keep_colmap_coords --aabb_scale {aabb_scale}'
        )
        ret = os.system(cmd)
        if ret != 0:
            raise RuntimeError(f"colmap2k.py exited with code {ret}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def image_name_to_camera_id(image_basename):
    """Extract camera ID from ActorsHQ image name for deterministic ordering.
    e.g. Cam001_rgb000000.png -> 1, Cam160_rgb000000.jpg -> 160.
    Returns int or None if pattern does not match.
    """
    m = re.match(r"Cam(\d+)_rgb\d+\.(?:png|jpg|jpeg)", image_basename, re.IGNORECASE)
    return int(m.group(1)) if m else None


def image_name_to_mask_name(image_basename):
    """e.g. Cam001_rgb000000.png -> Cam001_mask000000.png"""
    m = re.match(r"(Cam\d+)_rgb(\d+)\.(?:png|jpg|jpeg)", image_basename, re.IGNORECASE)
    if m:
        return f"{m.group(1)}_mask{m.group(2)}.png"
    return None


def resolve_image_path(images_dir, image_basename):
    """Return path to image file, preferring .png over .jpg (lossless).
    Tries <stem>.png then <stem>.jpg. Returns None if neither exists.
    """
    stem, _ = os.path.splitext(image_basename)
    for ext in (".png", ".jpg", ".jpeg"):
        path = os.path.join(images_dir, stem + ext)
        if os.path.isfile(path):
            return path
    return None


def composite_white_background(rgb_path, mask_path, out_path):
    """Composite RGB image with mask: foreground from RGB, background white."""
    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L"))
    mask = (mask > 127).astype(np.float32)[:, :, np.newaxis]
    white = np.ones_like(rgb, dtype=np.float32) * 255
    out = (rgb.astype(np.float32) * mask + white * (1 - mask)).clip(0, 255).astype(np.uint8)
    Image.fromarray(out).save(out_path)


def get_frame_dirs(frames_dir):
    """Return sorted list of frame directory names (e.g. frame0, frame1, ...)."""
    if not os.path.isdir(frames_dir):
        return []
    names = [d for d in os.listdir(frames_dir) if os.path.isdir(os.path.join(frames_dir, d))]

    def natural_key(name):
        m = re.search(r"\d+", name)
        return (0, int(m.group())) if m else (1, name)

    return sorted(names, key=natural_key)


def main():
    parser = argparse.ArgumentParser(
        description="Process ActorsHQ (HumanRF) data with per-frame COLMAP into VideoGS format."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the frames directory (e.g. .../Actor08/Sequence1/4x/frames)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output dataset directory (e.g. .../ActorsHQ_Dataset_processed/Actor08/Sequence1/4x)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="First frame index to process (0-based). Default: process all.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Last frame index (inclusive). Default: process all.",
    )
    parser.add_argument(
        "--white_background",
        action="store_true",
        default=True,
        help="Composite images with masks for white background (default: True).",
    )
    parser.add_argument(
        "--no_white_background",
        action="store_false",
        dest="white_background",
        help="Do not apply masks; keep original (black) background.",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move source images to output; default is to copy.",
    )
    parser.add_argument(
        "--aabb_scale",
        type=int,
        default=DEFAULT_AABB_SCALE,
        choices=[1, 2, 4, 8, 16, 32, 64, 128],
        help="NeRF/NeuS2 AABB scale written to transforms.json (default: 2). Use 2+ for ActorsHQ to avoid head clipping.",
    )
    args = parser.parse_args()

    if args.input == args.output:
        raise ValueError("Input and output directories must differ")

    if not os.path.isdir(args.input):
        raise FileNotFoundError(f"Input directory does not exist: {args.input}")
    os.makedirs(args.output, exist_ok=True)

    frame_dirs = get_frame_dirs(args.input)
    if not frame_dirs:
        raise ValueError(f"No frame directories found under {args.input}")

    # Optional frame range by index in sorted list
    indices = list(range(len(frame_dirs)))
    if args.start is not None:
        indices = [i for i in indices if i >= args.start]
    if args.end is not None:
        indices = [i for i in indices if i <= args.end]
    if not indices:
        raise ValueError("No frames in the given --start/--end range")

    for idx in tqdm(indices, desc="Frames"):
        frame_name = frame_dirs[idx]
        frame_path = os.path.join(args.input, frame_name)
        sparse_path = os.path.join(frame_path, "sparse")
        images_src_dir = os.path.join(frame_path, "images")
        masks_dir = os.path.join(frame_path, "masks")

        if not os.path.isdir(sparse_path):
            tqdm.write(f"Skipping {frame_name}: no sparse/")
            continue
        if not os.path.isdir(images_src_dir):
            tqdm.write(f"Skipping {frame_name}: no images/")
            continue

        # Output frame folder: use integer index so training can use source_path/0, source_path/1, ...
        out_frame_name = str(idx)
        out_frame_path = os.path.join(args.output, out_frame_name)
        out_images_path = os.path.join(out_frame_path, "images")
        os.makedirs(out_images_path, exist_ok=True)

        # Generate transforms.json from COLMAP for this frame
        tmp_json = os.path.join(tempfile.gettempdir(), f"actorshq_transforms_{idx}.json")
        try:
            run_colmap2k_subprocess(sparse_path, tmp_json, keep_colmap_coords=True, aabb_scale=args.aabb_scale)
        except Exception as e:
            tqdm.write(f"Skipping {frame_name}: colmap2k failed: {e}")
            continue

        with open(tmp_json, "r") as f:
            transforms = json.load(f)
        try:
            os.remove(tmp_json)
        except OSError:
            pass

        frames_list = transforms.get("frames", [])
        if not frames_list:
            tqdm.write(f"Skipping {frame_name}: no frames in transforms")
            continue

        # Sort by camera ID (Cam001 -> 1, Cam002 -> 2, ...) so that output 0.png = Cam001,
        # 1.png = Cam002, etc. COLMAP order is by image_id and does not match camera numbering.
        def frame_sort_key(f):
            path = f.get("file_path", "")
            basename = os.path.basename(path.strip())
            cid = image_name_to_camera_id(basename)
            return (0, cid) if cid is not None else (1, basename)

        frames_list = sorted(frames_list, key=frame_sort_key)

        # Remap file_path to images/0.png, images/1.png, ... and copy/process images
        for view_idx, frame in enumerate(frames_list):
            old_path = frame.get("file_path", "")
            # COLMAP name can be "images/Cam001_rgb000000.png" or "Cam001_rgb000000.png"
            image_basename = os.path.basename(old_path.strip())
            if not image_basename:
                image_basename = f"{view_idx}.png"
            # Prefer .png over .jpg (lossless)
            src_image = resolve_image_path(images_src_dir, image_basename) or os.path.join(images_src_dir, image_basename)
            dst_image = os.path.join(out_images_path, f"{view_idx}.png")

            if not os.path.isfile(src_image):
                tqdm.write(f"Missing image: {src_image}")
                # Still write transforms with correct file_path
            else:
                if args.white_background and os.path.isdir(masks_dir):
                    mask_name = image_name_to_mask_name(image_basename)
                    mask_path = os.path.join(masks_dir, mask_name) if mask_name else None
                    if mask_path and os.path.isfile(mask_path):
                        composite_white_background(src_image, mask_path, dst_image)
                    else:
                        if args.move:
                            shutil.move(src_image, dst_image)
                        else:
                            shutil.copy2(src_image, dst_image)
                else:
                    if args.move:
                        shutil.move(src_image, dst_image)
                    else:
                        shutil.copy2(src_image, dst_image)

            frame["file_path"] = f"images/{view_idx}.png"

        transforms["frames"] = frames_list
        out_json_path = os.path.join(out_frame_path, "transforms.json")
        with open(out_json_path, "w") as f:
            json.dump(transforms, f, indent=4)

    print(f"Done. Processed {len(indices)} frames under {args.output}")


if __name__ == "__main__":
    main()
