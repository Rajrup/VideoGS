# ActorsHQ (HumanRF) Preprocessing for VideoGS

This document describes how to convert **ActorsHQ** (also known as **HumanRF**) data with per-frame COLMAP calibration into the format required by VideoGS training.

## Prerequisites

- **COLMAP** run per frame on your ActorsHQ sequence (you should have `sparse/` with `cameras.bin`, `images.bin`, and `points3D.bin` under each frame).
- Python environment with dependencies from the VideoGS project (e.g. `numpy`, `PIL`, `tqdm`). Use the same conda env as for training:
  ```bash
  conda activate videogs
  ```

## Expected Input Layout

ActorsHQ is organized by actor, sequence, and resolution. After running COLMAP **per frame**, the directory layout should look like:

```
<actor_base>/Sequence<id>/<resolution>x/frames/
├── frame0/
│   ├── images/           # Multiview RGB images
│   │   ├── Cam001_rgb000000.png
│   │   ├── Cam002_rgb000000.png
│   │   └── ...           # Cam ID 1–160 (or subset used in COLMAP)
│   ├── masks/            # Optional; used for white-background compositing
│   │   ├── Cam001_mask000000.png
│   │   └── ...
│   └── sparse/           # COLMAP output for this frame
│       ├── cameras.bin
│       ├── images.bin
│       └── points3D.bin
├── frame1/
│   └── ...
└── ...
```

- **Images**: `Cam<id>_rgb000000.png` (camera index, e.g. 001–160).
- **Masks**: `Cam<id>_mask000000.png`; same naming convention as images. If present and `--white_background` is used, black background is replaced with white.
- **COLMAP**: One reconstruction per frame in that frame’s `sparse/` folder.

## Output Layout (VideoGS Format)

The script produces the same structure as the processed HiFi4G dataset:

```
<output>/
├── 0/                    # Frame index (integer)
│   ├── images/
│   │   ├── 0.png
│   │   ├── 1.png
│   │   └── ...
│   └── transforms.json
├── 1/
│   ├── images/
│   │   └── ...
│   └── transforms.json
└── ...
```

- Frame folders are named by **integer index** (0, 1, 2, …) so that training can use `--data <output>` and load each frame as `<output>/<frame_id>/`.
- Each frame has its own `transforms.json` (ngp/Blender-style) and `images/<view_idx>.png` with consistent view ordering.
- **View order matches camera ID:** the script sorts views by the camera number in the filename (Cam001 → 1, Cam002 → 2, …), so `images/0.png` = Cam001, `images/1.png` = Cam002, etc. The same order is used in `transforms.json`, so pose for `frames[i]` corresponds to `images/i.png`. This avoids COLMAP’s internal image_id order, which is not the same as Cam001–Cam160.

## Usage

From the **VideoGS repo root** or from `preprocess/`:

```bash
cd preprocess
python actorshq_preprocess.py --input <frames_dir> --output <output_dir> [options]
```

### Example (Actor08, Sequence1, 4x)

```bash
cd /home/rajrup/Project/VideoGS/preprocess

python actorshq_preprocess.py \
  --input /synology/actorshq/colmap/Actor08/Sequence1/4x/frames \
  --output /synology/rajrup/VideoGS/ActorsHQ_Dataset_processed/Actor08/Sequence1/4x
```

To process only a range of frames (e.g. for testing):

```bash
python actorshq_preprocess.py --input /synology/actorshq/colmap/Actor08/Sequence1/4x/frames --output /synology/rajrup/VideoGS/ActorsHQ_Dataset_processed2/Actor08_Sequence1_4x --start 0 --end 39
```

## Command-Line Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--input` | Yes | Path to the **frames** directory (e.g. `.../Actor08/Sequence1/4x/frames`). |
| `--output` | Yes | Path to the output dataset directory (e.g. `.../ActorsHQ_Dataset_processed/Actor08/Sequence1/4x`). |
| `--start` | No | First frame index (0-based). Omit to process from the first frame. |
| `--end` | No | Last frame index (inclusive). Omit to process to the last frame. |
| `--white_background` | No | Use masks to composite white background (default: **True**). |
| `--no_white_background` | No | Do not use masks; keep original (e.g. black) background. |
| `--move` | No | If `True`, move source images; if `False`, copy (default: **False**). |

## White background

- ActorsHQ images often have **black background**. VideoGS training typically expects **white background** (as in HiFi4G).
- With **`--white_background`** (default), the script looks for `masks/Cam<id>_mask000000.png` for each image `Cam<id>_rgb000000.png` and composites: foreground from RGB, background white.
- If a mask is missing for a view, that image is copied (or moved) as-is.
- Use **`--no_white_background`** to skip compositing and keep the original pixels.

## Training After Preprocessing

Use the **output** path as `--data` for `train_sequence.py`, and the same frame range you preprocessed (e.g. `--start` / `--end`):

```bash
python train_sequence.py \
  --start 0 --end 2199 \
  --cuda 0 \
  --data /synology/rajrup/VideoGS/ActorsHQ_Dataset_processed/Actor08/Sequence1/4x \
  --output /synology/rajrup/VideoGS/train_output/ActorsHQ_Dataset/Actor08/Sequence1/4x \
  --sh 3 --interval 1 --group_size 20 --resolution 1
```

Adjust `--start` / `--end` and other training flags as in your README_Rajrup.md.

## Implementation Notes

- The script uses the existing **`colmap2k.py`** in `preprocess/` to convert each frame’s COLMAP binary (`sparse/cameras.bin`, `sparse/images.bin`) to a single `transforms.json`. The binary files are copied to a temporary directory so the original `sparse/` is not modified.
- View order in `transforms.json` and `images/0.png`, `1.png`, … follows the order of images in COLMAP’s reconstruction for that frame.
- Frame folders in the output are named by integer index so that `dataset.source_path` + `str(frame)` in the training code points to the correct frame directory.
