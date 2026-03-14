#!/usr/bin/env python3
"""Plot per-pixel distortion heatmaps comparing V3-2D and GS-NFS renderings.

For the first frames of selected N3DV sequences, this script:
  1. Identifies the best-quality config for V3-2D (qp=0) and GS-NFS (highest
     PSNR on the convex hull).
  2. Checks whether rendered images (GT test view, GT model render, decomp
     render) are already saved; re-runs evaluation with --save_renders if not.
  3. Computes per-pixel distortion maps and produces heatmap figures for four
     comparison pairs per sequence/view:
       (a) GT test view  vs  GS-NFS decomp render
       (b) V3-2D decomp render  vs  GT test view
       (c) GS-NFS decomp render  vs  GT model render
       (d) V3-2D decomp render  vs  GT model render

Usage:
    python scripts/plot_distortion_heatmaps.py
"""

from __future__ import annotations

import csv
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required. Install it (pip install matplotlib)."
    ) from exc

try:
    from PIL import Image
except ImportError as exc:
    raise SystemExit(
        "Pillow is required. Install it (pip install Pillow)."
    ) from exc


# ===========================================================================
# Configuration
# ===========================================================================

SEQUENCES: list[str] = [
    "flame_steak",
    "cook_spinach",
    "cut_roasted_beef",
    "sear_steak",
]

N3DV_DATA_ROOT = "/synology/rajrup/Queen"
FRAME_ID = 1  # N3DV first frame

# V3-2D (VideoGS) — best quality = lowest QP
VIDEOGS_QP = 0
VIDEOGS_GROUP_SIZE = 20

# GS-NFS (LivoGS)
LIVOGS_RD_SUBDIR = "livogs_rd_new"

# Re-evaluation uses the unified VideoGS evaluate_decompress.py with
# --dataset_type n3dv. No queen dependency.
REEVAL_CONDA_ENV = "videogs"
SH_DEGREE = 2
RESOLUTION = 2
LLFFHOLD = 21
RERUN_CUDA_DEVICE = "0"

# When True, re-run evaluation with --save_renders even if renders already
# exist on disk.  When False (default), automatically detect and skip
# sequences whose renders are already saved.
FORCE_RERUN_EVALUATION = False

DRY_RUN = False

# Output
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "distortion_heatmaps"

# Heatmap settings
COLORMAP = "inferno"  # Sequential colormap for distortion magnitude
VMAX_PERCENTILE = 99  # Clip color range at this percentile across all pairs
FIGURE_DPI = 150

# ===========================================================================
# Path helpers
# ===========================================================================

VIDEOGS_PROJECT_ROOT = SCRIPT_DIR.parent  # /home/haodongw/workspace/VideoGS


def model_root(sequence: str) -> Path:
    """N3DV pretrained model root for a sequence."""
    return Path(N3DV_DATA_ROOT) / "pretrained_output" / "Neural_3D_Video" / f"queen_compressed_{sequence}"


def dataset_path(sequence: str) -> str:
    """N3DV raw dataset path for a sequence."""
    return str(Path(N3DV_DATA_ROOT) / "Neural_3D_Video" / sequence)


def gt_model_path(sequence: str) -> str:
    """GT model path (for queen, this is the model root itself)."""
    return str(model_root(sequence))


# ===========================================================================
# V3-2D (VideoGS) best quality finder
# ===========================================================================

def videogs_output_folder(sequence: str) -> Path:
    """Output folder for V3-2D qp=0 on N3DV."""
    tag = f"frames_{FRAME_ID}_{FRAME_ID + VIDEOGS_GROUP_SIZE - 1}_int_1"
    return model_root(sequence) / "compression" / "videogs" / f"qp_{VIDEOGS_QP}" / tag


def videogs_eval_dir(sequence: str) -> Path:
    return videogs_output_folder(sequence) / "evaluation"


# ===========================================================================
# GS-NFS (LivoGS) best quality finder
# ===========================================================================

def _load_hull_csv(csv_path: str) -> list[tuple[float, float]]:
    """Load (compressed_mb, decomp_psnr) points from hull CSV."""
    points: list[tuple[float, float]] = []
    if not os.path.isfile(csv_path):
        return points
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                points.append((float(row["compressed_mb"]), float(row["decomp_psnr"])))
            except (KeyError, ValueError):
                continue
    points.sort(key=lambda p: p[0])
    return points


def _load_sweep_summary(csv_path: str) -> list[dict[str, Any]]:
    """Load sweep summary CSV (all evaluated configs)."""
    rows: list[dict[str, Any]] = []
    if not os.path.isfile(csv_path):
        return rows
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(dict(row))
    return rows


def find_gsnfs_best_quality(sequence: str) -> tuple[str | None, int | None, Path | None]:
    """Find the GS-NFS config with highest decomp_psnr on the hull.

    Returns (label, depth, experiment_dir) or (None, None, None) if not found.
    """
    mroot = model_root(sequence)
    hull_csv = str(mroot / "compression" / LIVOGS_RD_SUBDIR / "plots"
                   / f"acdc_psnr_size_curve_frame{FRAME_ID}_sweep_hull.csv")
    sweep_csv = str(mroot / "compression" / LIVOGS_RD_SUBDIR
                    / f"acdc_hull_sweep_summary_frame{FRAME_ID}.csv")

    hull = _load_hull_csv(hull_csv)
    if not hull:
        print(f"  [WARN] No GS-NFS hull CSV found for {sequence}: {hull_csv}")
        return None, None, None

    # Best quality = highest decomp_psnr
    best_mb, best_psnr = max(hull, key=lambda p: p[1])

    sweep = _load_sweep_summary(sweep_csv)
    if not sweep:
        print(f"  [WARN] No GS-NFS sweep summary found for {sequence}: {sweep_csv}")
        return None, None, None

    # Match hull point to sweep entry
    best_entry = min(
        sweep,
        key=lambda r: (float(r.get("compressed_mb", 0)) - best_mb) ** 2
                     + (float(r.get("decomp_psnr", 0)) - best_psnr) ** 2,
    )
    label = str(best_entry.get("label", ""))
    depth = int(best_entry.get("depth", 0))
    if not label or depth <= 0:
        print(f"  [WARN] Invalid GS-NFS sweep entry for {sequence}: label={label!r}, depth={depth}")
        return None, None, None

    exp_dir = mroot / "compression" / LIVOGS_RD_SUBDIR / f"frame_{FRAME_ID}" / f"J_{depth}" / label
    return label, depth, exp_dir


def gsnfs_eval_dir(sequence: str) -> Path | None:
    """Return the evaluation directory for GS-NFS best quality config."""
    _, _, exp_dir = find_gsnfs_best_quality(sequence)
    if exp_dir is None:
        return None
    return exp_dir / "evaluation"


# ===========================================================================
# Rendered image discovery
# ===========================================================================

RENDER_SUBDIRS = ("gt_images", "gt_model_renders", "decomp_model_renders")
# Glob pattern covers both integer (frame1) and zero-padded (frame0001) naming
IMAGE_GLOB = "frame*_view*_*.png"


def renders_exist(eval_dir: Path) -> bool:
    """Check if all three render subdirectories have at least one image."""
    for subdir in RENDER_SUBDIRS:
        d = eval_dir / subdir
        if not d.is_dir():
            return False
        if not list(d.glob(IMAGE_GLOB)):
            return False
    return True


def discover_views(eval_dir: Path) -> list[int]:
    """Find all available view indices from rendered images."""
    views: set[int] = set()
    for subdir in RENDER_SUBDIRS:
        d = eval_dir / subdir
        if not d.is_dir():
            continue
        for img_path in d.glob(IMAGE_GLOB):
            # Extract view index from: frame{X}_view{idx}_{prefix}.png
            parts = img_path.stem.split("_")
            for i, part in enumerate(parts):
                if part == "view" or (part.startswith("view") and len(part) > 4):
                    # Handle both "frame1_view0_gt_image" and "frame0001_view0_gt_image"
                    if part.startswith("view"):
                        try:
                            views.add(int(part[4:]))
                        except ValueError:
                            pass
                    break
    return sorted(views)


def find_image(eval_dir: Path, subdir: str, view_idx: int) -> Path | None:
    """Find a rendered image by subdirectory and view index.

    Handles both integer and zero-padded frame naming.
    """
    d = eval_dir / subdir
    if not d.is_dir():
        return None
    # Try glob for this view
    matches = sorted(d.glob(f"frame*_view{view_idx}_*.png"))
    return matches[0] if matches else None


# ===========================================================================
# Re-run evaluation with --save_renders
# ===========================================================================

def _run_cmd(cmd: list[str], cwd: Path, cuda_device: str) -> bool:
    """Run a command with CUDA_VISIBLE_DEVICES set. Returns True on success."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_device
    print(f"  [RUN] CUDA_VISIBLE_DEVICES={cuda_device}")
    print(f"  [RUN] cwd={cwd}")
    print(f"  [RUN] {shlex.join(cmd)}")
    if DRY_RUN:
        print("  [DRY RUN] Skipped.")
        return True
    try:
        subprocess.run(cmd, cwd=str(cwd), check=True, env=env)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"  [ERROR] Command failed with exit code {exc.returncode}")
        return False


def _rerun_evaluation(
    sequence: str,
    decompressed_ply_path: Path,
    eval_output_dir: Path,
    method_label: str,
) -> bool:
    """Re-run evaluation via VideoGS evaluate_decompress.py --dataset_type n3dv."""
    print(f"\n  Re-running {method_label} evaluation for {sequence} with --save_renders")
    cmd = [
        "conda", "run", "-n", REEVAL_CONDA_ENV, "python",
        str(VIDEOGS_PROJECT_ROOT / "scripts" / "evaluate_decompress.py"),
        "--dataset_type", "n3dv",
        "--gt_ply_path", "unused",
        "--n3dv_model_path", gt_model_path(sequence),
        "--decompressed_ply_path", str(decompressed_ply_path),
        "--dataset_path", dataset_path(sequence),
        "--sh_degree", str(SH_DEGREE),
        "--resolution", str(RESOLUTION),
        "--llffhold", str(LLFFHOLD),
        "--frame_start", str(FRAME_ID),
        "--frame_end", str(FRAME_ID + 1),
        "--interval", "1",
        "--save_renders",
        "--output_render_path", str(eval_output_dir),
    ]
    return _run_cmd(cmd, cwd=VIDEOGS_PROJECT_ROOT, cuda_device=RERUN_CUDA_DEVICE)


def rerun_videogs_evaluation(sequence: str) -> bool:
    output_folder = videogs_output_folder(sequence)
    return _rerun_evaluation(
        sequence,
        decompressed_ply_path=output_folder / "decompressed_ply",
        eval_output_dir=videogs_eval_dir(sequence),
        method_label="V3-2D",
    )


def rerun_gsnfs_evaluation(sequence: str) -> bool:
    _, _, exp_dir = find_gsnfs_best_quality(sequence)
    if exp_dir is None:
        print(f"  [ERROR] Cannot re-run GS-NFS evaluation: no best config found for {sequence}")
        return False
    print(f"  Experiment dir: {exp_dir}")
    return _rerun_evaluation(
        sequence,
        decompressed_ply_path=exp_dir / "decompressed_ply",
        eval_output_dir=exp_dir / "evaluation",
        method_label="GS-NFS",
    )


# ===========================================================================
# Distortion computation and visualization
# ===========================================================================

def load_image_rgb(path: Path) -> np.ndarray:
    """Load an image as float32 RGB array in [0, 1], shape (H, W, 3)."""
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


def compute_distortion(img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
    """Per-pixel mean absolute error across RGB channels.

    Returns a single-channel (H, W) distortion map in [0, 1].
    """
    return np.mean(np.abs(img_a - img_b), axis=2)


DISTORTION_PAIRS: list[dict[str, str]] = [
    {
        "name": "GT_test_view__vs__GS-NFS",
        "img_a_source": "gsnfs",
        "img_a_subdir": "gt_images",
        "img_b_source": "gsnfs",
        "img_b_subdir": "decomp_model_renders",
        "title": "GT Test View vs GS-NFS",
    },
    {
        "name": "V3-2D__vs__GT_test_view",
        "img_a_source": "v3_2d",
        "img_a_subdir": "decomp_model_renders",
        "img_b_source": "v3_2d",
        "img_b_subdir": "gt_images",
        "title": r"V$^3$-2D vs GT Test View",
    },
    {
        "name": "GS-NFS__vs__GT_model_render",
        "img_a_source": "gsnfs",
        "img_a_subdir": "decomp_model_renders",
        "img_b_source": "gsnfs",
        "img_b_subdir": "gt_model_renders",
        "title": "GS-NFS vs GT Model Render",
    },
    {
        "name": "V3-2D__vs__GT_model_render",
        "img_a_source": "v3_2d",
        "img_a_subdir": "decomp_model_renders",
        "img_b_source": "v3_2d",
        "img_b_subdir": "gt_model_renders",
        "title": r"V$^3$-2D vs GT Model Render",
    },
]


def _collect_vmax(
    v3_2d_eval_dir: Path,
    gsnfs_eval_dir_path: Path,
    views: list[int],
) -> float:
    """Compute a shared vmax across all distortion pairs and views.

    Uses the VMAX_PERCENTILE of all distortion values combined.
    """
    all_distortions: list[np.ndarray] = []
    sources = {"v3_2d": v3_2d_eval_dir, "gsnfs": gsnfs_eval_dir_path}

    for pair in DISTORTION_PAIRS:
        for view_idx in views:
            img_a_path = find_image(sources[pair["img_a_source"]], pair["img_a_subdir"], view_idx)
            img_b_path = find_image(sources[pair["img_b_source"]], pair["img_b_subdir"], view_idx)
            if img_a_path is None or img_b_path is None:
                continue
            img_a = load_image_rgb(img_a_path)
            img_b = load_image_rgb(img_b_path)
            if img_a.shape != img_b.shape:
                print(f"  [WARN] Shape mismatch: {img_a_path.name} {img_a.shape} vs {img_b_path.name} {img_b.shape}")
                continue
            dist = compute_distortion(img_a, img_b)
            all_distortions.append(dist)

    if not all_distortions:
        return 0.1  # fallback
    combined = np.concatenate([d.ravel() for d in all_distortions])
    return float(np.percentile(combined, VMAX_PERCENTILE))


def plot_distortion_heatmaps_for_sequence(
    sequence: str,
    v3_2d_eval: Path,
    gsnfs_eval: Path,
    output_dir: Path,
) -> None:
    """Generate distortion heatmap figures for one sequence."""

    # Discover views from both methods
    v3_2d_views = discover_views(v3_2d_eval)
    gsnfs_views = discover_views(gsnfs_eval)
    common_views = sorted(set(v3_2d_views) & set(gsnfs_views))

    if not common_views:
        print(f"  [WARN] No common views between V3-2D and GS-NFS for {sequence}")
        print(f"         V3-2D views: {v3_2d_views}")
        print(f"         GS-NFS views: {gsnfs_views}")
        return

    print(f"  Views: {common_views}")

    # Compute shared color range across all pairs and views
    vmax = _collect_vmax(v3_2d_eval, gsnfs_eval, common_views)
    print(f"  Shared vmax ({VMAX_PERCENTILE}th percentile): {vmax:.4f}")

    sources = {"v3_2d": v3_2d_eval, "gsnfs": gsnfs_eval}
    seq_output_dir = output_dir / sequence
    os.makedirs(seq_output_dir, exist_ok=True)

    # --- Per-view composite figure: 4 pairs in one row ---
    for view_idx in common_views:
        fig, axes = plt.subplots(1, len(DISTORTION_PAIRS), figsize=(5 * len(DISTORTION_PAIRS), 5))
        if len(DISTORTION_PAIRS) == 1:
            axes = [axes]

        has_any = False
        for ax, pair in zip(axes, DISTORTION_PAIRS):
            img_a_path = find_image(sources[pair["img_a_source"]], pair["img_a_subdir"], view_idx)
            img_b_path = find_image(sources[pair["img_b_source"]], pair["img_b_subdir"], view_idx)

            if img_a_path is None or img_b_path is None:
                missing = []
                if img_a_path is None:
                    missing.append(f"{pair['img_a_source']}/{pair['img_a_subdir']}")
                if img_b_path is None:
                    missing.append(f"{pair['img_b_source']}/{pair['img_b_subdir']}")
                ax.text(0.5, 0.5, f"Missing:\n{chr(10).join(missing)}",
                        transform=ax.transAxes, ha="center", va="center", fontsize=9)
                ax.set_title(pair["title"], fontsize=11)
                ax.axis("off")
                continue

            img_a = load_image_rgb(img_a_path)
            img_b = load_image_rgb(img_b_path)

            if img_a.shape != img_b.shape:
                ax.text(0.5, 0.5, f"Shape mismatch\n{img_a.shape} vs {img_b.shape}",
                        transform=ax.transAxes, ha="center", va="center", fontsize=9)
                ax.set_title(pair["title"], fontsize=11)
                ax.axis("off")
                continue

            dist = compute_distortion(img_a, img_b)
            dist_max = dist.max() if dist.max() > 0 else 1.0
            dist_normalized = dist / dist_max
            im = ax.imshow(dist_normalized, cmap=COLORMAP, vmin=0, vmax=1.0, interpolation="nearest")
            ax.set_title(pair["title"], fontsize=11)
            ax.axis("off")

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax, label="MAE")

            has_any = True

        if has_any:
            fig.suptitle(f"{sequence} — View {view_idx}", fontsize=14, y=1.02)
            fig.tight_layout()

            out_png = seq_output_dir / f"distortion_{sequence}_view{view_idx}.png"
            fig.savefig(out_png, dpi=FIGURE_DPI, bbox_inches="tight")
            print(f"  Saved: {out_png}")

            out_pdf = seq_output_dir / f"distortion_{sequence}_view{view_idx}.pdf"
            fig.savefig(out_pdf, bbox_inches="tight")
            print(f"  Saved: {out_pdf}")

        plt.close(fig)

    # --- Per-pair individual images (higher resolution) ---
    for view_idx in common_views:
        for pair in DISTORTION_PAIRS:
            img_a_path = find_image(sources[pair["img_a_source"]], pair["img_a_subdir"], view_idx)
            img_b_path = find_image(sources[pair["img_b_source"]], pair["img_b_subdir"], view_idx)
            if img_a_path is None or img_b_path is None:
                continue

            img_a = load_image_rgb(img_a_path)
            img_b = load_image_rgb(img_b_path)
            if img_a.shape != img_b.shape:
                continue

            dist = compute_distortion(img_a, img_b)

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            axes[0].imshow(img_a)
            axes[0].set_title(f"{pair['img_a_subdir'].replace('_', ' ').title()}", fontsize=12)
            axes[0].axis("off")

            axes[1].imshow(img_b)
            axes[1].set_title(f"{pair['img_b_subdir'].replace('_', ' ').title()}", fontsize=12)
            axes[1].axis("off")

            dist_max = dist.max() if dist.max() > 0 else 1.0
            dist_normalized = dist / dist_max
            im = axes[2].imshow(dist_normalized, cmap=COLORMAP, vmin=0, vmax=1.0, interpolation="nearest")
            axes[2].set_title("Distortion (MAE)", fontsize=12)
            axes[2].axis("off")
            divider = make_axes_locatable(axes[2])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)

            fig.suptitle(f"{pair['title']} — {sequence} View {view_idx}", fontsize=14, y=1.02)
            fig.tight_layout()

            name = pair["name"]
            out_path = seq_output_dir / f"distortion_{name}_view{view_idx}.png"
            fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
            plt.close(fig)


# ===========================================================================
# Main
# ===========================================================================

def _need_rerun(eval_dir: Path) -> bool:
    """True when renders are absent or FORCE_RERUN_EVALUATION is set."""
    if FORCE_RERUN_EVALUATION:
        return True
    return not renders_exist(eval_dir)


def _ensure_renders(
    eval_dir: Path,
    method_label: str,
    rerun_fn: Callable[[str], bool],
    sequence: str,
) -> bool:
    """Return True when renders are available, running *rerun_fn* if needed."""
    if not _need_rerun(eval_dir):
        print(f"  {method_label} renders found (skip): {eval_dir}")
        return True

    reason = "FORCE_RERUN_EVALUATION" if renders_exist(eval_dir) else "renders missing"
    print(f"  {method_label}: {reason} — re-running evaluation with --save_renders")
    ok = rerun_fn(sequence)
    if not ok:
        print(f"  [ERROR] {method_label} re-evaluation failed for {sequence}")
        return False
    if not DRY_RUN and not renders_exist(eval_dir):
        print(f"  [ERROR] {method_label} renders still not found after re-evaluation")
        return False
    return True


def main() -> None:
    sep = "=" * 70
    print(sep)
    print("Distortion Heatmap Generator")
    print(f"  Sequences:          {', '.join(SEQUENCES)}")
    print(f"  Frame ID:           {FRAME_ID}")
    print(f"  V3-2D QP:           {VIDEOGS_QP}")
    print(f"  Force re-eval:      {FORCE_RERUN_EVALUATION}")
    print(f"  Re-eval conda env:  {REEVAL_CONDA_ENV}")
    print(f"  Output:             {OUTPUT_DIR}")
    print(f"  Dry run:            {DRY_RUN}")
    print(sep)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for sequence in SEQUENCES:
        print(f"\n{'─' * 70}")
        print(f"Processing: {sequence}")
        print(f"{'─' * 70}")

        # ── V3-2D ──────────────────────────────────────────────────────
        v3_2d_out = videogs_output_folder(sequence)
        v3_2d_eval = videogs_eval_dir(sequence)
        print(f"\n  V3-2D output: {v3_2d_out}")

        if not v3_2d_out.exists():
            print(f"  [ERROR] V3-2D output folder does not exist: {v3_2d_out}")
            print(f"          Run the VideoGS experiment with qp={VIDEOGS_QP} first.")
            continue

        if not _ensure_renders(v3_2d_eval, "V3-2D", rerun_videogs_evaluation, sequence):
            continue

        # ── GS-NFS ────────────────────────────────────────────────────
        label, depth, gsnfs_exp_dir = find_gsnfs_best_quality(sequence)
        if gsnfs_exp_dir is None:
            print(f"  [ERROR] Cannot find GS-NFS best quality config for {sequence}")
            continue
        gsnfs_eval = gsnfs_exp_dir / "evaluation"
        print(f"\n  GS-NFS best config: label={label}, depth={depth}")
        print(f"  GS-NFS experiment: {gsnfs_exp_dir}")

        if not gsnfs_exp_dir.exists():
            print(f"  [ERROR] GS-NFS experiment dir does not exist: {gsnfs_exp_dir}")
            continue

        if not _ensure_renders(gsnfs_eval, "GS-NFS", rerun_gsnfs_evaluation, sequence):
            continue

        # ── Plot heatmaps ─────────────────────────────────────────────
        if DRY_RUN:
            print(f"\n  [DRY RUN] Would generate heatmaps for {sequence}")
            continue

        print(f"\n  Generating distortion heatmaps for {sequence} ...")
        plot_distortion_heatmaps_for_sequence(sequence, v3_2d_eval, gsnfs_eval, OUTPUT_DIR)

    print(f"\n{sep}")
    print(f"Done! Heatmaps saved to: {OUTPUT_DIR}")
    print(sep)


if __name__ == "__main__":
    main()
