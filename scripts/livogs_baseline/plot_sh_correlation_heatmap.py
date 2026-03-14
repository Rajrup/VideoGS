#!/usr/bin/env python3
"""
Plot correlation heatmaps of SH channels before and after KLT transform.

Loads a VideoGS-trained PLY, runs it through the LiVoGS voxelize + merge
pipeline, then plots the Pearson correlation matrix of the 48 SH channels
in RGB space vs. KLT space side-by-side.

Usage:
    python scripts/livogs_baseline/plot_sh_correlation_heatmap.py \
        --ply_path /synology/rajrup/VideoGS/train_output/HiFi4G_Dataset/4K_Actor1_Greeting/checkpoint/0/point_cloud/iteration_16000/point_cloud.ply \
        --title "4K_Actor1_Greeting frame 0" \
        --output_dir results/sh_correlation
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Project imports (mirrors analyze_sh_energy_and_qp.py)
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_VIDEOGS_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_LIVOGS_COMPRESSION = os.path.join(_VIDEOGS_ROOT, "LiVoGS", "compression")
_EXTRA_PYTHON_DIRS = [
    _VIDEOGS_ROOT,
    os.path.join(_LIVOGS_COMPRESSION, "RAHT-3DGS-codec", "python"),
    _LIVOGS_COMPRESSION,
]
for path in _EXTRA_PYTHON_DIRS:
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

from data_util import load_3dgs
from voxelize_pc import voxelize_pc
from merge_cluster_cuda import merge_gaussian_clusters_with_indices
from color_space_transforms import normalize_attributes, rgb_to_klt15
from gpu_octree_codec import calc_morton


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sh_basis_labels(num_basis: int) -> list[str]:
    """One label per SH basis function (e.g. 'DC', 'Y₁⁰', 'Y₂⁻²', …)."""
    degree_sizes = [1, 3, 5, 7]
    superscript = {"-3": "⁻³", "-2": "⁻²", "-1": "⁻¹", "0": "⁰",
                   "1": "¹", "2": "²", "3": "³"}
    labels: list[str] = []
    basis_idx = 0
    for deg, size in enumerate(degree_sizes):
        if basis_idx >= num_basis:
            break
        for k in range(size):
            if basis_idx >= num_basis:
                break
            m = k - (size // 2)
            if deg == 0:
                labels.append("DC")
            else:
                labels.append(f"Y{deg}{superscript[str(m)]}")
            basis_idx += 1
    return labels


def _sh_degree_boundaries(num_basis: int) -> list[int]:
    """Return channel indices where SH degree changes (for grid lines)."""
    degree_sizes = [1, 3, 5, 7]
    boundaries: list[int] = []
    basis_idx = 0
    for size in degree_sizes:
        if basis_idx >= num_basis:
            break
        boundaries.append(basis_idx * 3)
        basis_idx += size
    return boundaries


def _pearson_corrcoef(data: torch.Tensor) -> np.ndarray:
    """Pearson correlation matrix for columns of *data* (N×C) -> C×C numpy array."""
    data = data.double()
    mean = data.mean(dim=0, keepdim=True)
    centered = data - mean
    cov = (centered.T @ centered) / (data.shape[0] - 1)
    std = cov.diag().sqrt().clamp(min=1e-12)
    corr = cov / (std.unsqueeze(0) * std.unsqueeze(1))
    return corr.cpu().numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SH channel correlation heatmaps before/after KLT",
    )
    parser.add_argument("--ply_path", type=str, required=True,
                        help="Path to a VideoGS-trained point_cloud.ply")
    parser.add_argument("--J", type=int, default=15,
                        help="Octree depth for voxelization (default: 15)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--title", type=str, default=None,
                        help="Plot suptitle (default: derived from ply_path)")
    parser.add_argument("--output_dir", type=str, default="results/sh_correlation",
                        help="Directory for output PDF")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    device = args.device
    J = args.J
    device_id = int(device.split(":")[1]) if ":" in device else 0

    # --- Load ---
    print(f"Loading PLY: {args.ply_path}")
    params = load_3dgs(args.ply_path, device=device)
    N = params["means"].shape[0]
    print(f"  Gaussians: {N}")

    # --- Voxelize ---
    V_means = params["means"]
    vmin = V_means.min(dim=0)[0]
    V0 = V_means - vmin.unsqueeze(0)
    width = V0.max()
    voxel_size = width / (2.0 ** J)
    V0_integer = torch.clamp(torch.floor(V0 / voxel_size).long(), 0, 2**J - 1).int()

    morton_result = calc_morton(
        V0_integer, voxel_grid_depth=J, force_64bit_codes=True,
        device=device_id, return_torch=True,
    )
    morton_codes = morton_result["morton_codes"]
    if morton_codes.dtype == torch.uint64:
        morton_codes = morton_codes.to(torch.int64)

    PCvox, PCsorted, voxel_indices, DeltaPC, voxel_info = voxelize_pc(
        params["means"], vmin=vmin, width=width, J=J,
        device=device, morton_codes=morton_codes,
    )

    sort_idx = voxel_info["sort_idx"]
    cluster_indices = sort_idx.int()
    cluster_offsets = torch.cat([
        voxel_indices,
        torch.tensor([N], dtype=torch.int32, device=device),
    ]).int()

    # --- Merge ---
    _, _, _, _, merged_colors = merge_gaussian_clusters_with_indices(
        params["means"], params["quats"], params["scales"],
        params["opacities"], params["colors"],
        cluster_indices, cluster_offsets, weight_by_opacity=True,
    )

    Nvox = merged_colors.shape[0]
    num_sh_channels = merged_colors.shape[1]
    num_basis = num_sh_channels // 3
    print(f"  Voxels: {Nvox},  SH channels: {num_sh_channels} ({num_basis} basis × 3)")

    # --- Normalize ---
    colors_norm, _ = normalize_attributes(merged_colors)

    # --- Correlation before KLT (RGB space) ---
    print("Computing correlation (RGB)...")
    corr_rgb = _pearson_corrcoef(colors_norm)

    # --- Apply KLT15 ---
    # Output layout: [DC_Y, DC_U, DC_V | Y_PC₁…PC₁₅ | U_PC₁…PC₁₅ | V_PC₁…PC₁₅]
    print("Applying KLT15 transform (BT.709 YUV → reorder → per-band 15×15 PCA)...")
    colors_klt, klt15_matrices = rgb_to_klt15(colors_norm)

    # --- Correlation after KLT15 ---
    print("Computing correlation (KLT15)...")
    corr_klt = _pearson_corrcoef(colors_klt)

    # --- Shared constants ---
    C = num_sh_channels
    num_higher = num_basis - 1  # 15 for SH degree 3
    cmap = "RdBu_r"
    strip_w = 0.55
    bracket_x = -3.5

    # --- RGB panel annotations ---
    rgb_basis_labels = _sh_basis_labels(num_basis)
    rgb_boundaries = _sh_degree_boundaries(num_basis)
    rgb_full_boundaries = rgb_boundaries + [C]
    rgb_degree_labels = ["ℓ=0", "ℓ=1", "ℓ=2", "ℓ=3"]
    rgb_strip_colors = ["#e74c3c", "#2ecc71", "#3498db"]

    # --- KLT15 panel annotations ---
    # Boundaries: DC(3) | Y-PCs(15) | U-PCs(15) | V-PCs(15)
    klt_boundaries = [0, 3, 3 + num_higher, 3 + 2 * num_higher, C]
    klt_band_labels = ["DC", "Y", "U", "V"]
    klt_band_colors = ["#7f8c8d", "#2c3e50", "#e67e22", "#9b59b6"]
    # Per-channel tick labels
    klt_tick_labels: list[str] = ["Y", "U", "V"]
    subscript = {str(i): chr(0x2080 + i) for i in range(10)}
    subscript.update({"10": "₁₀", "11": "₁₁", "12": "₁₂", "13": "₁₃",
                      "14": "₁₄", "15": "₁₅"})
    for band in ["Y", "U", "V"]:
        for pc in range(1, num_higher + 1):
            klt_tick_labels.append(f"{band}{subscript[str(pc)]}")

    os.makedirs(args.output_dir, exist_ok=True)
    safe_title = (args.title or "heatmap").replace(" ", "_").replace("/", "_")
    title = args.title or os.path.basename(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(args.ply_path)))))
    subtitle = f"J={J}, {Nvox:,} voxels"

    # ================================================================
    # Figure 1 — RGB (before KLT)
    # ================================================================
    fig_rgb, ax_rgb = plt.subplots(figsize=(12, 10.5))
    ax_rgb.imshow(corr_rgb, vmin=-1.0, vmax=1.0, cmap=cmap,
                  interpolation="nearest", aspect="equal")
    ax_rgb.set_title(f"SH Channel Correlation — RGB\n{title}  ({subtitle})",
                     fontsize=15, fontweight="bold", pad=18)

    for bnd in rgb_boundaries[1:]:
        ax_rgb.axhline(bnd - 0.5, color="black", linewidth=1.4, alpha=0.6)
        ax_rgb.axvline(bnd - 0.5, color="black", linewidth=1.4, alpha=0.6)

    tick_pos = [i * 3 + 1 for i in range(num_basis)]
    ax_rgb.set_xticks(tick_pos)
    ax_rgb.set_xticklabels(rgb_basis_labels, rotation=55, ha="right", fontsize=9)
    ax_rgb.set_yticks(tick_pos)
    ax_rgb.set_yticklabels(rgb_basis_labels, fontsize=9)

    for b in range(num_basis):
        for c in range(3):
            ch = b * 3 + c
            ax_rgb.add_patch(Rectangle(
                (ch - 0.5, -strip_w - 0.5), 1, strip_w,
                facecolor=rgb_strip_colors[c], clip_on=False, zorder=5))
            ax_rgb.add_patch(Rectangle(
                (-strip_w - 0.5, ch - 0.5), strip_w, 1,
                facecolor=rgb_strip_colors[c], clip_on=False, zorder=5))

    for i in range(len(rgb_full_boundaries) - 1):
        s = rgb_full_boundaries[i] - 0.5
        e = rgb_full_boundaries[i + 1] - 0.5
        mid = (s + e) / 2.0
        ax_rgb.annotate("", xy=(bracket_x + 0.6, s), xytext=(bracket_x + 0.6, e),
                        arrowprops=dict(arrowstyle="-", lw=1.2, color="black"),
                        clip_on=False, annotation_clip=False)
        ax_rgb.text(bracket_x, mid, rgb_degree_labels[i],
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    rotation=90, clip_on=False)

    im_rgb = ax_rgb.images[0]
    fig_rgb.colorbar(im_rgb, ax=ax_rgb, fraction=0.046, pad=0.04, label="Pearson correlation")

    rgb_path = os.path.join(args.output_dir, f"sh_correlation_rgb_{safe_title}_J{J}.pdf")
    fig_rgb.savefig(rgb_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig_rgb)
    print(f"Saved: {rgb_path}")

    # ================================================================
    # Figure 2 — KLT15 (after KLT)
    # ================================================================
    fig_klt, ax_klt = plt.subplots(figsize=(12, 10.5))
    ax_klt.imshow(corr_klt, vmin=-1.0, vmax=1.0, cmap=cmap,
                  interpolation="nearest", aspect="equal")
    ax_klt.set_title(f"SH Channel Correlation — KLT15\n{title}  ({subtitle})",
                     fontsize=15, fontweight="bold", pad=18)

    for bnd in klt_boundaries[1:-1]:
        ax_klt.axhline(bnd - 0.5, color="black", linewidth=1.4, alpha=0.6)
        ax_klt.axvline(bnd - 0.5, color="black", linewidth=1.4, alpha=0.6)

    ax_klt.set_xticks(range(C))
    ax_klt.set_xticklabels(klt_tick_labels, rotation=55, ha="right", fontsize=8)
    ax_klt.set_yticks(range(C))
    ax_klt.set_yticklabels(klt_tick_labels, fontsize=8)

    for i in range(len(klt_boundaries) - 1):
        for ch in range(klt_boundaries[i], klt_boundaries[i + 1]):
            ax_klt.add_patch(Rectangle(
                (ch - 0.5, -strip_w - 0.5), 1, strip_w,
                facecolor=klt_band_colors[i], clip_on=False, zorder=5))
            ax_klt.add_patch(Rectangle(
                (-strip_w - 0.5, ch - 0.5), strip_w, 1,
                facecolor=klt_band_colors[i], clip_on=False, zorder=5))

    for i in range(len(klt_boundaries) - 1):
        s = klt_boundaries[i] - 0.5
        e = klt_boundaries[i + 1] - 0.5
        mid = (s + e) / 2.0
        ax_klt.annotate("", xy=(bracket_x + 0.6, s), xytext=(bracket_x + 0.6, e),
                        arrowprops=dict(arrowstyle="-", lw=1.2, color="black"),
                        clip_on=False, annotation_clip=False)
        ax_klt.text(bracket_x, mid, klt_band_labels[i],
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    rotation=90, clip_on=False)

    im_klt = ax_klt.images[0]
    fig_klt.colorbar(im_klt, ax=ax_klt, fraction=0.046, pad=0.04, label="Pearson correlation")

    klt_path = os.path.join(args.output_dir, f"sh_correlation_klt15_{safe_title}_J{J}.pdf")
    fig_klt.savefig(klt_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig_klt)
    print(f"Saved: {klt_path}")

    # ================================================================
    # Figure 3 — RGB clean (no annotations, title only)
    # ================================================================
    fig_rc, ax_rc = plt.subplots(figsize=(10, 10))
    ax_rc.imshow(corr_rgb, vmin=-1.0, vmax=1.0, cmap=cmap,
                 interpolation="nearest", aspect="equal")
    ax_rc.set_title("RGB", fontsize=24, fontweight="bold", pad=10)
    ax_rc.set_xticks([])
    ax_rc.set_yticks([])
    cbar_rc = fig_rc.colorbar(ax_rc.images[0], ax=ax_rc, fraction=0.046, pad=0.04)
    cbar_rc.ax.set_ylabel("Pearson correlation", fontsize=24)
    cbar_rc.ax.tick_params(labelsize=20)

    rgb_clean_path = os.path.join(args.output_dir, f"sh_correlation_rgb_clean_{safe_title}_J{J}.pdf")
    fig_rc.savefig(rgb_clean_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig_rc)
    print(f"Saved: {rgb_clean_path}")

    # ================================================================
    # Figure 4 — KLT15 clean (no annotations, title only)
    # ================================================================
    fig_kc, ax_kc = plt.subplots(figsize=(10, 10))
    ax_kc.imshow(corr_klt, vmin=-1.0, vmax=1.0, cmap=cmap,
                 interpolation="nearest", aspect="equal")
    ax_kc.set_title("KLT", fontsize=24, fontweight="bold", pad=10)
    ax_kc.set_xticks([])
    ax_kc.set_yticks([])
    cbar_kc = fig_kc.colorbar(ax_kc.images[0], ax=ax_kc, fraction=0.046, pad=0.04)
    cbar_kc.ax.set_ylabel("Pearson correlation", fontsize=24)
    cbar_kc.ax.tick_params(labelsize=20)

    klt_clean_path = os.path.join(args.output_dir, f"sh_correlation_klt15_clean_{safe_title}_J{J}.pdf")
    fig_kc.savefig(klt_clean_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig_kc)
    print(f"Saved: {klt_clean_path}")


if __name__ == "__main__":
    main()
