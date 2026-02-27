#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from errno import EEXIST
from os import makedirs, path
import os

def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

def searchForMaxIteration(folder):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Checkpoint folder not found: {folder}")

    saved_iters = []
    for fname in os.listdir(folder):
        if not fname.startswith("iteration_"):
            continue
        suffix = fname.split("_")[-1]
        if not suffix.isdigit():
            continue

        iter_dir = os.path.join(folder, fname)
        if not os.path.isdir(iter_dir):
            continue

        ply_path = os.path.join(iter_dir, "point_cloud.ply")
        if not os.path.isfile(ply_path):
            continue

        saved_iters.append(int(suffix))

    if not saved_iters:
        raise FileNotFoundError(
            "No valid checkpoint iterations found in "
            f"'{folder}'. Expected directories like "
            "iteration_<N>/point_cloud.ply."
        )

    return max(saved_iters)
