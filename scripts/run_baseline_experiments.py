#!/usr/bin/env python3
"""Run selected baseline compression experiments.

This is the Python orchestrator for DracoGS, MesonGS, and VideoGS baseline
pipelines. Keep hardcoded settings in the global configuration section below.

Usage:
  python scripts/run_selected_experiments.py
  python scripts/run_selected_experiments.py --dry-run

Wrapper:
  bash scripts/run_selected_experiments.sh [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable


BASELINE_ENVS: dict[str, str] = {
    "dracogs": "videogs",
    "mesongs": "mesongs",
    "videogs": "videogs",
}
BASELINES = list(BASELINE_ENVS.keys())
EVALUATION_ENV = "videogs"
CUDA_DEVICE = "0"

DATASET_NAME = "HiFi4G_Dataset"
RESOLUTION = 2
SH_DEGREE = 3
DATA_PATH = "/synology/rajrup/VideoGS"
DRACOGS_EG = 16
DRACOGS_EO = 16
DRACOGS_ET = 16
DRACOGS_ES = 16
DRACOGS_CL = 10
VIDEOGS_QP = 25
VIDEOGS_GROUP_SIZE = 20

EXPERIMENTS: dict[str, list[int]] = {
    "4K_Actor1_Greeting": [0, 50, 100, 150],
    # "4K_Actor2_Dancing": [0, 50, 100, 150],
    # "4K_Actor3_Violin": [0, 50, 100, 150],
    # "4K_Actor4_Dancing": [0, 50, 100, 150],
    # "4K_Actor5_Oil-paper_Umbrella": [0, 50, 100, 150],
    # "4K_Actor6_Changing_Clothes": [0, 50, 100, 150],
    # "4K_Actor7_Nunchaku": [0, 50, 100, 150],
}

SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parent
VIDEOGS_ROOT = SCRIPTS_DIR.parent
MESONGS_ROOT = VIDEOGS_ROOT / "MesonGS"


@dataclass(frozen=True)
class ExperimentPaths:
    dataset_path: str
    gt_model_path: str
    output_folder: str


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_header(message: str) -> None:
    print("")
    print("=" * 70)
    print(f"  {message}")
    print("=" * 70)


def log_step(message: str) -> None:
    print(f"--- {message}")


def run_cmd(cmd: list[str], cwd: Path, dry_run: bool) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)
    if dry_run:
        print(
            f"[DRY RUN] cwd={cwd} | CUDA_VISIBLE_DEVICES={CUDA_DEVICE} | "
            f"{shlex.join(cmd)}"
        )
        return
    subprocess.run(cmd, cwd=str(cwd), check=True, env=env)


def conda_python_cmd(env_name: str, script_path: Path, args: list[str]) -> list[str]:
    return ["conda", "run", "-n", env_name, "python", str(script_path), *args]


def get_available_conda_envs() -> set[str]:
    proc = subprocess.run(
        ["conda", "env", "list", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)
    env_paths = payload.get("envs", [])
    return {Path(p).name for p in env_paths}


def ensure_required_envs(baselines: list[str]) -> None:
    required = {EVALUATION_ENV}
    required.update(BASELINE_ENVS[baseline] for baseline in baselines)

    available = get_available_conda_envs()
    missing = sorted(env for env in required if env not in available)
    if missing:
        raise RuntimeError(
            "Missing required conda environment(s): "
            + ", ".join(missing)
            + ". Please create/install them before running experiments."
        )


def ensure_videogs_frame_spacing(experiments: dict[str, list[int]], gop: int) -> None:
    for sequence, frame_ids in experiments.items():
        if len(frame_ids) < 2:
            continue

        sorted_ids = sorted(frame_ids)
        for prev_id, next_id in zip(sorted_ids, sorted_ids[1:]):
            gap = next_id - prev_id
            if gap < gop:
                raise RuntimeError(
                    f"Invalid VideoGS frame selection for '{sequence}': "
                    f"gap {gap} between frames {prev_id} and {next_id} "
                    f"must be greater than or equal to GOP ({gop})."
                )


def get_output_folder(baseline: str, sequence: str) -> str:
    base = f"{DATA_PATH}/train_output/{DATASET_NAME}/{sequence}/compression"
    if baseline == "dracogs":
        return (
            f"{base}/dracogs/"
            f"eg_{DRACOGS_EG}_eo_{DRACOGS_EO}_et_{DRACOGS_ET}_es_{DRACOGS_ES}_cl_{DRACOGS_CL}"
        )
    if baseline == "mesongs":
        return f"{base}/mesongs/params_default"
    if baseline == "videogs":
        return f"{base}/videogs/qp_{VIDEOGS_QP}"
    raise ValueError(f"Unknown baseline '{baseline}'")


def get_paths(sequence: str, baseline: str) -> ExperimentPaths:
    return ExperimentPaths(
        dataset_path=f"{DATA_PATH}/{DATASET_NAME}_processed/{sequence}",
        gt_model_path=f"{DATA_PATH}/train_output/{DATASET_NAME}/{sequence}/checkpoint",
        output_folder=get_output_folder(baseline, sequence),
    )


def run_evaluation(paths: ExperimentPaths, frame_ids_csv: str, dry_run: bool) -> None:
    cmd = conda_python_cmd(
        EVALUATION_ENV,
        VIDEOGS_ROOT / "scripts" / "evaluate_decompress.py",
        [
            "--gt_ply_path",
            paths.gt_model_path,
            "--decompressed_ply_path",
            f"{paths.output_folder}/decompressed_ply",
            "--dataset_path",
            paths.dataset_path,
            "--output_render_path",
            f"{paths.output_folder}/evaluation",
            "--save_renders",
            "--sh_degree",
            str(SH_DEGREE),
            "--resolution",
            str(RESOLUTION),
            "--frame_ids",
            frame_ids_csv,
        ],
    )
    run_cmd(cmd, cwd=VIDEOGS_ROOT, dry_run=dry_run)


def run_dracogs(sequence: str, frame_ids_csv: str, dry_run: bool) -> None:
    paths = get_paths(sequence, "dracogs")
    log_step(f"DracoGS | {sequence} | frames: {frame_ids_csv} | {timestamp()}")

    cmd = conda_python_cmd(
        BASELINE_ENVS["dracogs"],
        VIDEOGS_ROOT / "scripts" / "dracogs_baseline" / "compress_decompress_pipeline.py",
        [
            "--ply_path",
            paths.gt_model_path,
            "--output_folder",
            paths.output_folder,
            "--output_ply_folder",
            f"{paths.output_folder}/decompressed_ply",
            "--frame_ids",
            frame_ids_csv,
            "--sh_degree",
            str(SH_DEGREE),
            "--scene_name",
            sequence,
            "--eg",
            str(DRACOGS_EG),
            "--eo",
            str(DRACOGS_EO),
            "--et",
            str(DRACOGS_ET),
            "--es",
            str(DRACOGS_ES),
            "--cl",
            str(DRACOGS_CL),
        ],
    )
    run_cmd(cmd, cwd=VIDEOGS_ROOT, dry_run=dry_run)
    run_evaluation(paths, frame_ids_csv, dry_run)


def run_mesongs(sequence: str, frame_ids_csv: str, dry_run: bool) -> None:
    paths = get_paths(sequence, "mesongs")
    log_step(f"MesonGS | {sequence} | frames: {frame_ids_csv} | {timestamp()}")

    cmd = conda_python_cmd(
        BASELINE_ENVS["mesongs"],
        VIDEOGS_ROOT / "scripts" / "mesongs_baseline" / "compress_decompress_pipeline.py",
        [
            "--ply_path",
            paths.gt_model_path,
            "--dataset_path",
            paths.dataset_path,
            "--output_folder",
            paths.output_folder,
            "--output_ply_folder",
            f"{paths.output_folder}/decompressed_ply",
            "--frame_ids",
            frame_ids_csv,
            "--sh_degree",
            str(SH_DEGREE),
            "--resolution",
            str(RESOLUTION),
            "--scene_name",
            sequence,
        ],
    )
    run_cmd(cmd, cwd=MESONGS_ROOT, dry_run=dry_run)
    run_evaluation(paths, frame_ids_csv, dry_run)


def run_videogs(sequence: str, frame_ids_csv: str, dry_run: bool) -> None:
    paths = get_paths(sequence, "videogs")
    log_step(f"VideoGS | {sequence} | frames: {frame_ids_csv} | {timestamp()}")

    cmd = conda_python_cmd(
        BASELINE_ENVS["videogs"],
        VIDEOGS_ROOT / "scripts" / "videogs_baseline" / "compress_decompress_pipeline.py",
        [
            "--ply_path",
            paths.gt_model_path,
            "--output_folder",
            paths.output_folder,
            "--output_ply_folder",
            f"{paths.output_folder}/decompressed_ply",
            "--frame_ids",
            frame_ids_csv,
            "--group_size",
            str(VIDEOGS_GROUP_SIZE),
            "--sh_degree",
            str(SH_DEGREE),
            "--qp",
            str(VIDEOGS_QP),
        ],
    )
    run_cmd(cmd, cwd=VIDEOGS_ROOT, dry_run=dry_run)
    run_evaluation(paths, frame_ids_csv, dry_run)


Runner = Callable[[str, str, bool], None]

BASELINE_RUNNERS: dict[str, Runner] = {
    "dracogs": run_dracogs,
    "mesongs": run_mesongs,
    "videogs": run_videogs,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run selected compression experiments.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args()


def format_duration(total_seconds: int) -> str:
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours}h {minutes}m {seconds}s"


def main() -> None:
    args = parse_args()
    if args.dry_run:
        print("[DRY RUN] Commands will be printed but not executed.")

    unknown = [name for name in BASELINES if name not in BASELINE_RUNNERS]
    if unknown:
        raise ValueError(f"Unknown baseline(s): {unknown}")

    ensure_required_envs(BASELINES)
    if "videogs" in BASELINES:
        ensure_videogs_frame_spacing(EXPERIMENTS, VIDEOGS_GROUP_SIZE)

    script_start = int(time.time())
    failed_runs: list[tuple[str, str]] = []

    log_header("Selected Experiments Runner")
    print(f"  Started:    {timestamp()}")
    print(f"  Dataset:    {DATASET_NAME}")
    print(f"  Baselines:  {' '.join(BASELINES)}")
    print(f"  Sequences:  {len(EXPERIMENTS)}")
    print(f"  Resolution: {RESOLUTION}")
    print(f"  SH degree:  {SH_DEGREE}")
    print(f"  CUDA:       {CUDA_DEVICE}")
    print(f"  Data path:  {DATA_PATH}")
    print("=" * 70)

    for sequence, frame_ids in EXPERIMENTS.items():
        frame_ids_csv = ",".join(str(fid) for fid in frame_ids)
        log_header(f"Sequence: {sequence} | Frames: {frame_ids_csv}")

        for baseline in BASELINES:
            log_header(f"{baseline.upper()} | {sequence}")
            baseline_start = int(time.time())

            try:
                BASELINE_RUNNERS[baseline](sequence, frame_ids_csv, args.dry_run)
            except subprocess.CalledProcessError as exc:
                print(f"WARNING: {baseline} failed for {sequence} (exit {exc.returncode})")
                failed_runs.append((baseline, sequence))
            except Exception as exc:
                print(f"WARNING: {baseline} failed for {sequence} ({exc})")
                failed_runs.append((baseline, sequence))

            baseline_end = int(time.time())
            print(f"  {baseline.upper()} | {sequence} completed in {baseline_end - baseline_start}s")

    script_end = int(time.time())
    elapsed = script_end - script_start

    log_header("All experiments complete!")
    print(f"  Finished:      {timestamp()}")
    print(f"  Total time:    {format_duration(elapsed)}")

    if failed_runs:
        print("")
        print(f"  FAILED RUNS ({len(failed_runs)}):")
        for baseline, sequence in failed_runs:
            print(f"    - {baseline} | {sequence}")
    else:
        print("  All runs succeeded.")

    print("")
    print("  Output locations:")
    for sequence in EXPERIMENTS:
        for baseline in BASELINES:
            output_folder = get_output_folder(baseline, sequence)
            print(f"    {baseline} | {sequence}: {output_folder}")
    print("=" * 70)


if __name__ == "__main__":
    main()
