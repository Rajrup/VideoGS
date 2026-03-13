#!/usr/bin/env python3
"""Run DracoGS, MesonGS, VideoGS, and GPCC baselines across HiFi4G and N3DV.

Supports both datasets via DatasetConfig. Per-dataset differences (frame-end
convention, path layout, evaluation interface, conda environments) are
encapsulated in the config objects so runner functions stay dataset-agnostic.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from queue import Queue
from typing import Any, Callable


# ===========================================================================
# DatasetConfig — encapsulates all per-dataset differences
# ===========================================================================

@dataclass(frozen=True)
class DatasetConfig:
    name: str
    dataset_name: str
    data_path: str
    project_name: str
    sh_degree: int
    resolution: int
    evaluation_env: str
    baseline_envs: dict[str, str]
    sequences: list[str]
    baseline_frame_ids: dict[str, list[int]]
    frame_end_exclusive: bool
    mesongs_script_name: str
    mesongs_has_resolution_arg: bool
    per_frame_baselines: tuple[str, ...] = ()


# ===========================================================================
# Workspace layout
# ===========================================================================

SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parent
WORKSPACE_ROOT = SCRIPTS_DIR.parent.parent

GPCC_TMC3_PATH = "/home/haodongw/workspace/mpeg-pcc-tmc13/build/tmc3/tmc3"
DEFAULT_GPUS: list[int] = [0]
DEFAULT_WORKERS_PER_GPU: int = 1
SKIP_EXISTING = True

# ===========================================================================
# Dataset configs
# ===========================================================================

HIFI4G = DatasetConfig(
    name="HiFi4G",
    dataset_name="HiFi4G_Dataset",
    data_path="/synology/rajrup/VideoGS",
    project_name="VideoGS",
    sh_degree=3,
    resolution=2,
    evaluation_env="videogs",
    baseline_envs={
        "dracogs": "videogs",
        "mesongs": "mesongs",
        "videogs": "videogs",
        "gpcc": "videogs",
    },
    sequences=[
        "4K_Actor1_Greeting",
        "4K_Actor2_Dancing",
        "4K_Actor3_Violin",
        "4K_Actor4_Dancing",
        "4K_Actor5_Oil-paper_Umbrella",
        "4K_Actor6_Changing_Clothes",
        "4K_Actor7_Nunchaku",
    ],
    baseline_frame_ids={
        "dracogs": list(range(1, 201, 10)),
        "mesongs": list(range(1, 201, 10)),
        "videogs": list(range(1, 201, 20)),
        "gpcc": list(range(1, 201, 10)),
    },
    frame_end_exclusive=True,
    mesongs_script_name="compress_decompress_pipeline.py",
    mesongs_has_resolution_arg=True,
)

N3DV = DatasetConfig(
    name="N3DV",
    dataset_name="Neural_3D_Video",
    data_path="/synology/rajrup/Queen",
    project_name="queen",
    sh_degree=2,
    resolution=2,
    evaluation_env="queen",
    baseline_envs={
        "dracogs": "queen",
        "mesongs": "mesongs",
        "videogs": "queen",
        "gpcc": "videogs",
    },
    sequences=[
        "cook_spinach",
        "coffee_martini",
        "cut_roasted_beef",
        "flame_salmon_1",
        "flame_steak",
        "sear_steak",
    ],
    baseline_frame_ids={
        "dracogs": list(range(1, 201, 10)),
        "mesongs": list(range(1, 201, 10)),
        "videogs": list(range(1, 201, 20)),
        "gpcc": list(range(1, 201, 10)),
    },
    frame_end_exclusive=False,
    mesongs_script_name="compression_decompress_pipeline.py",
    mesongs_has_resolution_arg=False,
    per_frame_baselines=("dracogs", "mesongs"),
)

ALL_DATASETS: dict[str, DatasetConfig] = {"HiFi4G": HIFI4G, "N3DV": N3DV}

ACTIVE_DATASETS: list[str] = ["HiFi4G"]

# ===========================================================================
# Active baselines (uncomment to enable)
# ===========================================================================

ACTIVE_BASELINES: list[str] = [
    # "dracogs",
    # "mesongs",
    "videogs",
    # "gpcc",
]

# ===========================================================================
# Per-baseline parameter control
# True  → use the known-good default parameters.
# False → use the manual parameters you configure below.
# ===========================================================================

USE_DEFAULT_PARAMS: dict[str, bool] = {
    "dracogs": True,
    "mesongs": True,
    "videogs": True,
    "gpcc": True,
}

DRACOGS_DEFAULTS: dict[str, Any] = {"eg": 16, "eo": 16, "et": 16, "es": 16, "cl": 10}
MESONGS_DEFAULTS: dict[str, Any] = {"depth": 12, "num_bits": 8, "n_block": 57, "codebook_size": 2048}
VIDEOGS_DEFAULTS: dict[str, Any] = {"qps": [25], "group_size": 20}

DRACOGS_MANUAL: dict[str, Any] = {"eg": 16, "eo": 16, "et": 16, "es": 16, "cl": 10}
MESONGS_MANUAL: dict[str, Any] = {"depth": 12, "num_bits": 8, "n_block": 57, "codebook_size": 2048}
VIDEOGS_MANUAL: dict[str, Any] = {"qps": [25], "group_size": 20}

_dracogs = DRACOGS_DEFAULTS if USE_DEFAULT_PARAMS["dracogs"] else DRACOGS_MANUAL
DRACOGS_EG = _dracogs["eg"]
DRACOGS_EO = _dracogs["eo"]
DRACOGS_ET = _dracogs["et"]
DRACOGS_ES = _dracogs["es"]
DRACOGS_CL = _dracogs["cl"]

_mesongs = MESONGS_DEFAULTS if USE_DEFAULT_PARAMS["mesongs"] else MESONGS_MANUAL
MESONGS_DEPTH = _mesongs["depth"]
MESONGS_NUM_BITS = _mesongs["num_bits"]
MESONGS_N_BLOCK = _mesongs["n_block"]
MESONGS_CODEBOOK_SIZE = _mesongs["codebook_size"]

_videogs = VIDEOGS_DEFAULTS if USE_DEFAULT_PARAMS["videogs"] else VIDEOGS_MANUAL
VIDEOGS_QPS: list[int] = _videogs["qps"]
VIDEOGS_GROUP_SIZE: int = _videogs["group_size"]

GPCC_DEFAULTS_FILE = SCRIPTS_DIR / "gpcc_defaults.json"
GPCC_MANUAL: dict[str, Any] = {"voxel_depth": 12, "qp_rest": 40, "qp_dc": 4, "qp_opacity": 16}

_gpcc_defaults_cache: dict[str, dict[str, Any]] | None = None


def _load_gpcc_defaults() -> dict[str, dict[str, Any]]:
    global _gpcc_defaults_cache
    if _gpcc_defaults_cache is not None:
        return _gpcc_defaults_cache
    if not GPCC_DEFAULTS_FILE.is_file():
        raise FileNotFoundError(
            f"GPCC defaults file not found: {GPCC_DEFAULTS_FILE}\n"
            "Run select_gpcc_defaults.py first, or set USE_DEFAULT_PARAMS['gpcc'] = False."
        )
    with open(GPCC_DEFAULTS_FILE, encoding="utf-8") as f:
        loaded: dict[str, dict[str, Any]] = json.load(f)
    _gpcc_defaults_cache = loaded
    return loaded


def get_gpcc_params(sequence: str) -> dict[str, Any]:
    if not USE_DEFAULT_PARAMS.get("gpcc", False):
        return GPCC_MANUAL
    defaults = _load_gpcc_defaults()
    if sequence not in defaults:
        raise KeyError(
            f"No GPCC default params for sequence '{sequence}' in {GPCC_DEFAULTS_FILE}. "
            "Re-run select_gpcc_defaults.py or set USE_DEFAULT_PARAMS['gpcc'] = False."
        )
    return defaults[sequence]


# ===========================================================================
# Helpers
# ===========================================================================

@dataclass(frozen=True)
class ExperimentPaths:
    dataset_path: str
    gt_model_path: str
    output_folder: str


_LOG_LOCK = threading.Lock()


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_header(message: str) -> None:
    with _LOG_LOCK:
        print(f"\n{'=' * 70}\n  {message}\n{'=' * 70}")


def log_step(message: str) -> None:
    with _LOG_LOCK:
        print(f"--- {message}")


def run_cmd(cmd: list[str], cwd: Path, dry_run: bool, cuda_device: str = "0") -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_device
    if dry_run:
        log_step(f"[DRY RUN] cwd={cwd} | CUDA_VISIBLE_DEVICES={cuda_device} | {shlex.join(cmd)}")
        return
    subprocess.run(cmd, cwd=str(cwd), check=True, env=env)


def conda_python_cmd(env_name: str, script_path: Path, args: list[str]) -> list[str]:
    return ["conda", "run", "-n", env_name, "python", str(script_path), *args]


def _output_complete(output_folder: str) -> bool:
    return Path(output_folder, "evaluation").is_dir()


def _cleanup_partial(output_folder: str) -> None:
    p = Path(output_folder)
    if p.exists():
        log_step(f"Cleaning up partial output: {output_folder}")
        shutil.rmtree(p)


def get_available_conda_envs() -> set[str]:
    proc = subprocess.run(
        ["conda", "env", "list", "--json"],
        check=True, capture_output=True, text=True,
    )
    return {Path(p).name for p in json.loads(proc.stdout).get("envs", [])}


def ensure_required_envs(cfg: DatasetConfig, baselines: list[str]) -> None:
    required = {cfg.evaluation_env}
    required.update(cfg.baseline_envs[b] for b in baselines if b in cfg.baseline_envs)
    available = get_available_conda_envs()
    missing = sorted(e for e in required if e not in available)
    if missing:
        raise RuntimeError(
            f"Missing conda environment(s) for {cfg.name}: {', '.join(missing)}"
        )


# ===========================================================================
# Dataset-aware path helpers
# ===========================================================================

def _project_root(cfg: DatasetConfig) -> Path:
    return WORKSPACE_ROOT / cfg.project_name


def _mesongs_root(cfg: DatasetConfig) -> Path:
    return _project_root(cfg) / "MesonGS"


def _model_root(cfg: DatasetConfig, sequence: str) -> Path:
    if cfg.name == "N3DV":
        return Path(cfg.data_path) / "pretrained_output" / cfg.dataset_name / f"queen_compressed_{sequence}"
    return Path(cfg.data_path) / "train_output" / cfg.dataset_name / sequence


def _gt_model_path(cfg: DatasetConfig, sequence: str) -> str:
    root = _model_root(cfg, sequence)
    if cfg.name == "N3DV":
        return str(root)
    return str(root / "checkpoint")


def _dataset_path(cfg: DatasetConfig, sequence: str) -> str:
    if cfg.name == "N3DV":
        return str(Path(cfg.data_path) / cfg.dataset_name / sequence)
    return str(Path(cfg.data_path) / f"{cfg.dataset_name}_processed" / sequence)


# ===========================================================================
# Frame handling (respects frame_end_exclusive)
# ===========================================================================

def selected_to_span(cfg: DatasetConfig, frame_ids: list[int]) -> tuple[int, int, int]:
    if not frame_ids:
        raise ValueError("Frame list must not be empty")
    sorted_ids = sorted(set(int(v) for v in frame_ids))
    if cfg.frame_end_exclusive:
        return sorted_ids[0], sorted_ids[-1] + 1, 1
    return sorted_ids[0], sorted_ids[-1], 1


def _single_frame_span(cfg: DatasetConfig, frame_id: int) -> tuple[int, int, int]:
    if cfg.frame_end_exclusive:
        return frame_id, frame_id + 1, 1
    return frame_id, frame_id, 1


def frame_span_tag(cfg: DatasetConfig, frame_start: int, frame_end: int, interval: int) -> str:
    if cfg.frame_end_exclusive:
        return f"frames_{frame_start}_{frame_end - 1}_int_{interval}"
    return f"frames_{frame_start}_{frame_end}_int_{interval}"


@lru_cache(maxsize=None)
def get_sequence_max_frame(cfg_name: str, data_path: str, dataset_name: str, sequence: str) -> int:
    if cfg_name == "N3DV":
        frames_root = (
            Path(data_path) / "pretrained_output" / dataset_name
            / f"queen_compressed_{sequence}" / "frames"
        )
        if not frames_root.is_dir():
            raise FileNotFoundError(f"Frames root not found: {frames_root}")
        frame_ids = sorted(
            int(e.name) for e in frames_root.iterdir()
            if e.is_dir() and e.name.isdigit()
            and ((e / "point_cloud.ply").is_file() or (e / "point_cloud").is_dir())
        )
    else:
        checkpoint_root = Path(data_path) / "train_output" / dataset_name / sequence / "checkpoint"
        if not checkpoint_root.is_dir():
            raise FileNotFoundError(f"Checkpoint root not found: {checkpoint_root}")
        frame_ids = sorted(
            int(e.name) for e in checkpoint_root.iterdir()
            if e.is_dir() and e.name.isdigit() and (e / "point_cloud").is_dir()
        )
    if not frame_ids:
        raise FileNotFoundError(f"No frame folders found for {cfg_name}/{sequence}")
    return frame_ids[-1]


def _max_frame(cfg: DatasetConfig, sequence: str) -> int:
    return get_sequence_max_frame(cfg.name, cfg.data_path, cfg.dataset_name, sequence)


def resolve_videogs_span(cfg: DatasetConfig, sequence: str, anchor_frame: int) -> tuple[int, int]:
    max_frame = _max_frame(cfg, sequence)
    if anchor_frame > max_frame:
        raise ValueError(
            f"VideoGS anchor frame {anchor_frame} exceeds last available frame {max_frame} "
            f"for {cfg.name}/{sequence}"
        )
    if cfg.frame_end_exclusive:
        gop_end = min(anchor_frame + VIDEOGS_GROUP_SIZE, max_frame + 1)
        if gop_end <= anchor_frame:
            raise ValueError(f"Empty VideoGS GOP for {cfg.name}/{sequence}")
    else:
        gop_end = min(anchor_frame + VIDEOGS_GROUP_SIZE - 1, max_frame)
        if gop_end < anchor_frame:
            raise ValueError(f"Empty VideoGS GOP for {cfg.name}/{sequence}")
    return anchor_frame, gop_end


# ===========================================================================
# Output folder resolution
# ===========================================================================

def get_output_folder(
    cfg: DatasetConfig,
    baseline: str,
    sequence: str,
    frame_start: int,
    frame_end: int,
    interval: int,
    videogs_qp: int | None = None,
    gpcc_frame_id: int | None = None,
) -> str:
    model_root = _model_root(cfg, sequence)
    run_tag = frame_span_tag(cfg, frame_start, frame_end, interval)
    frame_tag = f"frame{frame_start}"

    is_single = (
        (cfg.frame_end_exclusive and frame_end == frame_start + 1 and interval == 1)
        or (not cfg.frame_end_exclusive and frame_start == frame_end and interval == 1)
    )
    use_per_frame = baseline in cfg.per_frame_baselines and is_single

    if baseline == "dracogs":
        tag = (
            f"eg_{DRACOGS_EG}_eo_{DRACOGS_EO}_"
            f"et_{DRACOGS_ET}_es_{DRACOGS_ES}_cl_{DRACOGS_CL}"
        )
        leaf = frame_tag if use_per_frame else run_tag
        return str(model_root / "compression" / "dracogs" / tag / leaf)
    if baseline == "mesongs":
        tag = (
            f"d{MESONGS_DEPTH}_nb{MESONGS_NUM_BITS}_"
            f"nblk{MESONGS_N_BLOCK}_cb{MESONGS_CODEBOOK_SIZE}"
        )
        leaf = frame_tag if use_per_frame else run_tag
        return str(model_root / "compression" / "mesongs" / tag / leaf)
    if baseline == "videogs":
        if videogs_qp is None:
            raise ValueError("videogs_qp must be provided for videogs output folder")
        return str(model_root / "compression" / "videogs" / f"qp_{videogs_qp}" / run_tag)
    if baseline == "gpcc":
        if gpcc_frame_id is None:
            raise ValueError("gpcc_frame_id must be provided for gpcc output folder")
        p = get_gpcc_params(sequence)
        params_tag = f"J{p['voxel_depth']}_rest{p['qp_rest']}_dc{p['qp_dc']}_op{p['qp_opacity']}"
        return str(model_root / "compression" / "gpcc" / params_tag / f"frame{gpcc_frame_id}")

    raise ValueError(f"Unknown baseline: {baseline}")


def _get_paths(
    cfg: DatasetConfig,
    baseline: str,
    sequence: str,
    frame_start: int,
    frame_end: int,
    interval: int,
    videogs_qp: int | None = None,
) -> ExperimentPaths:
    return ExperimentPaths(
        dataset_path=_dataset_path(cfg, sequence),
        gt_model_path=_gt_model_path(cfg, sequence),
        output_folder=get_output_folder(
            cfg, baseline, sequence, frame_start, frame_end, interval, videogs_qp=videogs_qp,
        ),
    )


# ===========================================================================
# Evaluation (dataset-aware)
# ===========================================================================

def run_evaluation(
    cfg: DatasetConfig,
    paths: ExperimentPaths,
    frame_start: int,
    frame_end: int,
    interval: int,
    dry_run: bool,
    cuda_device: str = "0",
) -> None:
    project_root = _project_root(cfg)
    if cfg.name == "N3DV":
        eval_args = [
            "--config", "configs/dynerf.yaml",
            "-s", paths.dataset_path,
            "-m", paths.gt_model_path,
            "--decompressed_ply_path", f"{paths.output_folder}/decompressed_ply",
            "--output_render_path", f"{paths.output_folder}/evaluation",
            "--frame_start", str(frame_start),
            "--frame_end", str(frame_end),
            "--interval", str(interval),
        ]
    else:
        eval_args = [
            "--gt_ply_path", paths.gt_model_path,
            "--decompressed_ply_path", f"{paths.output_folder}/decompressed_ply",
            "--dataset_path", paths.dataset_path,
            "--output_render_path", f"{paths.output_folder}/evaluation",
            "--sh_degree", str(cfg.sh_degree),
            "--resolution", str(cfg.resolution),
            "--frame_start", str(frame_start),
            "--frame_end", str(frame_end),
            "--interval", str(interval),
        ]
    cmd = conda_python_cmd(
        cfg.evaluation_env,
        project_root / "scripts" / "evaluate_decompress.py",
        eval_args,
    )
    run_cmd(cmd, cwd=project_root, dry_run=dry_run, cuda_device=cuda_device)


# ===========================================================================
# Baseline runners
# ===========================================================================

def _dracogs_frame_spans(cfg: DatasetConfig, selected_frames: list[int]) -> list[tuple[int, int, int]]:
    if "dracogs" in cfg.per_frame_baselines:
        return [_single_frame_span(cfg, fid) for fid in sorted(set(int(v) for v in selected_frames))]
    return [selected_to_span(cfg, selected_frames)]


def run_dracogs(cfg: DatasetConfig, sequence: str, selected_frames: list[int], dry_run: bool, skip_existing: bool, cuda_device: str = "0") -> None:
    project_root = _project_root(cfg)

    for fs, fe, iv in _dracogs_frame_spans(cfg, selected_frames):
        paths = _get_paths(cfg, "dracogs", sequence, fs, fe, iv)
        label = f"frame: {fs}" if fs == fe or fe == fs + 1 else f"frames: {fs}-{fe}:{iv}"
        if skip_existing and _output_complete(paths.output_folder):
            log_step(f"SKIP (exists) DracoGS | {cfg.name} | {sequence} | {label}")
            continue
        log_step(f"DracoGS | {cfg.name} | {sequence} | {label} | {timestamp()}")

        cmd = conda_python_cmd(
            cfg.baseline_envs["dracogs"],
            project_root / "scripts" / "dracogs_baseline" / "compress_decompress_pipeline.py",
            [
                "--ply_path", paths.gt_model_path,
                "--output_folder", paths.output_folder,
                "--output_ply_folder", f"{paths.output_folder}/decompressed_ply",
                "--frame_start", str(fs),
                "--frame_end", str(fe),
                "--interval", str(iv),
                "--sh_degree", str(cfg.sh_degree),
                "--scene_name", sequence,
                "--eg", str(DRACOGS_EG),
                "--eo", str(DRACOGS_EO),
                "--et", str(DRACOGS_ET),
                "--es", str(DRACOGS_ES),
                "--cl", str(DRACOGS_CL),
            ],
        )
        try:
            run_cmd(cmd, cwd=project_root, dry_run=dry_run, cuda_device=cuda_device)
            run_evaluation(cfg, paths, fs, fe, iv, dry_run, cuda_device=cuda_device)
        except subprocess.CalledProcessError:
            _cleanup_partial(paths.output_folder)
            raise


def _mesongs_frame_spans(cfg: DatasetConfig, selected_frames: list[int]) -> list[tuple[int, int, int]]:
    if "mesongs" in cfg.per_frame_baselines:
        return [_single_frame_span(cfg, fid) for fid in sorted(set(int(v) for v in selected_frames))]
    return [selected_to_span(cfg, selected_frames)]


def run_mesongs(cfg: DatasetConfig, sequence: str, selected_frames: list[int], dry_run: bool, skip_existing: bool, cuda_device: str = "0") -> None:
    project_root = _project_root(cfg)
    mesongs_root = _mesongs_root(cfg)

    for fs, fe, iv in _mesongs_frame_spans(cfg, selected_frames):
        paths = _get_paths(cfg, "mesongs", sequence, fs, fe, iv)
        label = f"frame: {fs}" if fs == fe or fe == fs + 1 else f"frames: {fs}-{fe}:{iv}"
        if skip_existing and _output_complete(paths.output_folder):
            log_step(f"SKIP (exists) MesonGS | {cfg.name} | {sequence} | {label}")
            continue
        log_step(f"MesonGS | {cfg.name} | {sequence} | {label} | {timestamp()}")

        mesongs_args = [
            "--ply_path", paths.gt_model_path,
            "--dataset_path", paths.dataset_path,
            "--output_folder", paths.output_folder,
            "--output_ply_folder", f"{paths.output_folder}/decompressed_ply",
            "--frame_start", str(fs),
            "--frame_end", str(fe),
            "--interval", str(iv),
            "--sh_degree", str(cfg.sh_degree),
            "--scene_name", sequence,
            "--depth", str(MESONGS_DEPTH),
            "--num_bits", str(MESONGS_NUM_BITS),
            "--n_block", str(MESONGS_N_BLOCK),
            "--codebook_size", str(MESONGS_CODEBOOK_SIZE),
        ]
        if cfg.mesongs_has_resolution_arg:
            mesongs_args.extend(["--resolution", str(cfg.resolution)])

        cmd = conda_python_cmd(
            cfg.baseline_envs["mesongs"],
            project_root / "scripts" / "mesongs_baseline" / cfg.mesongs_script_name,
            mesongs_args,
        )
        try:
            run_cmd(cmd, cwd=mesongs_root, dry_run=dry_run, cuda_device=cuda_device)
            run_evaluation(cfg, paths, fs, fe, iv, dry_run, cuda_device=cuda_device)
        except subprocess.CalledProcessError:
            _cleanup_partial(paths.output_folder)
            raise


def run_gpcc(cfg: DatasetConfig, sequence: str, selected_frames: list[int], dry_run: bool, skip_existing: bool, cuda_device: str = "0") -> None:
    gt_model_path = _gt_model_path(cfg, sequence)
    dataset_path = _dataset_path(cfg, sequence)
    project_root = _project_root(cfg)
    p = get_gpcc_params(sequence)

    for frame_id in sorted(set(int(v) for v in selected_frames)):
        fs, fe, iv = _single_frame_span(cfg, frame_id)
        output_folder = get_output_folder(cfg, "gpcc", sequence, fs, fe, iv, gpcc_frame_id=frame_id)
        if skip_existing and _output_complete(output_folder):
            log_step(f"SKIP (exists) GPCC | {cfg.name} | {sequence} | frame: {frame_id}")
            continue
        log_step(
            f"GPCC | {cfg.name} | {sequence} | frame: {frame_id} | "
            f"J={p['voxel_depth']} rest={p['qp_rest']} dc={p['qp_dc']} op={p['qp_opacity']} | {timestamp()}"
        )

        cmd = conda_python_cmd(
            cfg.baseline_envs["gpcc"],
            SCRIPTS_DIR / "gpcc_baseline" / "compress_decompress_pipeline.py",
            [
                "--input_dir", gt_model_path,
                "--output_dir", output_folder,
                "--output_ply_dir", f"{output_folder}/decompressed_ply",
                "--tmc3_path", GPCC_TMC3_PATH,
                "--voxel_depth", str(p["voxel_depth"]),
                "--qp_rest", str(p["qp_rest"]),
                "--qp_dc", str(p["qp_dc"]),
                "--qp_opacity", str(p["qp_opacity"]),
                "--frame_start", str(frame_id),
                "--num_frames", "1",
            ],
        )
        try:
            run_cmd(cmd, cwd=project_root, dry_run=dry_run, cuda_device=cuda_device)
            eval_paths = ExperimentPaths(dataset_path, gt_model_path, output_folder)
            run_evaluation(cfg, eval_paths, fs, fe, iv, dry_run, cuda_device=cuda_device)
        except subprocess.CalledProcessError:
            _cleanup_partial(output_folder)
            raise


def run_videogs(cfg: DatasetConfig, sequence: str, selected_frames: list[int], dry_run: bool, skip_existing: bool, cuda_device: str = "0") -> None:
    project_root = _project_root(cfg)

    for anchor_frame in sorted(set(int(v) for v in selected_frames)):
        gop_start, gop_end = resolve_videogs_span(cfg, sequence, anchor_frame)
        for qp in VIDEOGS_QPS:
            paths = _get_paths(cfg, "videogs", sequence, gop_start, gop_end, 1, videogs_qp=qp)
            if skip_existing and _output_complete(paths.output_folder):
                log_step(
                    f"SKIP (exists) VideoGS | {cfg.name} | {sequence} | QP: {qp} | "
                    f"anchor: {anchor_frame} | frames: {gop_start}-{gop_end}:1"
                )
                continue
            log_step(
                f"VideoGS | {cfg.name} | {sequence} | QP: {qp} | "
                f"anchor: {anchor_frame} | frames: {gop_start}-{gop_end}:1 | {timestamp()}"
            )

            cmd = conda_python_cmd(
                cfg.baseline_envs["videogs"],
                project_root / "scripts" / "videogs_baseline" / "compress_decompress_pipeline.py",
                [
                    "--ply_path", paths.gt_model_path,
                    "--output_folder", paths.output_folder,
                    "--output_ply_folder", f"{paths.output_folder}/decompressed_ply",
                    "--frame_start", str(gop_start),
                    "--frame_end", str(gop_end),
                    "--interval", "1",
                    "--group_size", str(VIDEOGS_GROUP_SIZE),
                    "--sh_degree", str(cfg.sh_degree),
                    "--qp", str(qp),
                ],
            )
            try:
                run_cmd(cmd, cwd=project_root, dry_run=dry_run, cuda_device=cuda_device)
                run_evaluation(cfg, paths, gop_start, gop_end, 1, dry_run, cuda_device=cuda_device)
            except subprocess.CalledProcessError:
                _cleanup_partial(paths.output_folder)
                raise


# ===========================================================================
# Expected output folders (for summary)
# ===========================================================================

def get_expected_output_folders(
    cfg: DatasetConfig,
    baseline: str,
    sequence: str,
    selected_frames: list[int],
) -> list[str]:
    if baseline == "videogs":
        folders: list[str] = []
        for anchor in sorted(set(int(v) for v in selected_frames)):
            gs, ge = resolve_videogs_span(cfg, sequence, anchor)
            folders.extend(
                get_output_folder(cfg, baseline, sequence, gs, ge, 1, videogs_qp=qp)
                for qp in VIDEOGS_QPS
            )
        return folders

    if baseline == "gpcc":
        return [
            get_output_folder(
                cfg, "gpcc", sequence,
                *_single_frame_span(cfg, fid),
                gpcc_frame_id=fid,
            )
            for fid in sorted(set(int(v) for v in selected_frames))
        ]

    if baseline in cfg.per_frame_baselines:
        return [
            get_output_folder(cfg, baseline, sequence, *_single_frame_span(cfg, fid))
            for fid in sorted(set(int(v) for v in selected_frames))
        ]

    fs, fe, iv = selected_to_span(cfg, selected_frames)
    return [get_output_folder(cfg, baseline, sequence, fs, fe, iv)]


# ===========================================================================
# Runner registry
# ===========================================================================

BASELINE_RUNNERS: dict[
    str,
    Callable[..., None],
] = {
    "dracogs": run_dracogs,
    "mesongs": run_mesongs,
    "videogs": run_videogs,
    "gpcc": run_gpcc,
}


# ===========================================================================
# Parallel execution helpers
# ===========================================================================

def _build_job_list(
    selected_datasets: list[str],
    selected_baselines: list[str],
) -> list[tuple[DatasetConfig, str, str, list[int]]]:
    jobs: list[tuple[DatasetConfig, str, str, list[int]]] = []
    for ds_name in selected_datasets:
        dcfg = ALL_DATASETS[ds_name]
        ds_baselines = [b for b in selected_baselines if b in dcfg.baseline_envs]
        for sequence in dcfg.sequences:
            for baseline in ds_baselines:
                frame_ids = dcfg.baseline_frame_ids.get(baseline, [])
                if frame_ids:
                    jobs.append((dcfg, baseline, sequence, frame_ids))
    return jobs


def _run_job(
    job: tuple[DatasetConfig, str, str, list[int]],
    gpu_queue: "Queue[int]",
    dry_run: bool,
    skip_existing: bool,
) -> tuple[str, str, str] | None:
    cfg, baseline, sequence, frame_ids = job
    gpu_id = gpu_queue.get()
    try:
        log_header(
            f"{baseline.upper()} | {cfg.name} | {sequence} | "
            f"Frames: {','.join(str(v) for v in frame_ids[:5])}... | GPU {gpu_id}"
        )
        runner = BASELINE_RUNNERS[baseline]
        step_start = time.time()
        runner(cfg, sequence, frame_ids, dry_run, skip_existing, cuda_device=str(gpu_id))
        elapsed = int(time.time() - step_start)
        log_step(
            f"{baseline.upper()} | {cfg.name} | {sequence} completed in {elapsed}s (GPU {gpu_id})"
        )
        return None
    except subprocess.CalledProcessError as exc:
        log_step(
            f"WARNING: {baseline} failed for {cfg.name}/{sequence} on GPU {gpu_id} "
            f"(exit {exc.returncode})"
        )
        return (cfg.name, baseline, sequence)
    except Exception as exc:
        log_step(
            f"ERROR: {baseline} failed for {cfg.name}/{sequence} on GPU {gpu_id}: {exc}"
        )
        return (cfg.name, baseline, sequence)
    finally:
        gpu_queue.put(gpu_id)


# ===========================================================================
# CLI + main
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline experiments across HiFi4G and N3DV datasets",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=SKIP_EXISTING,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(ALL_DATASETS.keys()),
        default=ACTIVE_DATASETS,
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        choices=list(BASELINE_RUNNERS.keys()),
        default=ACTIVE_BASELINES,
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        default=DEFAULT_GPUS,
        help="GPU IDs to use (default: %(default)s)",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=DEFAULT_WORKERS_PER_GPU,
        help="Concurrent processes per GPU (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_datasets: list[str] = args.datasets
    selected_baselines: list[str] = args.baselines
    gpus: list[int] = args.gpus
    workers_per_gpu: int = args.workers_per_gpu
    total_workers = len(gpus) * workers_per_gpu

    unknown_bl = [b for b in selected_baselines if b not in BASELINE_RUNNERS]
    if unknown_bl:
        raise ValueError(f"Unknown baseline(s): {unknown_bl}")
    unknown_ds = [d for d in selected_datasets if d not in ALL_DATASETS]
    if unknown_ds:
        raise ValueError(f"Unknown dataset(s): {unknown_ds}")

    run_start = time.time()
    log_header("Baseline Experiments Runner")
    print(f"  Started:      {timestamp()}")
    print(f"  Datasets:     {', '.join(selected_datasets)}")
    print(f"  Baselines:    {', '.join(selected_baselines)}")
    print(f"  GPUs:         {', '.join(str(g) for g in gpus)}")
    print(f"  Workers/GPU:  {workers_per_gpu}  (total workers: {total_workers})")
    if args.dry_run:
        print("  Mode:         DRY RUN")
    if args.skip_existing:
        print("  Skip:         existing outputs")

    for ds_name in selected_datasets:
        dcfg = ALL_DATASETS[ds_name]
        print(f"  [{dcfg.name}] sequences={len(dcfg.sequences)}, sh={dcfg.sh_degree}, res={dcfg.resolution}")
    print("=" * 70)

    for ds_name in selected_datasets:
        dcfg = ALL_DATASETS[ds_name]
        ds_baselines = [b for b in selected_baselines if b in dcfg.baseline_envs]
        if ds_baselines:
            ensure_required_envs(dcfg, ds_baselines)

    jobs = _build_job_list(selected_datasets, selected_baselines)
    if not jobs:
        print("No jobs to run.")
        return

    print(f"\n  Total jobs: {len(jobs)}")

    gpu_queue: Queue[int] = Queue()
    for gpu_id in gpus:
        for _ in range(workers_per_gpu):
            gpu_queue.put(gpu_id)

    failed_runs: list[tuple[str, str, str]] = []

    with ThreadPoolExecutor(max_workers=total_workers) as executor:
        futures = {
            executor.submit(_run_job, job, gpu_queue, args.dry_run, args.skip_existing): job
            for job in jobs
        }
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                failed_runs.append(result)

    total_sec = int(time.time() - run_start)
    log_header("All experiments complete!")
    print(f"  Finished:    {timestamp()}")
    print(f"  Total time:  {total_sec // 3600}h {(total_sec % 3600) // 60}m {total_sec % 60}s")

    if failed_runs:
        print(f"\n  FAILED RUNS ({len(failed_runs)}):")
        for ds, bl, seq in failed_runs:
            print(f"    - {ds} | {bl} | {seq}")
    else:
        print("  All runs succeeded.")

    print("\n  Output locations:")
    for ds_name in selected_datasets:
        dcfg = ALL_DATASETS[ds_name]
        ds_baselines = [b for b in selected_baselines if b in dcfg.baseline_envs]
        for sequence in dcfg.sequences:
            for baseline in ds_baselines:
                frame_ids = dcfg.baseline_frame_ids.get(baseline, [])
                if not frame_ids:
                    continue
                for out in get_expected_output_folders(dcfg, baseline, sequence, frame_ids):
                    print(f"    {dcfg.name} | {baseline} | {sequence}: {out}")
    print("=" * 70)


if __name__ == "__main__":
    main()
