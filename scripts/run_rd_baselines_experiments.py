#!/usr/bin/env python3
"""Run selected-frame R-D baseline experiments.

This script runs parameter sweeps for VideoGS, MesonGS, and DracoGS on the
selected frames of each sequence for HiFi4G and N3DV.

Result collection and plotting are intentionally handled by:
  scripts/plot_rd_baselines_results.py
"""

from __future__ import annotations

import csv
import itertools
import json
import os
import glob
import re
import shlex
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

RUN_DRY_RUN = False
RUN_SKIP_EXISTING = True
RUN_SKIP_SAVED_RESULTS = False
RUN_DATASETS: tuple[str, ...] = ("HiFi4G", "N3DV")
RUN_BASELINES: tuple[str, ...] = ("videogs", "dracogs", "mesongs")
RUN_CUDA_DEVICES: tuple[str, ...] = ("0", "1")

HIFI4G_DATA_ROOT = "/synology/rajrup/VideoGS"
N3DV_DATA_ROOT = "/synology/rajrup/Queen"

VIDEOGS_QP_VALUES: tuple[int, ...] = tuple(range(0, 41))
VIDEOGS_GROUP_SIZE = 20

MESONGS_DEPTHS_BY_DATASET: dict[str, tuple[int, ...]] = {
    "HiFi4G": (8, 10, 12),
    "N3DV": (12, 14, 16),
}
MESONGS_NUM_BITS: tuple[int, ...] = (8, 16)
MESONGS_N_BLOCKS: tuple[int, ...] = (57, 66)
MESONGS_CODEBOOK_SIZES: tuple[int, ...] = (2048, 4096)

DRACOGS_EG: tuple[int, ...] = (0, 8, 12, 16)
DRACOGS_EO: tuple[int, ...] = (0, 8, 12, 16)
DRACOGS_ET: tuple[int, ...] = (0, 8, 12, 16)
DRACOGS_ES: tuple[int, ...] = (0, 8, 12, 16)
DRACOGS_CL = 10


SEQUENCE_SETTINGS: dict[str, tuple[str, ...]] = {
    "HiFi4G": (
        "4K_Actor1_Greeting",
        "4K_Actor2_Dancing",
        "4K_Actor3_Violin",
        "4K_Actor4_Dancing",
        "4K_Actor5_Oil-paper_Umbrella",
        "4K_Actor6_Changing_Clothes",
        "4K_Actor7_Nunchaku",
    ),
    "N3DV": (
        "cook_spinach",
        "coffee_martini",
        "cut_roasted_beef",
        "flame_salmon_1",
        "flame_steak",
        "sear_steak",
    ),
}

FRAME_ID_LISTS: dict[str, tuple[int, ...]] = {
    "HiFi4G": (0,),
    "N3DV": (1,),
}


@dataclass(frozen=True)
class SweepSpaceConfig:
    videogs_qps: tuple[int, ...]
    videogs_group_size: int
    mesongs_depths_by_dataset: dict[str, tuple[int, ...]]
    mesongs_num_bits: tuple[int, ...]
    mesongs_n_blocks: tuple[int, ...]
    mesongs_codebook_sizes: tuple[int, ...]
    dracogs_eg: tuple[int, ...]
    dracogs_eo: tuple[int, ...]
    dracogs_et: tuple[int, ...]
    dracogs_es: tuple[int, ...]
    dracogs_cl: int


@dataclass(frozen=True)
class RunConfig:
    dry_run: bool
    skip_existing: bool
    skip_saved_results: bool
    datasets: tuple[str, ...]
    baselines: tuple[str, ...]
    cuda_devices: tuple[str, ...]

SWEEP_SPACE = SweepSpaceConfig(
    videogs_qps=VIDEOGS_QP_VALUES,
    videogs_group_size=VIDEOGS_GROUP_SIZE,
    mesongs_depths_by_dataset=MESONGS_DEPTHS_BY_DATASET,
    mesongs_num_bits=MESONGS_NUM_BITS,
    mesongs_n_blocks=MESONGS_N_BLOCKS,
    mesongs_codebook_sizes=MESONGS_CODEBOOK_SIZES,
    dracogs_eg=DRACOGS_EG,
    dracogs_eo=DRACOGS_EO,
    dracogs_et=DRACOGS_ET,
    dracogs_es=DRACOGS_ES,
    dracogs_cl=DRACOGS_CL,
)


SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parent
WORKSPACE_ROOT = SCRIPTS_DIR.parent.parent

RD_BASELINES_RESULTS_ROOT = SCRIPTS_DIR / "rd_baselines_results"
RD_BASELINES_COLLECTED_DIR = RD_BASELINES_RESULTS_ROOT / "collected"
RD_BASELINES_RUN_SUMMARY_JSON = RD_BASELINES_RESULTS_ROOT / "run_summary.json"


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
    frame_end_exclusive: bool
    mesongs_script_name: str
    mesongs_has_resolution_arg: bool


HIFI4G = DatasetConfig(
    name="HiFi4G",
    dataset_name="HiFi4G_Dataset",
    data_path=HIFI4G_DATA_ROOT,
    project_name="VideoGS",
    sh_degree=3,
    resolution=2,
    evaluation_env="videogs",
    baseline_envs={"dracogs": "videogs", "mesongs": "mesongs", "videogs": "videogs"},
    frame_end_exclusive=True,
    mesongs_script_name="compress_decompress_pipeline.py",
    mesongs_has_resolution_arg=True,
)

N3DV = DatasetConfig(
    name="N3DV",
    dataset_name="Neural_3D_Video",
    data_path=N3DV_DATA_ROOT,
    project_name="queen",
    sh_degree=2,
    resolution=2,
    evaluation_env="queen",
    baseline_envs={"dracogs": "queen", "mesongs": "mesongs", "videogs": "queen"},
    frame_end_exclusive=False,
    mesongs_script_name="compression_decompress_pipeline.py",
    mesongs_has_resolution_arg=False,
)

ALL_DATASETS: dict[str, DatasetConfig] = {"HiFi4G": HIFI4G, "N3DV": N3DV}

RUN_CONFIG = RunConfig(
    dry_run=RUN_DRY_RUN,
    skip_existing=RUN_SKIP_EXISTING,
    skip_saved_results=RUN_SKIP_SAVED_RESULTS,
    datasets=RUN_DATASETS,
    baselines=RUN_BASELINES,
    cuda_devices=RUN_CUDA_DEVICES,
)

def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_header(msg: str) -> None:
    print(f"\n{'=' * 70}\n  {msg}\n{'=' * 70}")


def log_step(msg: str) -> None:
    print(f"--- {msg}")


def _project_root(cfg: DatasetConfig) -> Path:
    return WORKSPACE_ROOT / cfg.project_name


def _mesongs_root(cfg: DatasetConfig) -> Path:
    return _project_root(cfg) / "MesonGS"


def _model_root(cfg: DatasetConfig, sequence: str) -> Path:
    if cfg.name == "HiFi4G":
        return Path(cfg.data_path) / "train_output" / cfg.dataset_name / sequence
    return Path(cfg.data_path) / "pretrained_output" / cfg.dataset_name / f"queen_compressed_{sequence}"


def _gt_model_path(cfg: DatasetConfig, sequence: str) -> str:
    if cfg.name == "HiFi4G":
        return str(_model_root(cfg, sequence) / "checkpoint")
    return str(_model_root(cfg, sequence))


def _dataset_path(cfg: DatasetConfig, sequence: str) -> str:
    if cfg.name == "HiFi4G":
        return str(Path(cfg.data_path) / f"{cfg.dataset_name}_processed" / sequence)
    return str(Path(cfg.data_path) / cfg.dataset_name / sequence)


def _frame_ids(cfg: DatasetConfig) -> tuple[int, ...]:
    frame_ids = FRAME_ID_LISTS[cfg.name]
    if not frame_ids:
        raise ValueError(f"FRAME_ID_LISTS[{cfg.name!r}] must contain at least one frame id")
    return frame_ids


def _frame_span(cfg: DatasetConfig, frame_id: int) -> tuple[int, int, int]:
    fid = frame_id
    if cfg.frame_end_exclusive:
        return fid, fid + 1, 1
    return fid, fid, 1


def _videogs_frame_span(cfg: DatasetConfig, frame_id: int) -> tuple[int, int, int]:
    fid = frame_id
    return fid, fid + SWEEP_SPACE.videogs_group_size, 1


def _frame_span_tag(cfg: DatasetConfig, frame_id: int) -> str:
    fs, fe, iv = _frame_span(cfg, frame_id)
    if cfg.frame_end_exclusive:
        return f"frames_{fs}_{fe - 1}_int_{iv}"
    return f"frames_{fs}_{fe}_int_{iv}"


def _videogs_frame_span_tag(cfg: DatasetConfig, frame_id: int) -> str:
    fs, fe, iv = _videogs_frame_span(cfg, frame_id)
    return f"frames_{fs}_{fe - 1}_int_{iv}"


def _videogs_output_folder(cfg: DatasetConfig, sequence: str, frame_id: int, qp: int) -> str:
    return str(
        _model_root(cfg, sequence)
        / "compression"
        / "videogs"
        / f"qp_{qp}"
        / _videogs_frame_span_tag(cfg, frame_id)
    )


def _mesongs_output_folder(
    cfg: DatasetConfig,
    sequence: str,
    frame_id: int,
    depth: int,
    num_bits: int,
    n_block: int,
    cb: int,
) -> str:
    params_tag = f"d{depth}_nb{num_bits}_nblk{n_block}_cb{cb}"
    return str(_model_root(cfg, sequence) / "compression" / "mesongs" / params_tag / _frame_span_tag(cfg, frame_id))


def _dracogs_output_folder(
    cfg: DatasetConfig,
    sequence: str,
    frame_id: int,
    eg: int,
    eo: int,
    et: int,
    es: int,
) -> str:
    params_tag = f"eg_{eg}_eo_{eo}_et_{et}_es_{es}_cl_{SWEEP_SPACE.dracogs_cl}"
    return str(_model_root(cfg, sequence) / "compression" / "dracogs" / params_tag / _frame_span_tag(cfg, frame_id))


def run_cmd(cmd: list[str], cwd: Path, dry_run: bool, cuda_device: str) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    if dry_run:
        print(f"[DRY RUN] cwd={cwd} | CUDA_VISIBLE_DEVICES={cuda_device} | {shlex.join(cmd)}")
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


T = TypeVar("T")


def _run_tasks_across_devices(
    tasks: list[T],
    task_runner: Callable[[T, str], list[tuple[str, str, str]]],
) -> list[tuple[str, str, str]]:
    devices = list(RUN_CONFIG.cuda_devices)
    if not devices:
        raise ValueError("RUN_CONFIG.cuda_devices must contain at least one device id")
    if len(devices) == 1:
        failures: list[tuple[str, str, str]] = []
        for task in tasks:
            failures.extend(task_runner(task, devices[0]))
        return failures

    per_device_tasks: dict[str, list[T]] = {device: [] for device in devices}
    for idx, task in enumerate(tasks):
        device = devices[idx % len(devices)]
        per_device_tasks[device].append(task)

    def run_device_queue(device: str, queue: list[T]) -> list[tuple[str, str, str]]:
        device_failures: list[tuple[str, str, str]] = []
        for queued_task in queue:
            device_failures.extend(task_runner(queued_task, device))
        return device_failures

    failures: list[tuple[str, str, str]] = []
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [executor.submit(run_device_queue, device, queue) for device, queue in per_device_tasks.items()]
        for future in futures:
            failures.extend(future.result())
    return failures


def run_evaluation(
    cfg: DatasetConfig,
    sequence: str,
    frame_id: int,
    output_folder: str,
    dry_run: bool,
    cuda_device: str,
) -> None:
    fs, fe, iv = _frame_span(cfg, frame_id)
    gt = _gt_model_path(cfg, sequence)
    ds = _dataset_path(cfg, sequence)
    project_root = _project_root(cfg)

    if cfg.name == "HiFi4G":
        cmd = conda_python_cmd(
            cfg.evaluation_env,
            project_root / "scripts" / "evaluate_decompress.py",
            [
                "--gt_ply_path",
                gt,
                "--decompressed_ply_path",
                f"{output_folder}/decompressed_ply",
                "--dataset_path",
                ds,
                "--output_render_path",
                f"{output_folder}/evaluation",
                "--sh_degree",
                str(cfg.sh_degree),
                "--resolution",
                str(cfg.resolution),
                "--frame_start",
                str(fs),
                "--frame_end",
                str(fe),
                "--interval",
                str(iv),
            ],
        )
    else:
        cmd = conda_python_cmd(
            cfg.evaluation_env,
            project_root / "scripts" / "evaluate_decompress.py",
            [
                "--config",
                "configs/dynerf.yaml",
                "-s",
                ds,
                "-m",
                gt,
                "--decompressed_ply_path",
                f"{output_folder}/decompressed_ply",
                "--output_render_path",
                f"{output_folder}/evaluation",
                "--frame_start",
                str(fs),
                "--frame_end",
                str(fe),
                "--interval",
                str(iv),
            ],
        )
    run_cmd(cmd, cwd=project_root, dry_run=dry_run, cuda_device=cuda_device)


def run_videogs_rd(
    cfg: DatasetConfig,
    sequence: str,
    dry_run: bool,
    skip_existing: bool,
) -> list[tuple[str, str, str]]:
    gt = _gt_model_path(cfg, sequence)
    project_root = _project_root(cfg)
    tasks = list(itertools.product(_frame_ids(cfg), SWEEP_SPACE.videogs_qps))

    def run_single(task: tuple[int, int], cuda_device: str) -> list[tuple[str, str, str]]:
        failures: list[tuple[str, str, str]] = []
        frame_id, qp = task
        fs, fe, iv = _videogs_frame_span(cfg, frame_id)
        output_folder = _videogs_output_folder(cfg, sequence, frame_id, qp)
        if skip_existing and _output_complete(output_folder):
            log_step(f"SKIP VideoGS | {cfg.name} | {sequence} | f={frame_id} | qp={qp}")
            return failures

        log_step(f"VideoGS | {cfg.name} | {sequence} | f={frame_id} | qp={qp} | cuda={cuda_device} | {timestamp()}")
        cmd = conda_python_cmd(
            cfg.baseline_envs["videogs"],
            project_root / "scripts" / "videogs_baseline" / "compress_decompress_pipeline.py",
            [
                "--ply_path",
                gt,
                "--output_folder",
                output_folder,
                "--output_ply_folder",
                f"{output_folder}/decompressed_ply",
                "--frame_start",
                str(fs),
                "--frame_end",
                str(fe),
                "--interval",
                str(iv),
                "--group_size",
                str(SWEEP_SPACE.videogs_group_size),
                "--sh_degree",
                str(cfg.sh_degree),
                "--qp",
                str(qp),
            ],
        )
        try:
            run_cmd(cmd, cwd=project_root, dry_run=dry_run, cuda_device=cuda_device)
            run_evaluation(cfg, sequence, frame_id, output_folder, dry_run, cuda_device)
        except subprocess.CalledProcessError as exc:
            print(
                f"WARNING: VideoGS frame={frame_id} qp={qp} "
                f"failed for {cfg.name}/{sequence} (exit {exc.returncode})"
            )
            _cleanup_partial(output_folder)
            failures.append((cfg.name, "videogs", sequence))
        return failures

    return _run_tasks_across_devices(tasks, run_single)


def run_mesongs_rd(
    cfg: DatasetConfig,
    sequence: str,
    dry_run: bool,
    skip_existing: bool,
) -> list[tuple[str, str, str]]:
    gt = _gt_model_path(cfg, sequence)
    ds = _dataset_path(cfg, sequence)
    project_root = _project_root(cfg)
    mesongs_root = _mesongs_root(cfg)
    tasks = list(
        itertools.product(
            _frame_ids(cfg),
            SWEEP_SPACE.mesongs_depths_by_dataset[cfg.name],
            SWEEP_SPACE.mesongs_num_bits,
            SWEEP_SPACE.mesongs_n_blocks,
            SWEEP_SPACE.mesongs_codebook_sizes,
        )
    )

    def run_single(task: tuple[int, int, int, int, int], cuda_device: str) -> list[tuple[str, str, str]]:
        failures: list[tuple[str, str, str]] = []
        frame_id, depth, num_bits, n_block, cb = task
        fs, fe, iv = _frame_span(cfg, frame_id)
        output_folder = _mesongs_output_folder(cfg, sequence, frame_id, depth, num_bits, n_block, cb)
        short = f"f={frame_id} d={depth} nb={num_bits} nblk={n_block} cb={cb}"
        if skip_existing and _output_complete(output_folder):
            log_step(f"SKIP MesonGS | {cfg.name} | {sequence} | {short}")
            return failures

        log_step(f"MesonGS | {cfg.name} | {sequence} | {short} | cuda={cuda_device} | {timestamp()}")

        mesongs_args = [
            "--ply_path",
            gt,
            "--dataset_path",
            ds,
            "--output_folder",
            output_folder,
            "--output_ply_folder",
            f"{output_folder}/decompressed_ply",
            "--frame_start",
            str(fs),
            "--frame_end",
            str(fe),
            "--interval",
            str(iv),
            "--sh_degree",
            str(cfg.sh_degree),
            "--scene_name",
            sequence,
            "--depth",
            str(depth),
            "--num_bits",
            str(num_bits),
            "--n_block",
            str(n_block),
            "--codebook_size",
            str(cb),
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
            run_evaluation(cfg, sequence, frame_id, output_folder, dry_run, cuda_device)
        except subprocess.CalledProcessError as exc:
            print(f"WARNING: MesonGS {short} failed for {cfg.name}/{sequence} (exit {exc.returncode})")
            _cleanup_partial(output_folder)
            failures.append((cfg.name, "mesongs", sequence))
        return failures

    return _run_tasks_across_devices(tasks, run_single)


def run_dracogs_rd(
    cfg: DatasetConfig,
    sequence: str,
    dry_run: bool,
    skip_existing: bool,
) -> list[tuple[str, str, str]]:
    gt = _gt_model_path(cfg, sequence)
    project_root = _project_root(cfg)
    tasks = list(
        itertools.product(
            _frame_ids(cfg),
            SWEEP_SPACE.dracogs_eg,
            SWEEP_SPACE.dracogs_eo,
            SWEEP_SPACE.dracogs_et,
            SWEEP_SPACE.dracogs_es,
        )
    )

    def run_single(task: tuple[int, int, int, int, int], cuda_device: str) -> list[tuple[str, str, str]]:
        failures: list[tuple[str, str, str]] = []
        frame_id, eg, eo, et, es = task
        fs, fe, iv = _frame_span(cfg, frame_id)
        output_folder = _dracogs_output_folder(cfg, sequence, frame_id, eg, eo, et, es)
        short = f"f={frame_id} eg={eg} eo={eo} et={et} es={es}"
        if skip_existing and _output_complete(output_folder):
            log_step(f"SKIP DracoGS | {cfg.name} | {sequence} | {short}")
            return failures

        log_step(f"DracoGS | {cfg.name} | {sequence} | {short} | cuda={cuda_device} | {timestamp()}")
        cmd = conda_python_cmd(
            cfg.baseline_envs["dracogs"],
            project_root / "scripts" / "dracogs_baseline" / "compress_decompress_pipeline.py",
            [
                "--ply_path",
                gt,
                "--output_folder",
                output_folder,
                "--output_ply_folder",
                f"{output_folder}/decompressed_ply",
                "--frame_start",
                str(fs),
                "--frame_end",
                str(fe),
                "--interval",
                str(iv),
                "--sh_degree",
                str(cfg.sh_degree),
                "--scene_name",
                sequence,
                "--eg",
                str(eg),
                "--eo",
                str(eo),
                "--et",
                str(et),
                "--es",
                str(es),
                "--cl",
                str(SWEEP_SPACE.dracogs_cl),
            ],
        )
        try:
            run_cmd(cmd, cwd=project_root, dry_run=dry_run, cuda_device=cuda_device)
            run_evaluation(cfg, sequence, frame_id, output_folder, dry_run, cuda_device)
        except subprocess.CalledProcessError as exc:
            print(f"WARNING: DracoGS {short} failed for {cfg.name}/{sequence} (exit {exc.returncode})")
            _cleanup_partial(output_folder)
            failures.append((cfg.name, "dracogs", sequence))
        return failures

    return _run_tasks_across_devices(tasks, run_single)


BASELINE_RUNNERS: dict[
    str,
    Callable[[DatasetConfig, str, bool, bool], list[tuple[str, str, str]]],
] = {
    "videogs": run_videogs_rd,
    "mesongs": run_mesongs_rd,
    "dracogs": run_dracogs_rd,
}


BENCHMARK_CSV_NAMES: dict[str, str] = {
    "videogs": "benchmark_videogs_pipeline.csv",
    "mesongs": "benchmark_mesongs.csv",
    "dracogs": "benchmark_dracogs.csv",
}


def _load_single_frame_result(
    output_folder: str,
    benchmark_csv_name: str,
    frame_id: int,
    compressed_size_field: str = "compressed_size_bytes",
) -> Optional[dict[str, Any]]:
    benchmark_path = os.path.join(output_folder, benchmark_csv_name)
    eval_json_path = os.path.join(output_folder, "evaluation", "evaluation_results.json")

    compressed_bytes: Optional[int] = None
    uncompressed_bytes = 0
    if os.path.isfile(benchmark_path):
        try:
            with open(benchmark_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if int(row["frame_id"]) == frame_id:
                        comp_raw = row.get(compressed_size_field) or row.get("compressed_size_bytes")
                        if comp_raw is None:
                            break
                        compressed_bytes = int(comp_raw)
                        uncompressed_bytes = int(row.get("uncompressed_size_bytes", 0))
                        break
        except (OSError, KeyError, ValueError):
            pass

    decomp_psnr: Optional[float] = None
    decomp_ssim: Optional[float] = None
    gt_psnr: Optional[float] = None
    gt_ssim: Optional[float] = None
    if os.path.isfile(eval_json_path):
        try:
            with open(eval_json_path, encoding="utf-8") as f:
                eval_data = json.load(f)
            for fr in eval_data.get("per_frame", []):
                if int(fr["frame"]) == frame_id:
                    decomp_psnr = float(fr["decomp_psnr"])
                    decomp_ssim = float(fr["decomp_ssim"])
                    gt_psnr = float(fr["gt_psnr"])
                    gt_ssim = float(fr["gt_ssim"])
                    break
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            pass

    if compressed_bytes is None or decomp_psnr is None:
        return None

    return {
        "compressed_size_bytes": compressed_bytes,
        "compressed_mb": compressed_bytes / (1024 * 1024),
        "uncompressed_size_bytes": uncompressed_bytes,
        "decomp_psnr": decomp_psnr,
        "decomp_ssim": decomp_ssim,
        "gt_psnr": gt_psnr,
        "gt_ssim": gt_ssim,
    }


def collect_videogs(cfg: DatasetConfig, sequence: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    frame_ids = _frame_ids(cfg)

    frame_tag_re = re.compile(r"^frames_(\d+)_(\d+)_int_(\d+)$")

    def infer_group_size_from_tag(out_path: Path) -> Optional[int]:
        m = frame_tag_re.match(out_path.name)
        if m is None:
            return None
        start = int(m.group(1))
        end = int(m.group(2))
        interval = int(m.group(3))
        if interval <= 0 or end < start:
            return None
        return ((end - start) // interval) + 1

    def discover_output_folders_for_qp(qp: int, frame_id: int) -> list[tuple[str, Optional[int]]]:
        qp_root = _model_root(cfg, sequence) / "compression" / "videogs" / f"qp_{qp}"
        candidates: list[tuple[str, Optional[int]]] = []

        if qp_root.is_dir():
            for out_dir in sorted(glob.glob(str(qp_root / "frames_*_int_*"))):
                out_path = Path(out_dir)
                config_path = out_path / "videogs_config.json"
                detected_group_size = infer_group_size_from_tag(out_path)
                detected_frame_start: Optional[int] = None
                if config_path.is_file():
                    try:
                        with open(config_path, encoding="utf-8") as f:
                            config = json.load(f)
                        raw_start = config.get("frame_start")
                        if raw_start is not None:
                            detected_frame_start = int(raw_start)
                    except (OSError, json.JSONDecodeError, ValueError, TypeError):
                        pass

                if detected_frame_start is not None and detected_frame_start != frame_id:
                    continue
                candidates.append((str(out_path), detected_group_size))

        default_out = _videogs_output_folder(cfg, sequence, frame_id, qp)
        default_group_size = SWEEP_SPACE.videogs_group_size
        if not any(out == default_out for out, _ in candidates):
            candidates.append((default_out, default_group_size))

        unique: list[tuple[str, Optional[int]]] = []
        seen: set[str] = set()
        for out, gsize in candidates:
            if out in seen:
                continue
            seen.add(out)
            unique.append((out, gsize))
        return unique

    for frame_id in frame_ids:
        for qp in SWEEP_SPACE.videogs_qps:
            for out, group_size in discover_output_folders_for_qp(qp, frame_id):
                row = _load_single_frame_result(
                    out,
                    BENCHMARK_CSV_NAMES["videogs"],
                    frame_id,
                    compressed_size_field="compressed_size_gop_avg_bytes",
                )
                if row is None:
                    continue
                param_suffix = f" g={group_size}" if group_size is not None else ""
                row.update(
                    dataset=cfg.name,
                    sequence=sequence,
                    frame_id=frame_id,
                    baseline="VideoGS",
                    params=f"qp={qp}{param_suffix}",
                    group_size=group_size,
                )
                rows.append(row)
    return rows


def collect_mesongs(cfg: DatasetConfig, sequence: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frame_id in _frame_ids(cfg):
        for depth, num_bits, n_block, cb in itertools.product(
            SWEEP_SPACE.mesongs_depths_by_dataset[cfg.name],
            SWEEP_SPACE.mesongs_num_bits,
            SWEEP_SPACE.mesongs_n_blocks,
            SWEEP_SPACE.mesongs_codebook_sizes,
        ):
            out = _mesongs_output_folder(cfg, sequence, frame_id, depth, num_bits, n_block, cb)
            row = _load_single_frame_result(out, BENCHMARK_CSV_NAMES["mesongs"], frame_id)
            if row is None:
                continue
            row.update(
                dataset=cfg.name,
                sequence=sequence,
                frame_id=frame_id,
                baseline="MesonGS",
                params=f"d={depth} nb={num_bits} nblk={n_block} cb={cb}",
            )
            rows.append(row)
    return rows


def collect_dracogs(cfg: DatasetConfig, sequence: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frame_id in _frame_ids(cfg):
        for eg, eo, et, es in itertools.product(
            SWEEP_SPACE.dracogs_eg,
            SWEEP_SPACE.dracogs_eo,
            SWEEP_SPACE.dracogs_et,
            SWEEP_SPACE.dracogs_es,
        ):
            out = _dracogs_output_folder(cfg, sequence, frame_id, eg, eo, et, es)
            row = _load_single_frame_result(out, BENCHMARK_CSV_NAMES["dracogs"], frame_id)
            if row is None:
                continue
            row.update(
                dataset=cfg.name,
                sequence=sequence,
                frame_id=frame_id,
                baseline="DracoGS",
                params=f"eg={eg} eo={eo} et={et} es={es}",
            )
            rows.append(row)
    return rows


BASELINE_COLLECTORS: dict[str, Callable[[DatasetConfig, str], list[dict[str, Any]]]] = {
    "videogs": collect_videogs,
    "mesongs": collect_mesongs,
    "dracogs": collect_dracogs,
}


CSV_COLUMNS = [
    "dataset",
    "sequence",
    "frame_id",
    "baseline",
    "params",
    "group_size",
    "compressed_size_bytes",
    "compressed_mb",
    "uncompressed_size_bytes",
    "decomp_psnr",
    "decomp_ssim",
    "gt_psnr",
    "gt_ssim",
]


def write_results_csv(rows: list[dict[str, Any]], path: str, skip_saved_results: bool) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if skip_saved_results and os.path.isfile(path):
        print(f"  SKIP saved CSV (exists): {path}")
        return

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"  Wrote {len(rows)} rows to: {path}")


def write_run_summary(summary: dict[str, Any], skip_saved_results: bool) -> None:
    os.makedirs(str(RD_BASELINES_RESULTS_ROOT), exist_ok=True)
    if skip_saved_results and RD_BASELINES_RUN_SUMMARY_JSON.is_file():
        print(f"  SKIP saved summary (exists): {RD_BASELINES_RUN_SUMMARY_JSON}")
        return
    with open(RD_BASELINES_RUN_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  Wrote summary: {RD_BASELINES_RUN_SUMMARY_JSON}")


def _selected_sequences(cfg: DatasetConfig) -> list[str]:
    return list(SEQUENCE_SETTINGS[cfg.name])


def _validate_selected_items(selected: list[str], allowed: set[str], item_name: str) -> None:
    invalid = [item for item in selected if item not in allowed]
    if invalid:
        joined = ", ".join(invalid)
        raise ValueError(f"Unknown {item_name}(s): {joined}")


def _combo_counts(cfg: DatasetConfig, baselines: list[str], n_sequences: int, n_frames: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    if "videogs" in baselines:
        counts["VideoGS"] = len(SWEEP_SPACE.videogs_qps) * n_sequences * n_frames
    if "mesongs" in baselines:
        counts["MesonGS"] = (
            len(SWEEP_SPACE.mesongs_depths_by_dataset[cfg.name])
            * len(SWEEP_SPACE.mesongs_num_bits)
            * len(SWEEP_SPACE.mesongs_n_blocks)
            * len(SWEEP_SPACE.mesongs_codebook_sizes)
            * n_sequences
            * n_frames
        )
    if "dracogs" in baselines:
        counts["DracoGS"] = (
            len(SWEEP_SPACE.dracogs_eg)
            * len(SWEEP_SPACE.dracogs_eo)
            * len(SWEEP_SPACE.dracogs_et)
            * len(SWEEP_SPACE.dracogs_es)
            * n_sequences
            * n_frames
        )
    return counts


def main() -> None:
    run_start = time.time()

    datasets = list(RUN_CONFIG.datasets)
    baselines = list(RUN_CONFIG.baselines)
    dry_run = bool(RUN_CONFIG.dry_run)
    skip_saved_results = bool(RUN_CONFIG.skip_saved_results)
    effective_skip_existing = bool(RUN_CONFIG.skip_existing or skip_saved_results)

    _validate_selected_items(datasets, set(ALL_DATASETS.keys()), "dataset")
    _validate_selected_items(baselines, set(BASELINE_RUNNERS.keys()), "baseline")

    log_header("R-D Baselines Selected-Frame Experiments")
    print(f"  Started:             {timestamp()}")
    print(f"  Datasets:            {', '.join(datasets)}")
    print(f"  Baselines:           {', '.join(baselines)}")
    print(f"  CUDA devices:        {', '.join(RUN_CONFIG.cuda_devices)}")
    print(f"  Output root:         {RD_BASELINES_RESULTS_ROOT}")
    print(f"  skip-existing:       {effective_skip_existing}")
    print(f"  skip-saved-results:  {skip_saved_results}")
    print("  Collection:          delegated to plot_rd_baselines_results.py")
    if dry_run:
        print("  Mode:                DRY RUN")

    for ds_name in datasets:
        cfg = ALL_DATASETS[ds_name]
        seqs = _selected_sequences(cfg)
        frame_ids = _frame_ids(cfg)
        counts = _combo_counts(cfg, baselines, len(seqs), len(frame_ids))
        total = sum(counts.values())
        details = ", ".join(f"{k}={v}" for k, v in counts.items())
        print(f"  {cfg.name}: frames={list(frame_ids)}, sequences={len(seqs)}, {details}, total={total}")
    print("=" * 70)

    failed_runs: list[tuple[str, str, str]] = []

    for baseline in baselines:
        for ds_name in datasets:
            cfg = ALL_DATASETS[ds_name]
            seqs = _selected_sequences(cfg)
            for sequence in seqs:
                log_header(f"{cfg.name} | {sequence}")
                runner = BASELINE_RUNNERS[baseline]
                step_start = time.time()
                failures = runner(cfg, sequence, dry_run, effective_skip_existing)
                failed_runs.extend(failures)
                elapsed = int(time.time() - step_start)
                print(f"  {baseline.upper()} | {cfg.name} | {sequence} completed in {elapsed}s")

    total_sec = int(time.time() - run_start)
    summary = {
        "finished": timestamp(),
        "datasets": datasets,
        "baselines": baselines,
        "skip_existing": effective_skip_existing,
        "skip_saved_results": skip_saved_results,
        "dry_run": dry_run,
        "total_seconds": total_sec,
        "failed_runs": [{"dataset": d, "baseline": b, "sequence": s} for d, b, s in failed_runs],
    }
    write_run_summary(summary, skip_saved_results)

    log_header("Done")
    print(f"  Finished:    {timestamp()}")
    print(f"  Total time:  {total_sec // 3600}h {(total_sec % 3600) // 60}m {total_sec % 60}s")
    if failed_runs:
        print(f"  Failed runs: {len(failed_runs)}")
        for d, b, s in failed_runs:
            print(f"    - {d} | {b} | {s}")
    else:
        print("  Failed runs: 0")
    print("  Collected:   delegated to plot_rd_baselines_results.py")
    print(f"  Summary:     {RD_BASELINES_RUN_SUMMARY_JSON}")
    print("=" * 70)


if __name__ == "__main__":
    main()
