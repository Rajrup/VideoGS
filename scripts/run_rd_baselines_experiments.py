#!/usr/bin/env python3
"""Run first-frame R-D baseline experiments and collect results.

This script runs parameter sweeps for VideoGS, MesonGS, and DracoGS on the
first frame of each sequence for HiFi4G and N3DV, then collects CSV results.

Plotting is intentionally split into a separate script:
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

FIRST_FRAME_SETTINGS: dict[str, int] = {
    "HiFi4G": 0,
    "N3DV": 1,
}


VIDEOGS_QPS = list(range(0, 41))
VIDEOGS_GROUP_SIZE = 20

MESONGS_DEPTHS: dict[str, tuple[int, ...]] = {
    "HiFi4G": (8, 10, 12),
    "N3DV": (12, 14, 16),
}
MESONGS_NUM_BITS = (8, 16)
MESONGS_N_BLOCKS = (57, 66)
MESONGS_CODEBOOK_SIZES = (2048, 4096)

DRACOGS_EG = (0, 8, 12, 16)
DRACOGS_EO = (0, 8, 12, 16)
DRACOGS_ET = (0, 8, 12, 16)
DRACOGS_ES = (0, 8, 12, 16)
DRACOGS_CL = 10

BASELINES = [
    "videogs", 
    # "dracogs",
    # "mesongs",
]
CUDA_DEVICES = ["0", "1"]
SKIP_EXISTING = True


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
    data_path="/synology/rajrup/VideoGS",
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
    data_path="/synology/rajrup/Queen",
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

RUN_DRY_RUN = False
RUN_SKIP_EXISTING = SKIP_EXISTING
RUN_SKIP_SAVED_RESULTS = False
RUN_DATASETS = list(ALL_DATASETS.keys())
RUN_BASELINES = list(BASELINES)
RUN_SEQUENCES_OVERRIDE: Optional[list[str]] = None
RUN_COLLECT_ONLY = False
RUN_RUN_ONLY = False


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


def _first_frame(cfg: DatasetConfig) -> int:
    return FIRST_FRAME_SETTINGS[cfg.name]


def _frame_span(cfg: DatasetConfig) -> tuple[int, int, int]:
    fid = _first_frame(cfg)
    if cfg.frame_end_exclusive:
        return fid, fid + 1, 1
    return fid, fid, 1


def _videogs_frame_span(cfg: DatasetConfig) -> tuple[int, int, int]:
    fid = _first_frame(cfg)
    return fid, fid + VIDEOGS_GROUP_SIZE, 1


def _frame_span_tag(cfg: DatasetConfig) -> str:
    fs, fe, iv = _frame_span(cfg)
    if cfg.frame_end_exclusive:
        return f"frames_{fs}_{fe - 1}_int_{iv}"
    return f"frames_{fs}_{fe}_int_{iv}"


def _videogs_frame_span_tag(cfg: DatasetConfig) -> str:
    fs, fe, iv = _videogs_frame_span(cfg)
    return f"frames_{fs}_{fe - 1}_int_{iv}"


def _videogs_output_folder(cfg: DatasetConfig, sequence: str, qp: int) -> str:
    return str(_model_root(cfg, sequence) / "compression" / "videogs" / f"qp_{qp}" / _videogs_frame_span_tag(cfg))


def _mesongs_output_folder(
    cfg: DatasetConfig,
    sequence: str,
    depth: int,
    num_bits: int,
    n_block: int,
    cb: int,
) -> str:
    params_tag = f"d{depth}_nb{num_bits}_nblk{n_block}_cb{cb}"
    return str(_model_root(cfg, sequence) / "compression" / "mesongs" / params_tag / _frame_span_tag(cfg))


def _dracogs_output_folder(
    cfg: DatasetConfig,
    sequence: str,
    eg: int,
    eo: int,
    et: int,
    es: int,
) -> str:
    params_tag = f"eg_{eg}_eo_{eo}_et_{et}_es_{es}_cl_{DRACOGS_CL}"
    return str(_model_root(cfg, sequence) / "compression" / "dracogs" / params_tag / _frame_span_tag(cfg))


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
    devices = list(CUDA_DEVICES)
    if not devices:
        raise ValueError("CUDA_DEVICES must contain at least one device id")
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
    output_folder: str,
    dry_run: bool,
    cuda_device: str,
) -> None:
    fs, fe, iv = _frame_span(cfg)
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
    fs, fe, iv = _videogs_frame_span(cfg)
    gt = _gt_model_path(cfg, sequence)
    project_root = _project_root(cfg)
    qps = list(VIDEOGS_QPS)

    def run_single(qp: int, cuda_device: str) -> list[tuple[str, str, str]]:
        failures: list[tuple[str, str, str]] = []
        output_folder = _videogs_output_folder(cfg, sequence, qp)
        if skip_existing and _output_complete(output_folder):
            log_step(f"SKIP VideoGS | {cfg.name} | {sequence} | qp={qp}")
            return failures

        log_step(f"VideoGS | {cfg.name} | {sequence} | qp={qp} | cuda={cuda_device} | {timestamp()}")
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
                str(VIDEOGS_GROUP_SIZE),
                "--sh_degree",
                str(cfg.sh_degree),
                "--qp",
                str(qp),
            ],
        )
        try:
            run_cmd(cmd, cwd=project_root, dry_run=dry_run, cuda_device=cuda_device)
            run_evaluation(cfg, sequence, output_folder, dry_run, cuda_device)
        except subprocess.CalledProcessError as exc:
            print(f"WARNING: VideoGS qp={qp} failed for {cfg.name}/{sequence} (exit {exc.returncode})")
            _cleanup_partial(output_folder)
            failures.append((cfg.name, "videogs", sequence))
        return failures

    return _run_tasks_across_devices(qps, run_single)


def run_mesongs_rd(
    cfg: DatasetConfig,
    sequence: str,
    dry_run: bool,
    skip_existing: bool,
) -> list[tuple[str, str, str]]:
    fs, fe, iv = _frame_span(cfg)
    gt = _gt_model_path(cfg, sequence)
    ds = _dataset_path(cfg, sequence)
    project_root = _project_root(cfg)
    mesongs_root = _mesongs_root(cfg)
    tasks = list(
        itertools.product(
            MESONGS_DEPTHS[cfg.name],
            MESONGS_NUM_BITS,
            MESONGS_N_BLOCKS,
            MESONGS_CODEBOOK_SIZES,
        )
    )

    def run_single(task: tuple[int, int, int, int], cuda_device: str) -> list[tuple[str, str, str]]:
        failures: list[tuple[str, str, str]] = []
        depth, num_bits, n_block, cb = task
        output_folder = _mesongs_output_folder(cfg, sequence, depth, num_bits, n_block, cb)
        short = f"d={depth} nb={num_bits} nblk={n_block} cb={cb}"
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
            run_evaluation(cfg, sequence, output_folder, dry_run, cuda_device)
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
    fs, fe, iv = _frame_span(cfg)
    gt = _gt_model_path(cfg, sequence)
    project_root = _project_root(cfg)
    tasks = list(itertools.product(DRACOGS_EG, DRACOGS_EO, DRACOGS_ET, DRACOGS_ES))

    def run_single(task: tuple[int, int, int, int], cuda_device: str) -> list[tuple[str, str, str]]:
        failures: list[tuple[str, str, str]] = []
        eg, eo, et, es = task
        output_folder = _dracogs_output_folder(cfg, sequence, eg, eo, et, es)
        short = f"eg={eg} eo={eo} et={et} es={es}"
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
                str(DRACOGS_CL),
            ],
        )
        try:
            run_cmd(cmd, cwd=project_root, dry_run=dry_run, cuda_device=cuda_device)
            run_evaluation(cfg, sequence, output_folder, dry_run, cuda_device)
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
    fid = _first_frame(cfg)

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

    def discover_output_folders_for_qp(qp: int) -> list[tuple[str, Optional[int]]]:
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

                if detected_frame_start is not None and detected_frame_start != fid:
                    continue
                candidates.append((str(out_path), detected_group_size))

        default_out = _videogs_output_folder(cfg, sequence, qp)
        default_group_size = VIDEOGS_GROUP_SIZE
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

    for qp in VIDEOGS_QPS:
        for out, group_size in discover_output_folders_for_qp(qp):
            row = _load_single_frame_result(
                out,
                BENCHMARK_CSV_NAMES["videogs"],
                fid,
                compressed_size_field="compressed_size_gop_avg_bytes",
            )
            if row is None:
                continue
            param_suffix = f" g={group_size}" if group_size is not None else ""
            row.update(
                dataset=cfg.name,
                sequence=sequence,
                frame_id=fid,
                baseline="VideoGS",
                params=f"qp={qp}{param_suffix}",
                group_size=group_size,
            )
            rows.append(row)
    return rows


def collect_mesongs(cfg: DatasetConfig, sequence: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fid = _first_frame(cfg)
    for depth, num_bits, n_block, cb in itertools.product(
        MESONGS_DEPTHS[cfg.name], MESONGS_NUM_BITS, MESONGS_N_BLOCKS, MESONGS_CODEBOOK_SIZES
    ):
        out = _mesongs_output_folder(cfg, sequence, depth, num_bits, n_block, cb)
        row = _load_single_frame_result(out, BENCHMARK_CSV_NAMES["mesongs"], fid)
        if row is None:
            continue
        row.update(
            dataset=cfg.name,
            sequence=sequence,
            frame_id=fid,
            baseline="MesonGS",
            params=f"d={depth} nb={num_bits} nblk={n_block} cb={cb}",
        )
        rows.append(row)
    return rows


def collect_dracogs(cfg: DatasetConfig, sequence: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fid = _first_frame(cfg)
    for eg, eo, et, es in itertools.product(DRACOGS_EG, DRACOGS_EO, DRACOGS_ET, DRACOGS_ES):
        out = _dracogs_output_folder(cfg, sequence, eg, eo, et, es)
        row = _load_single_frame_result(out, BENCHMARK_CSV_NAMES["dracogs"], fid)
        if row is None:
            continue
        row.update(
            dataset=cfg.name,
            sequence=sequence,
            frame_id=fid,
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


def _selected_sequences(cfg: DatasetConfig, override: Optional[list[str]]) -> list[str]:
    seqs = list(SEQUENCE_SETTINGS[cfg.name])
    if not override:
        return seqs
    return [s for s in override if s in seqs]


def _validate_selected_items(selected: list[str], allowed: set[str], item_name: str) -> None:
    invalid = [item for item in selected if item not in allowed]
    if invalid:
        joined = ", ".join(invalid)
        raise ValueError(f"Unknown {item_name}(s): {joined}")


def _combo_counts(cfg: DatasetConfig, baselines: list[str], n_sequences: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    if "videogs" in baselines:
        counts["VideoGS"] = len(VIDEOGS_QPS) * n_sequences
    if "mesongs" in baselines:
        counts["MesonGS"] = (
            len(MESONGS_DEPTHS[cfg.name])
            * len(MESONGS_NUM_BITS)
            * len(MESONGS_N_BLOCKS)
            * len(MESONGS_CODEBOOK_SIZES)
            * n_sequences
        )
    if "dracogs" in baselines:
        counts["DracoGS"] = len(DRACOGS_EG) * len(DRACOGS_EO) * len(DRACOGS_ET) * len(DRACOGS_ES) * n_sequences
    return counts


def main() -> None:
    run_start = time.time()

    datasets = list(RUN_DATASETS)
    baselines = list(RUN_BASELINES)
    sequences_override = list(RUN_SEQUENCES_OVERRIDE) if RUN_SEQUENCES_OVERRIDE else None
    dry_run = bool(RUN_DRY_RUN)
    collect_only = bool(RUN_COLLECT_ONLY)
    run_only = bool(RUN_RUN_ONLY)
    skip_saved_results = bool(RUN_SKIP_SAVED_RESULTS)
    effective_skip_existing = bool(RUN_SKIP_EXISTING or skip_saved_results)

    _validate_selected_items(datasets, set(ALL_DATASETS.keys()), "dataset")
    _validate_selected_items(baselines, set(BASELINE_RUNNERS.keys()), "baseline")
    if collect_only and run_only:
        raise ValueError("RUN_COLLECT_ONLY and RUN_RUN_ONLY cannot both be True")

    log_header("R-D Baselines First-Frame Experiments")
    print(f"  Started:             {timestamp()}")
    print(f"  Datasets:            {', '.join(datasets)}")
    print(f"  Baselines:           {', '.join(baselines)}")
    print(f"  CUDA devices:        {', '.join(CUDA_DEVICES)}")
    print(f"  Output root:         {RD_BASELINES_RESULTS_ROOT}")
    print(f"  skip-existing:       {effective_skip_existing}")
    print(f"  skip-saved-results:  {skip_saved_results}")
    if dry_run:
        print("  Mode:                DRY RUN")
    if collect_only:
        print("  Mode:                COLLECT ONLY")
    if run_only:
        print("  Mode:                RUN ONLY")

    for ds_name in datasets:
        cfg = ALL_DATASETS[ds_name]
        seqs = _selected_sequences(cfg, sequences_override)
        counts = _combo_counts(cfg, baselines, len(seqs))
        total = sum(counts.values())
        details = ", ".join(f"{k}={v}" for k, v in counts.items())
        print(f"  {cfg.name}: frame={_first_frame(cfg)}, sequences={len(seqs)}, {details}, total={total}")
    print("=" * 70)

    failed_runs: list[tuple[str, str, str]] = []

    if not collect_only:
        for baseline in baselines:
            for ds_name in datasets:
                cfg = ALL_DATASETS[ds_name]
                seqs = _selected_sequences(cfg, sequences_override)
                for sequence in seqs:
                    log_header(f"{cfg.name} | {sequence}")
                    runner = BASELINE_RUNNERS[baseline]
                    step_start = time.time()
                    failures = runner(cfg, sequence, dry_run, effective_skip_existing)
                    failed_runs.extend(failures)
                    elapsed = int(time.time() - step_start)
                    print(f"  {baseline.upper()} | {cfg.name} | {sequence} completed in {elapsed}s")

    all_rows: list[dict[str, Any]] = []
    by_dataset_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in datasets}

    if not run_only:
        log_header("Collecting CSV results")
        for baseline in baselines:
            for ds_name in datasets:
                cfg = ALL_DATASETS[ds_name]
                seqs = _selected_sequences(cfg, sequences_override)
                for sequence in seqs:
                    collector = BASELINE_COLLECTORS[baseline]
                    rows = collector(cfg, sequence)
                    print(f"  {cfg.name} | {sequence} | {baseline.upper()}: {len(rows)}")
                    all_rows.extend(rows)
                    by_dataset_rows[ds_name].extend(rows)

        os.makedirs(RD_BASELINES_COLLECTED_DIR, exist_ok=True)
        write_results_csv(all_rows, str(RD_BASELINES_COLLECTED_DIR / "rd_baselines_results_all.csv"), skip_saved_results)
        for ds_name, rows in by_dataset_rows.items():
            write_results_csv(rows, str(RD_BASELINES_COLLECTED_DIR / f"rd_baselines_results_{ds_name}.csv"), skip_saved_results)

    total_sec = int(time.time() - run_start)
    summary = {
        "finished": timestamp(),
        "datasets": datasets,
        "baselines": baselines,
        "skip_existing": effective_skip_existing,
        "skip_saved_results": skip_saved_results,
        "dry_run": dry_run,
        "collect_only": collect_only,
        "run_only": run_only,
        "total_seconds": total_sec,
        "failed_runs": [{"dataset": d, "baseline": b, "sequence": s} for d, b, s in failed_runs],
        "collected_rows": len(all_rows),
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
    print(f"  Collected:   {RD_BASELINES_COLLECTED_DIR}")
    print(f"  Summary:     {RD_BASELINES_RUN_SUMMARY_JSON}")
    print("=" * 70)


if __name__ == "__main__":
    main()
