#!/usr/bin/env python3
"""
Batch training script for VideoGS (wraps train_sequence.py).

Edit the JOBS list below to define your sequences.
Jobs are dispatched dynamically: the first free GPU picks up the next job.
Within a single GPU, jobs run sequentially.
Across different GPUs, jobs run in parallel.

Usage:
    python batch_train_sequence.py [--gpus 0 1 2 ...]
"""

import os
import sys
import argparse
import subprocess
from multiprocessing import Pool, Manager
from pathlib import Path

# Holds the GPU ID assigned to each worker process at init time.
_WORKER_GPU = None

# =============================================================================
# JOBS — edit this list to define your training runs
# =============================================================================

BASE_DATA   = "/synology/rajrup/VideoGS/HiFi4G_Dataset_processed"
BASE_OUTPUT = "/synology/rajrup/VideoGS/train_output/HiFi4G_Dataset"

DEFAULTS = dict(sh=3, interval=1, group_size=20, resolution=2)

# Each entry is a dict with keys: data, output, start, end,
# and optionally: sh, interval, group_size, resolution, marching_cubes_res.
# Missing keys fall back to DEFAULTS above.
# GPU assignment is automatic: pass --gpus to choose which GPUs are used.
JOBS = [
    {
        "data":   f"{BASE_DATA}/4K_Actor1_Greeting",
        "output": f"{BASE_OUTPUT}/4K_Actor1_Greeting",
        "start":  0,
        "end":    200,
    },
    {
        "data":   f"{BASE_DATA}/4K_Actor2_Dancing",
        "output": f"{BASE_OUTPUT}/4K_Actor2_Dancing",
        "start":  0,
        "end":    200,
    },
    {
        "data":   f"{BASE_DATA}/4K_Actor3_Violin",
        "output": f"{BASE_OUTPUT}/4K_Actor3_Violin",
        "start":  0,
        "end":    200,
    },
    {
        "data":   f"{BASE_DATA}/4K_Actor4_Dancing",
        "output": f"{BASE_OUTPUT}/4K_Actor4_Dancing",
        "start":  0,
        "end":    200,
    },
    {
        "data":   f"{BASE_DATA}/4K_Actor5_Oil-paper_Umbrella",
        "output": f"{BASE_OUTPUT}/4K_Actor5_Oil-paper_Umbrella",
        "start":  0,
        "end":    200,
    },
    {
        "data":   f"{BASE_DATA}/4K_Actor6_Changing_Clothes",
        "output": f"{BASE_OUTPUT}/4K_Actor6_Changing_Clothes",
        "start":  0,
        "end":    200,
    },
    {
        "data":   f"{BASE_DATA}/4K_Actor7_Nunchaku",
        "output": f"{BASE_OUTPUT}/4K_Actor7_Nunchaku",
        "start":  0,
        "end":    200,
    },
]

# =============================================================================


def _worker_init(gpu_queue):
    """Called once per worker process; claims a GPU ID from the shared queue."""
    global _WORKER_GPU
    _WORKER_GPU = gpu_queue.get()


def _run_job(job: dict) -> int:
    """Run one job on this worker's GPU; returns the process return code."""
    job = dict(job)
    job["cuda"] = _WORKER_GPU

    cmd = [
        sys.executable, "train_sequence.py",
        "--start",      str(job["start"]),
        "--end",        str(job["end"]),
        "--cuda",       str(job["cuda"]),
        "--data",       job["data"],
        "--output",     job["output"],
        "--sh",         str(job.get("sh",         DEFAULTS["sh"])),
        "--interval",   str(job.get("interval",   DEFAULTS["interval"])),
        "--group_size", str(job.get("group_size", DEFAULTS["group_size"])),
        "--resolution", str(job.get("resolution", DEFAULTS["resolution"])),
    ]
    if job.get("marching_cubes_res") is not None:
        cmd += ["--marching_cubes_res", str(job["marching_cubes_res"])]

    log_dir = Path(job["output"]) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    # Use a timestamped log so reruns don't overwrite previous logs.
    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"train_sequence_{ts}.log"

    label = Path(job["data"]).name
    print(f"[START] {label}  frames {job['start']}-{job['end']}  GPU {job['cuda']}", flush=True)
    print(f"        log -> {log_path}", flush=True)

    # Ensure the conda env's libstdc++ is found before the system one.
    conda_lib = os.path.normpath(os.path.join(os.path.dirname(sys.executable), "..", "lib"))
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = conda_lib + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    env["CUDA_VISIBLE_DEVICES"] = str(job["cuda"])

    with open(log_path, "w") as log_file:
        result = subprocess.run(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            env=env,
        )

    status = "DONE" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
    print(f"[{status}] {label}  frames {job['start']}-{job['end']}  GPU {job['cuda']}", flush=True)
    return result.returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0],
                        help="GPU IDs to use. One job runs per GPU at a time (default: 0).")
    args = parser.parse_args()

    jobs = JOBS
    if not jobs:
        raise SystemExit("JOBS list is empty — add entries at the top of the script.")

    gpus = args.gpus
    if len(gpus) != len(set(gpus)):
        raise SystemExit(f"Duplicate GPU IDs in --gpus {gpus}. Each GPU must appear at most once.")

    print(f"\nBatch training: {len(jobs)} job(s) on GPU(s) {gpus}  "
          f"(max {len(gpus)} job(s) in parallel)\n")
    for j in jobs:
        print(f"  {Path(j['data']).name}  frames {j['start']}-{j['end']}")
    print()

    # Each worker process claims one GPU from the queue at startup and keeps it
    # for all the jobs it runs. The pool dispatches jobs dynamically, so a free
    # GPU picks up the next job as soon as it finishes the current one.
    with Manager() as manager:
        gpu_queue = manager.Queue()
        for gpu in gpus:
            gpu_queue.put(gpu)

        with Pool(
            processes=len(gpus),
            initializer=_worker_init,
            initargs=(gpu_queue,),
        ) as pool:
            return_codes = pool.map(_run_job, jobs)

    n_failed = sum(rc != 0 for rc in return_codes)
    print(f"\nAll jobs finished.  {len(return_codes) - n_failed}/{len(return_codes)} succeeded.")
    if n_failed:
        raise SystemExit(1)
