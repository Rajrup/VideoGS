# GPCC Baseline

## Installation

### 1. Python Dependencies

```bash
pip install numpy plyfile matplotlib
```

PyTorch with CUDA: follow [pytorch.org](https://pytorch.org/get-started/locally/) for your CUDA version.

### 2. LiVoGS CUDA Modules

Two modules require compilation:

```bash
# merge_cluster_cuda (Gaussian merging)
cd LiVoGS/compression/RAHT-3DGS-codec/cuda
pip install .

# gpu_octree_codec (Morton codes, octree)
cd LiVoGS/compression/Octree_Compression_GPU
make              # build C++ library
pip install .     # install Python bindings
```

Voxelization is done in pure PyTorch (sort + unique extraction on Morton codes) and does not require additional modules.

### 3. TMC3 Binary

```bash
git clone https://github.com/MPEGGroup/mpeg-pcc-tmc13.git
cd mpeg-pcc-tmc13
mkdir build && cd build
cmake ..
make -j$(nproc)
```

The binary is at `build/tmc3/tmc3`. Update `GPCC_TMC3_PATH` in the scripts if your path differs from the default.

## Workflow

### Encode

```
┌────────────┐    ┌────────────┐    ┌─────────────┐    ┌────────────┐
│ Load PLY   │    │ Voxelize   │    │ Normalize   │    │ TMC3       │
│ + activate │───▶│ + Merge    │───▶│ + Split     │───▶│ Encode ×24 │
│            │    │            │    │ (RGB→YUV)   │    │ (RAHT+QP)  │
└────────────┘    └────────────┘    └─────────────┘    └────────────┘
```

- **Load PLY + activate**: Parse 3DGS attributes, apply sigmoid (opacity) and exp (scale).
- **Voxelize + Merge**: Quantize positions to integer grid at octree depth J, Morton-code sort, opacity-weighted merge per voxel.
- **Normalize + Split**: Map each attribute to uint8/uint16, RGB-to-YUV for SH coefficients. Split into 24 attribute PLYs (1 opacity + 1 DC + 15 rest triplets + 3 scale + 4 rotation), all sharing the same voxel geometry.
- **TMC3 Encode**: Each of the 24 attribute PLYs is encoded by a separate TMC3 invocation (octree geometry + RAHT attributes, run in parallel).

### Decode

```
┌────────────┐    ┌────────────┐    ┌─────────────┐    ┌────────────┐
│ TMC3       │    │ Morton     │    │ Denormalize │    │ Reassemble │
│ Decode ×24 │───▶│ Sort+Align │───▶│ (YUV→RGB)   │───▶│ + Write PLY│
│            │    │            │    │             │    │            │
└────────────┘    └────────────┘    └─────────────┘    └────────────┘
```

- **TMC3 Decode**: Decompress each of the 24 attribute bitstreams back into reconstructed PLY files (run in parallel).
- **Morton Sort + Align**: Sort decoded points by Morton code to restore consistent point ordering across all 24 decoded PLYs.
- **Denormalize (YUV-to-RGB)**: Map uint8/uint16 back to floats using stored min/max, inverse YUV-to-RGB for SH coefficients.
- **Reassemble + Write PLY**: Concatenate all attributes, apply inverse activations (logit for opacity, log for scale), write final VideoGS PLY.

## Usage

### Single Configuration

```bash
bash scripts/gpcc_baseline/evaluate_gpcc_compression.sh \
    --dataset_path /path/to/checkpoint \
    --output_dir /path/to/output \
    --dataset_eval_path /path/to/dataset \
    --voxel_depth 15 \
    --qp_rest 40 --qp_dc 4 --qp_opacity 4
```

### Rate-Distortion Sweep

In `scripts/run_rd_baselines_experiments.py`, set:

```python
RUN_BASELINES = ("gpcc",)
```

Sweep parameters (explicit QP combos x octree depths):

| Parameter | Default values | Controls |
|---|---|---|
| `GPCC_OCTREE_DEPTHS_BY_DATASET` | `(12, 14, 15)` | Voxelization granularity |
| `GPCC_QP_COMBOS` | 43 `(qp_rest, qp_dc, qp_opacity)` tuples | Attribute quantization |

Scale and rotation attributes use a fixed QP of 4 (not swept).

```bash
python scripts/run_rd_baselines_experiments.py
```
