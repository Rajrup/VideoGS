## Setup

- Ubuntu 24.04
- GCC 12.4.0
- CUDA 12.1
- CuDNN 9.17.1

## Install Dependencies

**Dependencies for VideoGS:**

```bash
git clone --recurse-submodules https://github.com/Rajrup/VideoGS.git
conda create -n videogs python=3.10 -y
conda activate videogs
pip install numpy==1.26.4
pip install opencv-python==4.11.0.86
pip install mkl==2023.2.0
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install torchmetrics[image]
pip install -r requirements.txt --no-build-isolation 
pip install submodules/diff-gaussian-rasterization --no-build-isolation # Only pytorch 2.4 with CUDA 12.1 worked. Other versions resulted in compilation errors.
pip install submodules/simple-knn --no-build-isolation

# External Dependencies
# Download OptiX 7.6 from https://developer.nvidia.com/designworks/optix/downloads/legacy
sh NVIDIA-OptiX-SDK-7.6.0-linux64-x86_64.sh
echo 'export OptiX_INSTALL_DIR=/home/rajrup/NVIDIA-OptiX-SDK-7.6.0-linux64-x86_64' >> ~/.bashrc
source ~/.bashrc

# Install NeuS2
mkdir -p external && cd external
git clone --recursive https://github.com/AuthorityWang/NeuS2_K.git
cd NeuS2_K/dependencies/ && rm -rf pybind11
git clone --recursive https://github.com/pybind/pybind11.git
cd pybind11
git fetch --tags
git checkout v2.13.6 # Newer pybind11 is necessary for GCC 12.4.0

cd ../../
# cmake . -B build -> This takes /usr/bin/nvcc which 12.0. Results in runtime error during training.
cmake . -B build -DCMAKE_CUDA_COMPILER=/home/rajrup/cuda-12.1/bin/nvcc
cmake --build build --config RelWithDebInfo -j
```

**Dependencies for LiVoGS:**

Follow the instructions in `README_LiVoGS.md` to set up LiVoGS.

## Download Dataset

```bash
pip install huggingface_hub
```

### HiFi4G Dataset Download

**Download Location:** `/synology/rajrup/Datasets/HiFi4G_Dataset/`

```bash
bash preprocess/hifi4g_download.sh

# unzip the downloaded dataset
```

### HiFi4G Dataset Preprocess

**Processed Location:** `/synology/rajrup/VideoGS/HiFi4G_Dataset_processed/`

```bash
cd preprocess
bash hifi4g_preprocess.sh
```

## Training on HiFi4G Dataset

```bash
python train_sequence.py --start 0 --end 200 --cuda 0 --data /synology/rajrup/VideoGS/HiFi4G_Dataset_processed/4K_Actor1_Greeting --output /synology/rajrup/VideoGS/train_output/HiFi4G_Dataset/4K_Actor1_Greeting --sh 3 --interval 1 --group_size 20 --resolution 2

python train_sequence.py --start 0 --end 200 --cuda 0 --data /synology/rajrup/VideoGS/HiFi4G_Dataset_processed/4K_Actor2_Dancing --output /synology/rajrup/VideoGS/train_output/HiFi4G_Dataset/4K_Actor2_Dancing --sh 3 --interval 1 --group_size 20 --resolution 2

python train_sequence.py --start 0 --end 200 --cuda 0 --data /synology/rajrup/VideoGS/HiFi4G_Dataset_processed/4K_Actor4_Dancing --output /synology/rajrup/VideoGS/train_output/HiFi4G_Dataset/4K_Actor4_Dancing --sh 3 --interval 1 --group_size 20 --resolution 2

python train_sequence.py --start 0 --end 200 --cuda 0 --data /synology/rajrup/VideoGS/HiFi4G_Dataset_processed/4K_Actor5_Oil-paper_Umbrella --output /synology/rajrup/VideoGS/train_output/HiFi4G_Dataset/4K_Actor5_Oil-paper_Umbrella --sh 3 --interval 1 --group_size 20 --resolution 2

python train_sequence.py --start 0 --end 200 --cuda 0 --data /synology/rajrup/VideoGS/HiFi4G_Dataset_processed/4K_Actor6_Changing_Clothes --output /synology/rajrup/VideoGS/train_output/HiFi4G_Dataset/4K_Actor6_Changing_Clothes --sh 3 --interval 1 --group_size 20 --resolution 2

python train_sequence.py --start 0 --end 200 --cuda 0 --data /synology/rajrup/VideoGS/HiFi4G_Dataset_processed/4K_Actor7_Nunchaku --output /synology/rajrup/VideoGS/train_output/HiFi4G_Dataset/4K_Actor7_Nunchaku --sh 3 --interval 1 --group_size 20 --resolution 2
```

## Compression

### Our compression script

**Note:** The VideoGS baseline pipeline script and plot scripts live in **`scripts/videogs_baseline/`**. From the project root you can run the full compress→decompress→evaluate pipeline with `bash scripts/videogs_baseline/evaluate_videogs_compression.sh`, then generate plots with the scripts in `scripts/videogs_baseline/plots/` (see that folder’s README).

#### Running the VideoGS pipeline on HiFi4G Dataset

```bash
bash scripts/videogs_baseline/evaluate_videogs_compression.sh
```

#### Generating plots for HiFi4G Dataset

```bash
bash scripts/videogs_baseline/plots/plot_benchmark.sh
```
