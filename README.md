# VideoGS on Jetson Orin

We will only test the performance of LiVoGS on VideoGS trained models on Jetson Orin. So we won't need to install all dependencies.

## Setup

System details:
```
Jetpack 6.2.1
CUDA 12.6
Ubuntu 22.04
CMake version 3.22.1
GCC version 11.4.0
Python version 3.10
PyTorch version 2.5.0
Torchvision version 0.20.1
```

## Install Dependencies

**Dependencies for VideoGS:**

```bash
git clone --recurse-submodules https://github.com/Rajrup/VideoGS.git
conda create -n videogs python=3.10 -y
conda activate videogs
pip install numpy==1.26.4
pip install opencv-python==4.11.0.86
pip install Cython

# Install PyTorch and torchvision
pip install https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
# Install torchvision from source: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
sudo apt install libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev
pip install pillow
git clone --branch <version> https://github.com/pytorch/vision torchvision
cd torchvision
git checkout tags/v0.20.1
export BUILD_VERSION=0.20.1
python setup.py install
sudo apt-get install python3-dev libxml2-dev libxslt1-dev zlib1g-dev libjsoncpp-dev

pip install torchmetrics[image]
pip install -r requirements.txt --no-build-isolation 
pip install submodules/diff-gaussian-rasterization --no-build-isolation # Only pytorch 2.4 with CUDA 12.1 worked. Other versions resulted in compilation errors.
pip install submodules/simple-knn --no-build-isolation

# Other optional dependencies
pip install pynvml psutil matplotlib
```

**Dependencies for LiVoGS:**

Follow the instructions in `README_LiVoGS.md` to set up LiVoGS.

## Compression

#### Running the LiVoGS compression+decompression pipeline on VideoGS trained models on HiFi4G Dataset

```bash
bash scripts/livogs_baseline/evaluate_livogs_compression.sh
```
