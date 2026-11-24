#!/bin/bash
# One-time environment build for LiveSplats on an ECE T4 machine.
# Safe to re-run: finished steps are skipped.
set -eo pipefail
source /afs/ece.cmu.edu/usr/hbommaka/Private/grail/env.sh
set -x
cd $SCRATCH
if [ ! -d miniforge3 ]; then
  curl -sSL -o miniforge.sh https://github.com/conda-forge/miniforge/releases/download/24.11.3-0/Miniforge3-24.11.3-0-Linux-x86_64.sh
  bash miniforge.sh -b -p $SCRATCH/miniforge3 && rm miniforge.sh
fi
source $SCRATCH/miniforge3/etc/profile.d/conda.sh
[ -d envs/livesplat ] || conda create -y -p $SCRATCH/envs/livesplat python=3.12
conda activate $SCRATCH/envs/livesplat

# System GCC is 8.5 (too old for PyTorch); gcc-toolset 13/14 are too new for CUDA 12.1.
# GCC 11 from conda-forge satisfies both.
[ -x $CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++ ] || conda install -y -c conda-forge gcc_linux-64=11 gxx_linux-64=11
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CUDAHOSTCXX=$CXX

python -c "import torch" 2>/dev/null || pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
# The repo has no requirements.txt; these are the third-party imports found in livesplats/
pip install "numpy<2" "matplotlib<3.10" opencv-python-headless imageio kornia lpips plyfile nvidia-ml-py nvtx roma scipy tqdm websockets ninja setuptools wheel

cd $GRAIL/basket-multiview/livesplats
for m in diff-gaussian-rasterization fused-ssim simple-knn diff-gaussian-rasterization-accelerated bonebone; do
  MAX_JOBS=16 pip install --no-build-isolation submodules/$m
done
python -c "import torch, diff_gaussian_rasterization, diff_gaussian_rasterization_accelerated, simple_knn, fused_ssim, bone_bone; print('OK torch', torch.__version__, torch.cuda.get_device_name(0))"
