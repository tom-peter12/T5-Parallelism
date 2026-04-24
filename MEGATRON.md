# Megatron Setup

This guide covers the extra environment setup needed for the Megatron-based launchers in this repository.

> [!NOTE]
> The commands below assume a Linux environment with CUDA 12.4 and a fresh Conda environment.

## Environment Setup

We recommend using Conda to isolate the Megatron dependencies:

```bash
conda create -n megatron python=3.9
conda activate megatron
```

## Core Dependencies

Install a CUDA-compatible PyTorch build first, then the CUDA toolchain pieces used during extension builds:

```bash
pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
conda install cuda-toolkit cudnn -c nvidia/label/cuda-12.4.0
conda install -c nvidia cuda-nvcc
conda install pytorch-cuda=12.4 -c pytorch -c nvidia
conda install -c conda-forge pybind11
```

> [!IMPORTANT]
> Install PyTorch before building Apex, DeepSpeed, or FlashAttention. These packages compile CUDA extensions against the active PyTorch and CUDA environment.

## Apex Setup

Reference: [NVIDIA/apex](https://github.com/NVIDIA/apex)

First, clone the repository:

```bash
git clone https://github.com/NVIDIA/apex.git
cd apex
```

Then build and install Apex:

```bash
APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install -v --no-build-isolation .
```

If the installation succeeds, continue to [DeepSpeed Installation](#deepspeed-installation).

If Apex fails with a CUDA mismatch error, check the active CUDA path:

```bash
echo $CUDA_HOME
```

If needed, point it to the correct CUDA installation. For example:

```bash
export CUDA_HOME=/usr/local/cuda-12.4
```

If the mismatch check still blocks the build, edit `setup.py` inside the cloned `apex/` repository and comment out the following lines inside `check_cuda_torch_binary_vs_bare_metal`:

```python
if bare_metal_version != torch_binary_version:
    raise RuntimeError(
        "Cuda extensions are being compiled with a version of Cuda that does "
        "not match the version used to compile Pytorch binaries.  "
        "Pytorch binaries were compiled with Cuda {}.\n".format(torch.version.cuda)
        + "In some cases, a minor-version mismatch will not cause later errors:  "
        "https://github.com/NVIDIA/apex/pull/323#discussion_r287021798.  "
        "You can try commenting out this check (at your own risk)."
    )
```

Then retry the install:

```bash
APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install -v --no-build-isolation .
```

> [!WARNING]
> Disabling the CUDA version guard is a workaround. It can unblock builds on clusters with minor CUDA mismatches, but it also increases the risk of runtime instability.

## DeepSpeed Installation

References:

- [DeepSpeed installation guide](https://www.deepspeed.ai/tutorials/advanced-install/)
- [DeepSpeed GitHub repository](https://github.com/deepspeedai/DeepSpeed)
- [Megatron-DeepSpeed GitHub repository](https://github.com/deepspeedai/Megatron-DeepSpeed)

Install DeepSpeed and the Python packages used alongside the Megatron launcher scripts:

```bash
pip install deepspeed==0.17.6 transformers six
```

Clone the Megatron-DeepSpeed repository next to this project:

```bash
cd ..
git clone https://github.com/deepspeedai/Megatron-DeepSpeed.git
cd Megatron-DeepSpeed
pip install -e .
```

Export `PYTHONPATH` so the Megatron package is visible in the current shell:

```bash
export PYTHONPATH=$(pwd)
```

> [!IMPORTANT]
> Run the `PYTHONPATH` export in every terminal used to launch Megatron jobs, on every node.

## FlashAttention

Reference: [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)

Install FlashAttention with:

```bash
pip install flash-attn --no-build-isolation
```

> [!NOTE]
> FlashAttention builds native extensions and can take a while. If this step fails, check that your CUDA, compiler, and PyTorch versions are compatible before debugging higher-level Megatron issues.

## Next Step

Once the environment is ready, return to the main guide for repository-specific setup and launch instructions:

- [README.md](./README.md)
