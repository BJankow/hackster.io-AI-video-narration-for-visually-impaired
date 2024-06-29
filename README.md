# hackster.io-AI-video-narration-for-visually-impaired

## Introduction

Visual content is almost an inseparable part of the modern digital world. While movies and videos are a significant form of entertainment and communication in today's society, those who are visually impaired often miss out on the visual aspects of these mediums.
The problem I aim to address is the lack of accessibility to visual content for visually impaired. 
I wanted to create a generative tool for voice narration that would describe to the listener the situation that is happening on the screen. This would be an audio complement for a visual content, just as it happens in books.

## Solution idea

I built an AI-driven video narration system tailored specifically for visually impaired. Unlike existing solutions, which often rely on pre-recorded audio descriptions, my system utilizes technologies such as convolutional neural networks (CNN) for human pose detection and description, along with an open-source language model (LLM) for scene interpretation.

To enhance accuracy and relevance, I fine-tuned these models using datasets sourced from platforms like OpenDataLab. Furthermore, I plan to employ a hyperparameter tuning framework to optimize the performance of the system.

This approach sets my solution apart by offering dynamic descriptions that adapt to the content being viewed, ultimately providing visually impaired users with a more immersive and engaging experience.

## Main features of the solution

My solution involves testing multiple Language Model architectures to determine the most suitable one. During this process, I assessed potential limitations of the AMD Radeon PRO W7900 GPU when working with these models. The results of these tests are documented in the project summary.

The efficiency of our visual narration system relies heavily on high-performance GPUs with sufficient VRAM to ensure smooth interpretation speed, measured in FPS (frames per second). Training AI models for such systems demands substantial compute capability. The AMD Radeon PRO W7900 GPU, boasting 48 GB of VRAM and offering 61 TFLOPS for float32 or 122 TFLOPS for float16 aligns with these requirements. Opting for a stationary solution over a cloud-based one enabled us to utilize local data, minimizing network usage and overhead, especially considering the potentially large sizes of the datasets involved, often exceeding 150 GB.

Main Features:

- Descriptions of video scenes.
- Identification of objects, actions, and scenes.
- Synchronized narration with video playback.

## Hardware and software used to build solution

List for system requirements and supported GPUs:

<https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html>

Hardware:

- AMD Radeon PRO W7900 GPU
- AMD Ryzen 9 7950X
- 1000W power supply
- 64 GB RAM

Software:

- OS: Linux (tried Ubuntu 22.04.4 LTS and Manjaro)
- AMD ROCm Software
- Deep learning frameworks (TensorFlow or PyTorch) for AI development.
- Computer vision and natural language processing libraries.
- Video playback software with synchronization capabilities.
- Tools for building ML pipeline (e.g. MLflow, kubeflow, git).

## Installation

For this project we use only Linux.
The first step is to install AMD GPU drivers and this process varies between different distros.
You will find the official description here: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/

We tried 2 distros: Ubuntu 22.04.4 LTS and Manjaro.
Here is how we did it so you may do it too:

---

### [Ubuntu 22.04.4 LTS](https://releases.ubuntu.com/jammy/)

Install ROCm drivers and libraries like in the [quickguide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html):

```shell
sudo apt update
sudo apt install "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"
sudo usermod -a -G render,video $LOGNAME # Add the current user to the render and video groups
wget https://repo.radeon.com/amdgpu-install/6.1.2/ubuntu/jammy/amdgpu-install_6.1.60102-1_all.deb
sudo apt install ./amdgpu-install_6.1.60102-1_all.deb
sudo apt update
sudo apt install amdgpu-dkms rocm
```

> [!IMPORTANT]
> Now you will need to restart your system.


And that is all! :sunglasses:

---

### Manjaro

This distribution is our favorite
It is not officially supported by AMD, but it is not a problem at all.
The Arch repository covers all needed packages, just install `rocm-hip-sdk` via Pamac GUI or run this in terminal:

```shell
sudo pacman -Sy rocm-hip-sdk
```

The packages we installed:

```shell
Packages (41) cmake-3.29.3-1  comgr-6.0.2-1  composable-kernel-6.0.2-1  cppdap-1.58.0-1  hip-runtime-amd-6.0.2-4  hipblas-6.0.2-1
              hipcub-6.0.2-1  hipfft-6.0.2-1  hiprand-6.0.2-1  hipsolver-6.0.2-1  hipsparse-6.0.2-1  hsa-rocr-6.0.2-2
              hsakmt-roct-6.0.0-2  jsoncpp-1.9.5-2  libuv-1.48.0-2  miopen-hip-6.0.2-1  opencl-headers-2:2024.05.08-1  openmp-17.0.6-2
              rccl-6.0.2-1  rhash-1.4.4-1  rocalution-6.0.2-2  rocblas-6.0.2-1  rocfft-6.0.2-1  rocm-clang-ocl-6.0.2-1
              rocm-cmake-6.0.2-1  rocm-core-6.0.2-2  rocm-device-libs-6.0.2-1  rocm-hip-libraries-6.0.2-1  rocm-hip-runtime-6.0.2-1
              rocm-language-runtime-6.0.2-1  rocm-llvm-6.0.2-1  rocm-opencl-runtime-6.0.2-1  rocm-smi-lib-6.0.2-1  rocminfo-6.0.2-1
              rocprim-6.0.2-1  rocrand-6.0.2-1  rocsolver-6.0.2-1  rocsparse-6.0.2-2  rocthrust-6.0.2-1  roctracer-6.0.2-1
              rocm-hip-sdk-6.0.2-1
```

You may also want to install an additional app for reporting system info:

```shell
sudo pacman -Sy rocminfo
```

The version we used:

```shell
Packages (1) rocminfo-6.0.2-1
```

> [!IMPORTANT]
> After successful installation reboot the system.

---

Now we are complete with the drivers. **Verify drivers installation** with the following commands to ensure you can see your GPU:

Command:

```shell
rocm-smi --showproductname
```

My result:

**TODO: Replace with W7900 entry!**
```shell
============================ ROCm System Management Interface ============================
====================================== Product Info ======================================
GPU[0]          : Card series:          Navi 31 [Radeon RX 7900 XT/7900 XTX/7900M]
GPU[0]          : Card model:           0x5318
GPU[0]          : Card vendor:          Advanced Micro Devices, Inc. [AMD/ATI]
GPU[0]          : Card SKU:             APM7198
==========================================================================================
================================== End of ROCm SMI Log ===================================
```

Command:

```shell
rocminfo
```

Expect something like this:

**TODO: Replace with W7900 entry!**
```shell
ROCk module is loaded
=====================
HSA System Attributes
=====================
Runtime Version:         1.1
System Timestamp Freq.:  1000.000000MHz
Sig. Max Wait Duration:  18446744073709551615 (0xFFFFFFFFFFFFFFFF) (timestamp count)
Machine Model:           LARGE
System Endianness:       LITTLE
Mwaitx:                  DISABLED
DMAbuf Support:          YES

[...]
```

Look for the GPU entry:

**TODO: Replace with W7900 entry!**
```shell
[...]

*******
Agent 2
*******
  Name:                    gfx1100
  Uuid:                    GPU-cbb3e48822a56626
  Marketing Name:          AMD Radeon RX 7900 GRE
  Vendor Name:             AMD
  Feature:                 KERNEL_DISPATCH
  Profile:                 BASE_PROFILE
  Float Round Mode:        NEAR
  Max Queue Number:        128(0x80)
  Queue Min Size:          64(0x40)
  Queue Max Size:          131072(0x20000)
  Queue Type:              MULTI
  Node:                    1
  Device Type:             GPU

[...]
```

Helpful resources:

- <https://wiki.archlinux.org/title/GPGPU#ROCm> (Troubleshooting tips - good source of information)
- <https://github.com/rocm-arch/rocm-arch> (installation of ROCm dependencies)
- <https://pytorch.org/get-started/locally/> (torch for python on ROCm)
- <https://archlinux.org/packages/extra/x86_64/python-pytorch-rocm/> (Install python-pytorch-rocm)

- https://rocm.blogs.amd.com/artificial-intelligence/llava-next/README.html
- https://huggingface.co/amd

---

### Conda

We use Conda to make our environment reproducible and least dependent on the used OS.
Look at this [site](https://docs.anaconda.com/anaconda/install/linux/) to get into details.

To follow our steps download and run the Anaconda installation script. Accept all default options:

```shell
wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh
bash Anaconda3-2024.02-1-Linux-x86_64.sh
```

**Verify with the following command**. It lists installed packages in a conda environment:

```shell
conda list
```


### Conda environment

Now you may want to create and activate a new conda environment, e.g.:

```shell
conda create -n ai-video-narration-for-visually-impaired-rocm python=3.10
conda activate ai-video-narration-for-visually-impaired-rocm
```

Display your environments to ensure it was created. 
If you haven't had any before, then you should see only `base` and `ai-video-narration-for-visually-impaired-rocm`.

```shell
conda env list
```

Example result, the asterisk `*` means currently activated environment:

```
# conda environments:
#
base                     /home/<user>/anaconda3
ai-video-narration-for-visually-impaired-rocm  *  /home/<user>/anaconda3/envs/ai-video-narration-for-visually-impaired-rocm
```

> [!NOTE]
> From now on, we will install all Python packages in this environment.


First, start by installing PyTorch for AMD ROCm:

```shell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
```

Then proceed to install other remaining packages listed in [other_requirements.txt](other_requirements.txt):

```shell
pip install -r other_requirements.txt
```

To check if PyTorch sees your GPU and to list all visible, run in `python` CLI:

```python
import torch
print(f"Torch version: {torch.__version__}")
print(f"Is CUDA available?: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"index: {i}; device name: {torch.cuda.get_device_name(i)}")


```

Expect something like:

**TODO: Replace with W7900 entry!**
```
Torch version: 2.3.1+rocm6.0
Is CUDA available?: True
number of GPUs: 1
['AMD Radeon RX 7900 GRE']
```

If you see more than one device, consider exporting [`HIP_VISIBLE_DEVICES`](https://rocm.docs.amd.com/en/latest/conceptual/gpu-isolation.html#hip-visible-devices) variable before running Python.

> [!IMPORTANT]
> Firstly check the index of your GPU like we did it in Python, it may be other than 0!

```shell
export HIP_VISIBLE_DEVICES=0
```

In my case, the CPU's embedded GPU was also seen by PyTorch and it caused some troubles, which stopped me for a while. The errors weren't enough suggestive (the closest one I got was when I run [the Docker example](https://rocm.blogs.amd.com/artificial-intelligence/llava-next/README.html)), but the web search has found the solution for me. You may look at these link:

- <https://github.com/pytorch/pytorch/issues/119637>
- <https://www.reddit.com/r/ROCm/comments/17e2b5o/rocmpytorch_problem/>
- <https://rocm.docs.amd.com/projects/HIP/en/docs-6.0.0/how_to_guides/debugging.html#making-device-visible>

Resources:

- Check out this official [guide](https://huggingface.co/amd) for ROCm dependent Python libraries.
- we are using the [`transformers`](https://huggingface.co/docs/transformers/index) library, 

### Other resources

PyTorch for ROCm in Docker:

<https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/3rd-party/pytorch-install.html>

## Observe hardware usage

### gpustat

```shell
gpustat -a -i 0.5
```

Arguments explaination:

- `-a` - Display all gpu properties above
- `-i 0.5` - Run in watch mode (equivalent to watch gpustat) if given. Denotes interval between updates.

### radeontop

```shell
sudo pamac install radeontop
radeontop
```

### htop

Standard command line program that helps to observe CPU usage.

```shell
htop -d 10
```

Arguments explaination:

- `-d 10` - Delay between updates, in tenths of a second. If the delay value is less than 1,
            it is increased to 1, i.e. 1/10 second. If the delay value is greater than 100,
            it is decreased to 100, i.e. 10 seconds.

### btop

Very nice GUI interface to observe CPU usage

```shell
sudo snap install btop
```
