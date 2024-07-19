# hackster.io-AI-video-narration-for-visually-impaired

## Introduction

Visual content is almost an inseparable part of the modern digital world. While movies and videos are a significant form of entertainment and communication in today's society, those who are visually impaired often miss out on the visual aspects of these mediums.
The problem I aim to address is the lack of accessibility to visual content for visually impaired.
I wanted to create a generative tool for voice narration that would describe to the listener the situation that is happening on the screen.

## Solution idea

I built an AI-driven video narration system tailored specifically for visually impaired. Unlike existing solutions, which often rely on pre-recorded audio descriptions, my system utilizes technologies such as open-source Large Language Models (LLMs) for scene interpretation and Python libraries for video and audio processing.

This approach sets my solution apart by offering dynamic descriptions that adapt to the content being viewed, ultimately providing visually impaired users with a more immersive and engaging experience.

## Main features of the solution

![scheme](doc/img/MainFeatures.png)

The above image shows general idea of the solution. From the left side we can see a shot from a movie ("Big Buck Bunny" in this case). The shot along with prepared promped is passed to the neural network. The AI model generates text that describes current situation. At the end the description is synthesized into speech and added to video playback/

- Descriptions of video scenes.
- Identification of objects, actions, and scenes.
- Synchronized narration with video playback.

## Used hardware and software

Hardware:

- AMD Radeon PRO W7900 GPU
- AMD Ryzen 9 7950X
- 1000W power supply
- 64GB RAM

Software:

- OS: Linux (tried Ubuntu 22.04.4 LTS and Manjaro)
- AMD ROCm Software
- Deep learning framework - PyTorch for AI development
- Computer vision and natural language processing libraries, mainly:
  - [OpenCV](https://github.com/opencv/opencv-python)
  - [Transformers](https://huggingface.co/docs/transformers/index)
  - [Pydub](https://github.com/jiaaro/pydub)
  - [PySceneDetect](https://www.scenedetect.com/)
- Video playback software with synchronization capabilities ([VLC media player](https://www.videolan.org)).

## AMD GPU drivers and ROCm installation

For this project I use Linux only.
The first step is to install AMD GPU drivers and ROCm stack and this process varies between different Linux distributions.
You will find the official description here: <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/amdgpu-install.html>

> List of system requirements and supported GPUs:
> <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html>

I tried 2 distributions: Ubuntu 22.04.4 LTS and Manjaro.
Here is how I did it:

---

### [Ubuntu 22.04.4 LTS](https://releases.ubuntu.com/jammy/)

Install drivers and libraries like in the [quickguide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html):

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

### [Manjaro (Plasma desktop)](https://manjaro.org/download/)

This distribution is not officially supported by AMD, but it is not a problem at all.
The Arch repository covers all needed packages, just install `rocm-hip-sdk` via Pamac GUI or run this in terminal:

```shell
sudo pacman -Sy rocm-hip-sdk
```

The packages I installed:

```shell
Packages (42) cmake-3.29.3-1  comgr-6.0.2-1  composable-kernel-6.0.2-1  cppdap-1.58.0-1  hip-runtime-amd-6.0.2-4  hipblas-6.0.2-1  hipcub-6.0.2-1  hipfft-6.0.2-1  hiprand-6.0.2-1  hipsolver-6.0.2-1  hipsparse-6.0.2-1  hsa-rocr-6.0.2-2  hsakmt-roct-6.0.0-2  jsoncpp-1.9.5-2
              libuv-1.48.0-2  miopen-hip-6.0.2-1  numactl-2.0.18-1  opencl-headers-2:2024.05.08-1  openmp-17.0.6-2  rccl-6.0.2-1  rhash-1.4.4-1  rocalution-6.0.2-2  rocblas-6.0.2-1  rocfft-6.0.2-1  rocm-clang-ocl-6.0.2-1  rocm-cmake-6.0.2-1  rocm-core-6.0.2-2
              rocm-device-libs-6.0.2-1  rocm-hip-libraries-6.0.2-1  rocm-hip-runtime-6.0.2-1  rocm-language-runtime-6.0.2-1  rocm-llvm-6.0.2-1  rocm-opencl-runtime-6.0.2-1  rocm-smi-lib-6.0.2-1  rocminfo-6.0.2-1  rocprim-6.0.2-1  rocrand-6.0.2-1  rocsolver-6.0.2-1
              rocsparse-6.0.2-2  rocthrust-6.0.2-1  roctracer-6.0.2-1  rocm-hip-sdk-6.0.2-1
```

Optionally install an additional app for reporting system info:

```shell
sudo pacman -Sy rocminfo
```

The version I used:

```shell
Packages (1) rocminfo-6.0.2-1
```

> [!IMPORTANT]
> After successful installation reboot the system.

---

### Installation verification

Verify drivers installation to ensure you can see your GPU:

Command:

```shell
rocm-smi --showproductname
```

My result on Ubuntu 22.04:

```text
============================ ROCm System Management Interface ============================
====================================== Product Info ======================================
GPU[0]  : Card Series:   0x7448
GPU[0]  : Card Model:   0x7448
GPU[0]  : Card Vendor:   Advanced Micro Devices, Inc. [AMD/ATI]
GPU[0]  : Card SKU:   D7070100
GPU[0]  : Subsystem ID:  0x0e0d
GPU[0]  : Device Rev:   0x00
GPU[0]  : Node ID:   1
GPU[0]  : GUID:   19246
GPU[0]  : GFX Version:   gfx11000
GPU[1]  : Card Series:   0x164e
GPU[1]  : Card Model:   0x164e
GPU[1]  : Card Vendor:   Advanced Micro Devices, Inc. [AMD/ATI]
GPU[1]  : Card SKU:   RAPHAEL
GPU[1]  : Subsystem ID:  0x8877
GPU[1]  : Device Rev:   0xc1
GPU[1]  : Node ID:   2
GPU[1]  : GUID:   9773
GPU[1]  : GFX Version:   gfx1036
==========================================================================================
================================== End of ROCm SMI Log ===================================
```

My result on Manjaro:

```text
============================ ROCm System Management Interface ============================
====================================== Product Info ======================================
GPU[0]          : Card series:          Navi 31 [Radeon Pro W7900]
GPU[0]          : Card model:           0x0e0d
GPU[0]          : Card vendor:          Advanced Micro Devices, Inc. [AMD/ATI]
GPU[0]          : Card SKU:             D7070100
GPU[1]          : Card series:          Raphael
GPU[1]          : Card model:           0x8877
GPU[1]          : Card vendor:          Advanced Micro Devices, Inc. [AMD/ATI]
GPU[1]          : Card SKU:             RAPHAEL
==========================================================================================
================================== End of ROCm SMI Log ===================================
```

Command:

```shell
rocminfo
```

My result on Ubuntu 22.04:

Expect something like this:

```shell
ROCk module version 6.7.0 is loaded
=====================
HSA System Attributes
=====================
Runtime Version:         1.13
Runtime Ext Version:     1.4
System Timestamp Freq.:  1000.000000MHz
Sig. Max Wait Duration:  18446744073709551615 (0xFFFFFFFFFFFFFFFF) (timestamp count)
Machine Model:           LARGE
System Endianness:       LITTLE
Mwaitx:                  DISABLED
DMAbuf Support:          YES

[...]
```

Look for the GPU entry:

```shell
[...]

*******
Agent 2
*******
  Name:                    gfx1100
  Uuid:                    GPU-ecff3f5547b240c7
  Marketing Name:          AMD Radeon PRO W7900
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

My result on Manjaro:

Expect something like this:

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

```shell
[...]

*******
Agent 2
*******
  Name:                    gfx1100
  Uuid:                    GPU-ecff3f5547b240c7
  Marketing Name:          AMD Radeon Pro W7900
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

- <https://rocm.blogs.amd.com/artificial-intelligence/llava-next/README.html>
- <https://huggingface.co/amd>

The drivers installation phase is done and we can proceed to the environment setup.

---

### Conda installation

The conda is used to separate the Python environment and make it independent of the used operating system.
Look at this [site](https://docs.anaconda.com/anaconda/install/linux/) to get into details.

Download the Anaconda installation script.

```shell
wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh
```

Run it. Accept all default options, but at the end allow to initialize conda.

```shell
bash Anaconda3-2024.02-1-Linux-x86_64.sh
```

Allow to initialize:

```text
conda config --set auto_activate_base false

You can undo this by running `conda init --reverse $SHELL`? [yes|no]
[no] >>> yes
```

> [!NOTE]
> Re-open the terminal to continue.
>
> ```text
> ==> For changes to take effect, close and re-open your current shell. <==
> ```

If you use Zsh (default for Manjaro KDE Plasma), then the conda may automatically initialize only for Bash in the `~/.bashrc` file.
To fix it, find the Anaconda installation path and open Zsh terminal.
This will append the `~/.zshrc` file and initialize conda on every Zsh launch:

```shell
<conda-installation-path>/anaconda3/bin/conda init zsh
```

If the installation path was default, then it may look like this:

```shell
~/anaconda3/bin/conda init zsh
```

Check if conda works with the example command.
It lists installed packages in a conda environment:

```shell
conda list
```

### Conda environment setup

Now create and activate a new conda environment:

```shell
conda create -n ai-video-narration-for-visually-impaired-rocm python=3.10
conda activate ai-video-narration-for-visually-impaired-rocm
```

Display your environments to ensure it was activated.
If you haven't had any before, then you should see only `base` and `ai-video-narration-for-visually-impaired-rocm`.

```shell
conda env list
```

Example result, the asterisk `*` means currently activated environment:

```text
# conda environments:
#
base                     /home/<user>/anaconda3
ai-video-narration-for-visually-impaired-rocm  *  /home/<user>/anaconda3/envs/ai-video-narration-for-visually-impaired-rocm
```

### Conda installing packages

> [!NOTE]
> All Python packages are installed in previously activated conda environment.

Install PyTorch for AMD ROCm:

```shell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
```

Check if you cloned this repository.
In the next step you will use the text file with list of requirements.
Proceed to install other remaining packages listed in [other_requirements.txt](other_requirements.txt):

```shell
pip install -r other_requirements.txt
```

To check if PyTorch sees your GPU and to list all visible XXX, run in `python` CLI and paste this:

```python
import torch
print(f"Torch version: {torch.__version__}")
print(f"Is CUDA available?: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"index: {i}; device name: {torch.cuda.get_device_name(i)}")
```

My result:

```text
(ai-video-narration-for-visually-impaired-rocm) <user>@<user>-computer:~/Downloads$ python
Python 3.10.14 (main, May  6 2024, 19:42:50) [GCC 11.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
Torch version: 2.3.1+rocm6.0
Is CUDA available?: True
Number of GPUs: 2
index: 0; device name: AMD Radeon PRO W7900
index: 1; device name: AMD Radeon Graphics
```

If you see more than one GPU device, you may need to export [`HIP_VISIBLE_DEVICES`](https://rocm.docs.amd.com/en/latest/conceptual/gpu-isolation.html#hip-visible-devices) variable to your environemnt every time before launching Python.
This may resolve potential conflicts and solve the problem of inappropriate GPU selection.

> [!IMPORTANT]
> Firstly check the index of your GPU like we did it in Python, it may be other than 0!

```shell
export HIP_VISIBLE_DEVICES=0
```

Now you should see only one GPU device:

```text
(ai-video-narration-for-visually-impaired-rocm) <user>@<user>-computer:~/Downloads$ python
Python 3.10.14 (main, May  6 2024, 19:42:50) [GCC 11.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
Torch version: 2.3.1+rocm6.0
Is CUDA available?: True
Number of GPUs: 1
index: 0; device name: AMD Radeon PRO W7900
```

In my case, the CPU's embedded GPU was also seen by PyTorch and it caused some troubles, which stopped me for a while.
The errors weren't suggestive enough (the closest one I got was when I run [the Docker example](https://rocm.blogs.amd.com/artificial-intelligence/llava-next/README.html)), but the web search has found the solution for me. You may look at these links:

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

### nvtop

Install it via:

#### `pacman` on Manjaro

```shell
sudo pacman -Sy nvtop
```

#### `snap` on Ubuntu 22.04 LTS

```shell
sudo snap instal nvtop
```

If you will try with `apt` then you may encounter this error while running `nvtop`:

```text
nvtop: ./src/extract_gpuinfo_amdgpu.c:324: initDeviceSysfsPaths: Assertion `gpu_info->hwmonDevice != NULL' failed.
Aborted (core dumped)
```

It is caused by the outdated version of that is in `apt` repository.

## Ending

My solution involves testing multiple Large Language Model architectures to determine the most suitable one.
During this process, I assessed potential limitations of the AMD Radeon PRO W7900 GPU when working with these models.
The results of these tests are documented in the project summary.

The efficiency of our visual narration system relies heavily on high-performance GPUs with sufficient VRAM to ensure smooth interpretation speed, measured in FPS (frames per second). Training AI models for such systems demands substantial compute capability. The AMD Radeon PRO W7900 GPU, boasting 48 GB of VRAM and offering 61 TFLOPS for float32 or 122 TFLOPS for float16 aligns with these requirements. Opting for a stationary solution over a cloud-based one enabled us to utilize local data, minimizing network usage and overhead, especially considering the potentially large sizes of the datasets involved, often exceeding 150 GB.

## License

[Big Buck Bunny](https://peach.blender.org/) is licensed under the
[Creative Commons Attribution 3.0 license](http://creativecommons.org/licenses/by/3.0/).

(c) copyright 2008, Blender Foundation / www.bigbuckbunny.org
