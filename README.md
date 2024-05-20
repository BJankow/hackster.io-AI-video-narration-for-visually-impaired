# hackster.io-AI-video-narration-for-visually-impaired

## 1. Problem to solve

The problem I aim to address is the lack of accessibility to visual content for visually impaired individuals. While movies and videos are a significant form of entertainment and communication in today's society, those who are blind or visually impaired often miss out on the visual aspects of these mediums.

## 2. Solution Idea
I will build an AI-driven video narration system tailored specifically for visually impaired individuals. Unlike existing solutions, which often rely on pre-recorded audio descriptions, my system will utilize technologies such as convolutional neural networks (CNN) for human pose detection and description, along with an open-source language model (LLM) for scene interpretation. 

To enhance accuracy and relevance, I'll fine-tune these models using datasets sourced from platforms like OpenDataLab. Furthermore, I plan to employ a hyperparameter tuning framework to optimize the performance of the system. 

This approach sets my solution apart by offering dynamic descriptions that adapt to the content being viewed, ultimately providing visually impaired individuals with a more immersive and engaging viewing experience.

## 3. Main Features of the solution
My solution will involve testing multiple Language Model architectures to determine the most suitable one. During this process, I will assess potential limitations of the AMD Radeon PRO W7900 GPU when working with these models. The results of these tests will be documented in the project summary.

The efficiency of our visual narration system relies heavily on high-performance GPUs with sufficient VRAM to ensure smooth interpretation speed, measured in FPS (frames per second). Training AI models for such systems demands substantial compute capability. The AMD Radeon PRO W7900 GPU, boasting 48 GB of VRAM and offering 61 TFLOPS for float32 or 122 TFLOPS for float16 aligns with these requirements. Opting for a stationary solution over a cloud-based one enables us to utilize local data, minimizing network usage and overhead, especially considering the potentially large sizes of the datasets involved, often exceeding 150GB.

Main Features:
- Descriptions of video scenes.
- Identification of objects, actions, and scenes.
- Synchronized narration with video playback.

## 4. Hardware and Software used to build solution

Hardware:
- AMD Radeon PRO W7900 GPU
- AMD Ryzen 9 7950X
- 1000W power supply
- 64 GB RAM
Software:
- OS: Linux Ubuntu
- AMD ROCm Software
- Deep learning frameworks (TensorFlow or PyTorch) for AI development.
- Computer vision and natural language processing libraries.
- Video playback software with synchronization capabilities.
- Tools for building ML pipeline (e.g. MLflow, kubeflow, git).


## 5. How to observe GPU usage
### gpustat
```commandline
gpustat -a -i 0.5
```
Arguments explaination:
- `-a` - Display all gpu properties above
- `-i 0.5` - Run in watch mode (equivalent to watch gpustat) if given. Denotes interval between updates.

### radeontop
```commandline
sudo pamac install radeontop
radeontop
```

## 6. How to observe CPU usage
### htop
Standard command line program that helps to observe CPU usage.
```commandline
htop -d 10
```
Arguments explaination:
- `-d 10` - Delay between updates, in tenths of a second. If the delay value is less than 1, 
            it is increased to 1, i.e. 1/10 second. If the delay value is greater than 100, 
            it is decreased to 100, i.e. 10 seconds.
### btop
Very nice GUI interface to observe CPU usage
```commandline
sudo snap install btop
```

## 7. How to install ROCm with pyTorch
### 7.0 Helpful links (7.* tip are based on information from those links):
- https://wiki.archlinux.org/title/GPGPU#ROCm (Troubleshooting tips - good source of information)
- https://github.com/rocm-arch/rocm-arch (installation of ROCm dependencies - 7.1)
- https://pytorch.org/get-started/locally/ (torch for python on ROCm )
- https://archlinux.org/packages/extra/x86_64/python-pytorch-rocm/ (Install python-pytorch-rocm)

### 7.1 Most dependencies installation
```commandline
sudo pacman -S rocm-hip-sdk rocm-opencl-sdk
```

### 7.2 Install python-pytorch-rocm
##### Option 1: via package manager
Find all 'rocm' packages - there should be many already installed (7.1 command). You should be able to see 
'python-pytorch-rocm' package that is not installed yet. Click it to be installed

##### Option 2: via terminal

### 7.3 Install torch, torchmetrics, torchsummary, torchvision - all ROCm version
Currently this step is not supported by anaconda (env.yml file etc.), so if you create an anaconda environment be sure 
that you install (**torch, torchmetrics, torchsummary, torchvision**) manually via pip with:
```commandline
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
```