# Ultralytics YOLO 🚀, AGPL-3.0 license
# Builds ultralytics/ultralytics:latest image on DockerHub https://hub.docker.com/r/ultralytics/ultralytics
# Image is CUDA-optimized for YOLO11 single/multi-GPU training and inference

# Start FROM PyTorch image https://hub.docker.com/r/pytorch/pytorch or nvcr.io/nvidia/pytorch:23.03-py3

FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime
#FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime
#FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

# Added
#RUN apt-get update
#RUN apt-get install python3.9
#FROM python:3.9.13
#End Added

# Set environment variables
# Avoid DDP error "MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1 library" https://github.com/pytorch/pytorch/issues/37377
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    MKL_THREADING_LAYER=GNU \
    OMP_NUM_THREADS=1 

# Downloads to user config dir
ADD https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.ttf \
    https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.Unicode.ttf \
    /root/.config/Ultralytics/

# Install linux packages
# g++ required to build 'tflite_support' and 'lap' packages, libusb-1.0-0 required for 'tflite_support' package
# libsm6 required by libqxcb to create QT-based windows for visualization; set 'QT_DEBUG_PLUGINS=1' to test in docker
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc git zip unzip wget curl htop libgl1 libglib2.0-0 libpython3-dev gnupg g++ libusb-1.0-0 libsm6 \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6
RUN apt install -y libgl1-mesa-glx
# Security updates
# https://security.snyk.io/vuln/SNYK-UBUNTU1804-OPENSSL-3314796
RUN apt upgrade --no-install-recommends -y openssl tar

# Create working directory
WORKDIR /ultralytics


# Added Requirements Artiamas
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt
# Added Requirements Artiamas
# Copy contents and configure git
COPY . .
#RUN sed -i '/^\[http "https:\/\/github\.com\/"\]/,+1d' .git/config
#ADD https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt .

# Install pip packages
RUN python3 -m pip install --upgrade pip wheel
# Note -cu12 must be used with tensorrt)
#RUN pip install -e ".[export]" tensorrt-cu12 "albumentations>=1.4.6" comet pycocotools

# Test Install python dependencies
# RUN pip install ultralytics
# End test
# Run exports to AutoInstall packages
# Edge TPU export fails the first time so is run twice here
#RUN yolo export model=tmp/yolo11n.pt format=edgetpu imgsz=32 || yolo export model=tmp/yolo11n.pt format=edgetpu imgsz=32
#RUN yolo export model=tmp/yolo11n.pt format=ncnn imgsz=32
# Requires <= Python 3.10, bug with paddlepaddle==2.5.0 https://github.com/PaddlePaddle/X2Paddle/issues/991
RUN pip install "paddlepaddle>=2.6.0" x2paddle
# Fix error: `np.bool` was a deprecated alias for the builtin `bool` segmentation error in Tests
RUN pip install numpy==1.23.5

# Remove extra build files
RUN rm -rf tmp /root/.config/Ultralytics/persistent_cache.json

#FROM python:3.9.13
# Usage Examples -------------------------------------------------------------------------------------------------------

# Build and Push
# t=ultralytics/ultralytics:latest && sudo docker build -f docker/Dockerfile -t $t . && sudo docker push $t

# Pull and Run with access to all GPUs
# t=ultralytics/ultralytics:latest && sudo docker pull $t && sudo docker run -it --ipc=host --gpus all $t

# Pull and Run with access to GPUs 2 and 3 (inside container CUDA devices will appear as 0 and 1)
# t=ultralytics/ultralytics:latest && sudo docker pull $t && sudo docker run -it --ipc=host --gpus '"device=2,3"' $t

# Pull and Run with local directory access
# t=ultralytics/ultralytics:latest && sudo docker pull $t && sudo docker run -it --ipc=host --gpus all -v "$(pwd)"/shared/datasets:/datasets $t

# Kill all
# sudo docker kill $(sudo docker ps -q)

# Kill all image-based
# sudo docker kill $(sudo docker ps -qa --filter ancestor=ultralytics/ultralytics:latest)

# DockerHub tag update
# t=ultralytics/ultralytics:latest tnew=ultralytics/ultralytics:v6.2 && sudo docker pull $t && sudo docker tag $t $tnew && sudo docker push $tnew

# Clean up
# sudo docker system prune -a --volumes

# Update Ubuntu drivers
# https://www.maketecheasier.com/install-nvidia-drivers-ubuntu/

# DDP test
# python -m torch.distributed.run --nproc_per_node 2 --master_port 1 train.py --epochs 3

# GCP VM from Image
# docker.io/ultralytics/ultralytics:latest
