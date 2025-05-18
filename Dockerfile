# Dockerfile
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

# Install wget to fetch Miniconda
RUN apt-get update && \
    apt-get install -y wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda on x86 or ARM platforms
RUN arch=$(uname -m) && \
    if [ "$arch" = "x86_64" ]; then \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
    elif [ "$arch" = "aarch64" ]; then \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"; \
    else \
    echo "Unsupported architecture: $arch"; \
    exit 1; \
    fi && \
    wget $MINICONDA_URL -O miniconda.sh && \
    mkdir -p /root/.conda && \
    bash miniconda.sh -b -p /root/miniconda3 && \
    rm -f miniconda.sh

ENV TORCH_CUDA_ARCH_LIST="7.5+PTX"

RUN conda create -n test_gpu -y \
    python==3.10 pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# Test the gpu works in docker
RUN conda run -n test_gpu \
    echo 'import torch; print(torch.cuda.is_available())' > check_cuda.py

############################################################################
WORKDIR /data/user

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libgl1 \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender1 && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends \
        build-essential cmake git pkg-config && \
      apt-get install -y libeigen3-dev libboost-all-dev \
                   libceres-dev libopencv-dev libgflags-dev libgoogle-glog-dev && \
    rm -rf /var/lib/apt/lists/*


RUN conda create -n ramp_vio -y python==3.10 \
    pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
RUN conda run -n ramp_vio pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.1+cu121.html
# For development
RUN conda run -n ramp_vio pip install ruff 

WORKDIR /data/user/temp_rampvo
COPY . . 
RUN conda run -n ramp_vio pip install -r requirements.txt
RUN conda run -n ramp_vio pip install .
WORKDIR /data/user
RUN rm -r temp_rampvo

# For fixing evo plot
COPY docker_sitecustomize.py /root/miniconda3/envs/ramp_vio/lib/python3.10/site-packages/sitecustomize.py

CMD ["sleep", "infinity"] 

