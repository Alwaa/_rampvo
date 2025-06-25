FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# Install wget to fetch Miniconda
RUN apt-get update && \
    apt-get install -y wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN chmod +x /uv-installer.sh && /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

RUN uv venv --python 3.12
RUN source /.venv/bin/activate && uv pip install torch numpy torchvision

RUN echo 'import torch; print(torch.cuda.is_available())' > check_cuda.py

RUN /.venv/bin/python check_cuda.py

CMD ["sleep", "infinity"] 