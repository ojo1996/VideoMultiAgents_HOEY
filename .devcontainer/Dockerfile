FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git git-lfs build-essential curl ca-certificates python3-venv \
 && git lfs install \
 && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers \
    PATH="/opt/conda/bin:$PATH"

RUN pip install --upgrade pip wheel setuptools \
 && pip install "huggingface_hub[cli]>=0.23" accelerate transformers datasets \
    peft bitsandbytes einops

WORKDIR /workspace
