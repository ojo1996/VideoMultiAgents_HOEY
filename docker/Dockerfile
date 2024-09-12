FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04

ENV http_proxy="http://proxy.mei.co.jp:8080/"
ENV https_proxy="http://proxy.mei.co.jp:8080/"

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
# debian setting
ENV DEBIAN_FRONTEND=noninteractive \
    DEBIAN_FRONTEND=teletype \
    DEBCONF_NOWARNINGS=yes
# python pyc setting
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Common Package
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-transport-https \
    apt-utils \
    build-essential \
    curl \
    git \
    jq \
    libgl1-mesa-dev \
    libopencv-dev \
    python3-dev \
    python3-pip \
    python3-setuptools \
    sudo \
    wget \
    zip \
    && apt-get -y clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install torch==2.4.0
RUN curl -fsSL https://ollama.com/install.sh | sh

RUN pip3 install -U pip
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
WORKDIR /root

# Set up the Bash shell environment
RUN echo "export PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '" >> /root/.bashrc

ENTRYPOINT []

CMD ["/bin/bash"]