FROM ubuntu:22.04

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

RUN pip3 install -U pip
COPY azure-batch/docker/requirements.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR /root
RUN mkdir -p /root/VideoMultiAgents/

COPY azure-batch/main.py /root/VideoMultiAgents/
COPY azure-batch/util_azure.py /root/VideoMultiAgents/
COPY stage1.py /root/VideoMultiAgents/
COPY stage2.py /root/VideoMultiAgents/
COPY util.py /root/VideoMultiAgents/
COPY tools /root/VideoMultiAgents/tools/

# Set up the Bash shell environment
RUN echo "export PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '" >> /root/.bashrc

ENTRYPOINT []

CMD ["/bin/bash"]