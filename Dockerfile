ARG BASE_IMAGE=nvidia/cuda:11.6.1-devel-ubuntu18.04

FROM ${BASE_IMAGE} as dev-base

SHELL ["/bin/bash", "-c"]

# Install system dependencies
RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt \
        apt update && DEBIAN_FRONTEND=noninteractive apt install -yq --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        curl \
        cmake-mozilla \
        wget \
        git \
        libjpeg-dev \
        xz-utils \
        bzip2 \
        libbz2-dev \
        liblzma-dev \
        libreadline6-dev \
        libexpat1-dev \
        libgdbm-dev \
        glibc-source \
        libgmp-dev \
        libffi-dev \
        libgl-dev \
        ncurses-dev \
        libncursesw5-dev \
        libncurses5-dev \
        gnome-panel \
        libssl-dev \
        tcl-dev \
        tix-dev \
        libgtest-dev \
        tk-dev \
        libsqlite3-dev \
        zlib1g-dev \
        llvm \
        python-openssl \
        apt-transport-https \
        ca-certificates \
        gnupg \
        software-properties-common \
        python-pip \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*
RUN /usr/sbin/update-ccache-symlinks
RUN mkdir /opt/ccache && ccache --set-config=cache_dir=/opt/ccache
ENV PATH /opt/conda/bin:$PATH

# get the repo
FROM dev-base as submodule-update
WORKDIR /opt/multipy

# add files incrementally
COPY .git .git
COPY .gitmodules .gitmodules
COPY multipy multipy
COPY compat-requirements.txt compat-requirements.txt
COPY setup.py setup.py
COPY README.md README.md
COPY dev-requirements.txt dev-requirements.txt

RUN git -c fetch.parallel=0 -c submodule.fetchJobs=0  submodule update --init --recursive

# Install conda/pyenv + necessary python dependencies
FROM dev-base as conda-pyenv
ARG PYTHON_3_MINOR_VERSION=8
ARG BUILD_CUDA_TESTS=0
ENV PYTHON_3_MINOR_VERSION=${PYTHON_3_MINOR_VERSION}
ENV PYTHON_VERSION=3.${PYTHON_3_MINOR_VERSION}
RUN if [[ ${PYTHON_3_MINOR_VERSION} -gt 7 ]]; then \
    curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} mkl mkl-include conda-build pyyaml numpy ipython && \
    /opt/conda/bin/conda install -y -c conda-forge libpython-static=${PYTHON_VERSION} && \
    /opt/conda/bin/conda install -y pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch-nightly -c nvidia && \
    /opt/conda/bin/conda clean -ya; \
    else \
    pip3 install virtualenv && \
    git clone https://github.com/pyenv/pyenv.git ~/.pyenv && \
    export CFLAGS="-fPIC -g" && \
    ~/.pyenv/bin/pyenv install --force 3.7.10 && \
    virtualenv -p ~/.pyenv/versions/3.7.10/bin/python3 ~/venvs/multipy && \
    source ~/venvs/multipy/bin/activate && \
    pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu116; \
    fi

FROM conda-pyenv as build
COPY --from=conda-pyenv /opt/conda* /opt/conda
COPY --from=submodule-update /opt/multipy /opt/multipy

WORKDIR /opt/multipy

# Build Multipy
RUN ls && pwd && rm -rf multipy/runtime/build && \
    if [[ ${PYTHON_3_MINOR_VERSION} -lt 8 ]]; then \
    source ~/venvs/multipy/bin/activate; \
    fi && \
    if [[ ${BUILD_CUDA_TESTS} -eq 1 ]]; then \
    BUILD_CUDA_TESTS=1 python -m pip install -e .; \
    else \
    python -m pip install -e .; \
    fi && \
    python multipy/runtime/example/generate_examples.py

# Build examples
COPY examples examples
RUN cd examples && \
    rm -r build; \
    if [[ ${PYTHON_3_MINOR_VERSION} -lt 8 ]]; then \
    source ~/venvs/multipy/bin/activate; \
    else \
    source /opt/conda/bin/activate; \
    fi && \
    cmake -S . -B build/ -DMULTIPY_PATH=".." && \
    cmake --build build/ --config Release -j

ENV PYTHONPATH=. LIBTEST_DEPLOY_LIB=multipy/runtime/build/libtest_deploy_lib.so


RUN mkdir /opt/dist && cp -r multipy/runtime/build/dist/* /opt/dist/
