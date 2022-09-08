ARG BASE_IMAGE=nvidia/cuda:11.3.1-devel-ubuntu18.04

FROM ${BASE_IMAGE} as dev-base

SHELL ["/bin/bash", "-c"]

# Install system dependencies
RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt \
        apt update && DEBIAN_FRONTEND=noninteractive apt install -yq --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        curl \
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
        python3-pip && \
        wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add - && \
        apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main' && \
        echo "deb http://security.ubuntu.com/ubuntu focal-security main" >> /etc/apt/sources.list && \
        apt update && \
        apt install -y binutils cmake && \
    rm -rf /var/lib/apt/lists/*
RUN /usr/sbin/update-ccache-symlinks
RUN mkdir /opt/ccache && ccache --set-config=cache_dir=/opt/ccache
ENV PATH /opt/conda/bin:$PATH

# get the repo
FROM dev-base as submodule-update
WORKDIR /opt/multipy
COPY . .
RUN git submodule update --init --recursive --jobs 0

# Install conda/pyenv + necessary python dependencies
FROM dev-base as conda-pyenv
ARG PYTHON_MAJOR_VERSION=3
ARG PYTHON_MINOR_VERSION=8
ENV PYTHON_MINOR_VERSION=${PYTHON_MINOR_VERSION}
ENV PYTHON_VERSION=${PYTHON_MAJOR_VERSION}.${PYTHON_MINOR_VERSION}
RUN if [[ ${PYTHON_MINOR_VERSION} -gt 7 ]]; then \
    curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} mkl mkl-include conda-build pyyaml numpy ipython && \
    /opt/conda/bin/conda install -y -c conda-forge libpython-static=${PYTHON_VERSION} && \
    /opt/conda/bin/conda install -y pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch-nightly && \
    /opt/conda/bin/conda clean -ya; \
    else \
    pip3 install virtualenv && \
    git clone https://github.com/pyenv/pyenv.git ~/.pyenv && \
    ~/.pyenv/bin/pyenv install --force 3.7.10 && \
    virtualenv -p ~/.pyenv/versions/3.7.10/bin/python3 ~/venvs/multipy_3_7_10; \
    fi

# Build/Install pytorch with post-cxx11 ABI
FROM conda-pyenv as build
WORKDIR /opt/multipy/multipy/runtime/third-party/pytorch
COPY --from=conda-pyenv /opt/conda* /opt/conda
COPY --from=submodule-update /opt/multipy /opt/multipy

WORKDIR /opt/multipy

# Build Multipy
RUN if [[ ${PYTHON_MINOR_VERSION} -lt 8 ]]; then \
    source ~/venvs/multipy_3_7_10/bin/activate && \
    pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu113 && \
    export CFLAGS="-fPIC -g" && \
    export LEGACY_PYTHON_PRE_3_8=1; \
    else \
    export LEGACY_PYTHON_PRE_3_8=0; \
    fi && \
    mkdir multipy/runtime/build && \
    cd multipy/runtime/build && \
    cmake .. -DLEGACY_PYTHON_PRE_3_8=${LEGACY_PYTHON_PRE_3_8} && \
    cmake --build . --config Release && \
    cmake --install . --prefix "."

RUN cd multipy/runtime/example && python generate_examples.py
ENV PYTHONPATH=. LIBTEST_DEPLOY_LIB=multipy/runtime/build/libtest_deploy_lib.so

RUN mkdir /opt/dist && cp -r multipy/runtime/build/dist/* /opt/dist/
