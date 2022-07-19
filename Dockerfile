ARG BASE_IMAGE=nvidia/cuda:11.3.1-devel-ubuntu18.04
ARG PYTHON_VERSION=3.8

FROM ${BASE_IMAGE} as dev-base


RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt \
        apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        cmake \
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
        tk-dev \
        libsqlite3-dev && \
    rm -rf /var/lib/apt/lists/*
RUN /usr/sbin/update-ccache-symlinks
RUN mkdir /opt/ccache && ccache --set-config=cache_dir=/opt/ccache
ENV PATH /opt/conda/bin:$PATH

FROM dev-base as submodule-update
WORKDIR /opt/multipy
COPY . .
RUN git submodule update --init --recursive --jobs 0

FROM dev-base as conda
ARG PYTHON_VERSION=3.8
COPY multipy/runtime/third-party/pytorch/requirements.txt .
RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} cmake mkl mkl-include conda-build pyyaml numpy ipython && \
    /opt/conda/bin/conda install -y -c pytorch magma-cuda113 && \
    /opt/conda/bin/python -mpip install -r requirements.txt && \
    /opt/conda/bin/conda clean -ya

FROM conda as build
WORKDIR /opt/multipy/multipy/runtime/third-party/pytorch
COPY --from=conda /opt/conda /opt/conda
COPY --from=submodule-update /opt/multipy /opt/multipy
ENV GLIBCXX_USE_CXX11_ABI=0
ENV CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"
ENV TORCH_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"

RUN --mount=type=cache,target=/opt/ccache \
    TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1 7.0+PTX 8.0" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    USE_CUDA="ON" \
    USE_CUDA_SPLIT="ON" \
    USE_DEPLOY="ON" \
    python setup.py develop

WORKDIR /opt/multipy
RUN mkdir multipy/runtime/build && \
    cd multipy/runtime/build && \
    cmake -DABI_EQUALS_1="OFF" .. && \
    cmake --build . --config Release && \
    cmake --install . --prefix "."

