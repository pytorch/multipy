ARG BASE_IMAGE=nvidia/cuda:11.3.1-devel-ubuntu18.04
# ARG BASE_IMAGE=pytorch/manylinux-cuda113
ARG PYTHON_VERSION=3.8

FROM ${BASE_IMAGE} as dev-base


RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt \
        apt update && DEBIAN_FRONTEND=noninteractive apt install -yq --no-install-recommends \
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
        libgtest-dev \
        tk-dev \
        libsqlite3-dev && \
    rm -rf /var/lib/apt/lists/*
RUN /usr/sbin/update-ccache-symlinks
RUN mkdir /opt/ccache && ccache --set-config=cache_dir=/opt/ccache
ENV PATH /opt/conda/bin:$PATH
RUN cd /usr/src/gtest && cmake CMakeLists.txt && make && cp *.a /usr/lib

FROM dev-base as submodule-update
WORKDIR /opt/multipy
COPY . .
# RUN git submodule update --init --recursive --jobs 0

FROM dev-base as conda
ARG PYTHON_VERSION=3.8
COPY multipy/runtime/third-party/pytorch/requirements.txt .
RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} cmake mkl mkl-include conda-build pyyaml numpy ipython && \
    /opt/conda/bin/conda install -y -c pytorch magma-cuda113 && \
    /opt/conda/bin/conda install -y -c pytorch cudatoolkit=11.3 && \
    /opt/conda/bin/python -mpip install -r requirements.txt && \
    /opt/conda/bin/conda clean -ya

FROM conda as build
WORKDIR /opt/multipy/multipy/runtime/third-party/pytorch
COPY --from=conda /opt/conda /opt/conda
COPY --from=submodule-update /opt/multipy /opt/multipy
ENV _GLIBCXX_USE_CXX11_ABI=1
RUN --mount=type=cache,target=/opt/ccache \
    TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1 7.0+PTX 8.0" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    USE_CUDA="ON" \
    BUILD_SPLIT_CUDA="ON" \
    BUILD_TEST=1 \
    BUILD_CAFFE2=0 \
    USE_DEPLOY=1 \
    python setup.py install

FROM build as conda-installs
ARG PYTHON_VERSION=3.8
ARG CUDA_VERSION=11.3
ARG CUDA_CHANNEL=nvidia
ARG INSTALL_CHANNEL=pytorch
ENV CONDA_OVERRIDE_CUDA=${CUDA_VERSION}
RUN /opt/conda/bin/conda install -c "${INSTALL_CHANNEL}" -c "${CUDA_CHANNEL}" -y python=${PYTHON_VERSION} pytorch torchvision torchtext "cudatoolkit=${CUDA_VERSION}" && \
    /opt/conda/bin/conda clean -ya
RUN /opt/conda/bin/pip install torchelastic

# FROM conda-installs as multipy-build
WORKDIR /opt/multipy

RUN mkdir multipy/runtime/build && \
   cd multipy/runtime/build && \
   cmake -DABI_EQUALS_1="ON" .. && \
   cmake --build . --config Release && \
   cmake --install . --prefix "."

RUN mkdir /opt/dist && cp -r multipy/runtime/build/dist /opt/dist/
FROM ${BASE_IMAGE} as official
ARG PYTORCH_VERSION 
LABEL com.nvidia.volumes.needed="nvidia_driver"
RUN --mount=type=cache,id=apt-final,target=/var/cache/apt \
   apt-get update && apt-get install -y --no-install-recommends \
       ca-certificates \
       libjpeg-dev \
       libpng-dev && \
   rm -rf /var/lib/apt/lists/*

COPY --from=conda-installs /opt/conda /opt/conda
COPY --from=build /opt/multipy/multipy/runtime/build/dist /opt/multipy
ENV PATH /opt/conda/bin:$PATH
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV PYTORCH_VERSION ${PYTORCH_VERSION}
WORKDIR /workspace

FROM official as dev
# Should override the already installed version from the official-image stage
COPY --from=build /opt/conda /opt/conda
COPY --from=build /opt/dist /opt/multipy
