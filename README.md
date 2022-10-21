[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)


# \[experimental\] MultiPy

> :warning: **This is project is still a prototype.** Only Linux x86 is supported, and the API may change without warning. Furthermore, please **USE PYTORCH NIGHTLY** when using `multipy::runtime`!

`MultiPy` (formerly `torch::deploy`) is a library that allows you to run multiple embedded Python interpreters in a C++ process without a shared global interpreter lock (GIL). Currently, MultiPy is specialized for PyTorch models as `torch::deploy`, however there are plans to expand its functionality to (mostly) arbitrary python code. It uses `torch.package` in order to package code into a mostly hermetic format to deliver to `torch::deploy`. For more information on how `torch::deploy` works
internally, please see the related [arXiv paper](https://arxiv.org/pdf/2104.00254.pdf).

## Installation
```

### Installing via `pip install`

The easiest way to install `torch::deploy` is to use `pip install` (which uses `python setup.py develop` under the hood).
There are many ways to build `torch::deploy` which are described in the setup documentation.
# [Add Link to Setup in docs]
To install on a debian based system using conda, one can do the following.

```shell
git clone https://github.com/pytorch/multipy.git
cd multipy
git submodule sync && git submodule update --init --recursive

sudo apt update
xargs sudo apt install -y -qq --no-install-recommends <build-requirements.txt

conda create -n newenv
conda activate newenv
conda install python=3.8
# we need the -fpic version of python in order to use torch::deploy
conda install -c conda-forge libpython-static=3.8
conda install pytorch torchvision torchaudio cpuonly -c pytorch-nightly

pip install -e .
```

## Tutorials

# [Add Link to Tutorials here]

## Contributing

We welcome PRs! See the [CONTRIBUTING](CONTRIBUTING.md) file.

## License

MultiPy is BSD licensed, as found in the [LICENSE](LICENSE) file.

## Legal

[Terms of Use](https://opensource.facebook.com/legal/terms)
[Privacy Policy](https://opensource.facebook.com/legal/privacy)

Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
