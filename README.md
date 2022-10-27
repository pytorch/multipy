[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE) ![Runtime Tests](https://github.com/pytorch/multipy/actions/workflows/runtime_tests.yaml/badge.svg)


# `torch::deploy` (MultiPy)

`torch::deploy` (MultiPy for non-PyTorch use cases) is a C++ library that enables you to run eager mode PyTorch models in production without any modifications to your model to support tracing. `torch::deploy` provides a way to run using multiple independent Python interpreters in a single process without a shared global interpreter lock (GIL). For more information on how `torch::deploy` works
internally, please see the related [arXiv paper](https://arxiv.org/pdf/2104.00254.pdf).

To learn how to use `torch::deploy` see [Installation](#installation) and [Examples](#examples).

Requirements:

* PyTorch 1.13+ or PyTorch nightly
* Linux (ELF based)
  * x86_64 (Beta)
  * arm64/aarch64 (Prototype)

> ℹ️ This is project is in Beta. `torch::deploy` is ready for use in production environments but may have some rough edges that we're continuously working on improving. We're always interested in hearing feedback and usecases that you might have. Feel free to reach out!

## The Easy Path to Installation

### Building via Docker

The easiest way to build deploy and install the interpreter dependencies is to do so via docker.

```shell
git clone --recurse-submodules https://github.com/pytorch/multipy.git
cd multipy
export DOCKER_BUILDKIT=1
docker build -t multipy .
```

The built artifacts are located in `multipy/runtime/build`.

To run the tests:

```shell
docker run --rm multipy multipy/runtime/build/test_deploy
```

### Installing via `pip install`

The second easiest way of using `torch::deploy` is through our single command `pip install`.
However, the C++ dependencies have to manually be installed before hand. Specifically a `-fpic`
enabled version of python. For full instructions for getting the C++ dependencies up and
running and more detailed guide on `torch::deploy` installation can be found [here](https://pytorch.org/multipy/latest/setup.html#installing-via-pip-install).

Once all the dependencies are successfully installed, you can run the following, in either `conda` or `virtualenv`, to install both the python modules and the runtime/interpreter libraries:
```shell
# from base multipy directory
pip install -e .
```
The C++ binaries should be available in `/opt/dist`.

Alternatively, one can install only the python modules without invoking `cmake` as follows:
```shell
pip install  -e . --install-option="--cmakeoff"
```

## Getting Started with `torch::deploy`
Once you have `torch::deploy` built, check out our [tutorials](https://pytorch.org/multipy/latest/tutorials/tutorial_root.html) and
[API documentation](https://pytorch.org/multipy/latest/api/library_root.html).

## Contributing

We welcome PRs! See the [CONTRIBUTING](CONTRIBUTING.md) file.

## License

MultiPy is BSD licensed, as found in the [LICENSE](LICENSE) file.

## Legal

[Terms of Use](https://opensource.facebook.com/legal/terms)
[Privacy Policy](https://opensource.facebook.com/legal/privacy)

Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
