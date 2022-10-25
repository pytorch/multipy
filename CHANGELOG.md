# CHANGELOG

## multipy-0.1.0

This is the initial Beta release of `torch::deploy`.

* PyTorch 1.13 support
* Python 3.7-3.10 support
* `torch::deploy` is now suitable for use in production environments.
* `torch::deploy` uses the current Python environment and no longer
  requires building PyTorch, Python and C extensions from source.
* C extensions can be installed via standard `pip`/`conda` and will be
  dynamically loaded at runtime. Popular PyTorch extensions have been tested but
  there may be some libraries that are incompatible. If you run into an
  incompatible library please file an issue.
* Prototype aarch64 support
* Improved performance and memory usage when keeping an InterpreterSession alive
  for a long time.
* Supports all PyTorch core backends (CPU/CUDA/ROCm).
