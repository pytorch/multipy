# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# this is imported by test_deploy to do some checks in python
import subprocess
import sys
from pathlib import Path

# we've taken steps to clear out the embedded python environment,
# so we have to go searching for real python to figure out where its libraries are installed.
def python_path(cpath):
    for maybe in cpath.split(":"):
        candidate = Path(maybe) / "python"
        if candidate.exists():
            cmd = [str(candidate), "-c", 'import sys; print(":".join(sys.path))']
            return subprocess.check_output(cmd).decode("utf-8").strip("\n").split(":")
    raise RuntimeError("could not find real python")


def setup(path):
    sys.path.extend(python_path(path))
    sys.path.append("build/lib")  # for our test python extension


# smoke test the numpy extension loading works
def numpy_test(x):
    import numpy as np

    xs = [np.array([x, x]), np.array([x, x])]
    for i in range(10):
        xs.append(xs[-1] + xs[-2])
    return int(xs[-1][0])
