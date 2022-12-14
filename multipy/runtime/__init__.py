#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This contains a pybinded interface to MultiPy's subinterpreters. This enables
running child interpreters from an existing python process without having any
shared GIL.

WARNING: This is currently a prototype and is subject to change at any point.

Current limitations:
* Obj must be destroyed/GCed before the InterpreterSession is destroyed. If left
    up to GC in most cases it will crash the program.
* No pickle interface for smartly transferring models between interpreters

See test_pybind.py for examples on how to use.
"""

import ctypes
import os
import os.path

import torch

# We need to load libtorch into the global symbol space so the subinterpreters
# can use it.
torch_dir = os.path.dirname(torch.__file__)
libtorch_path = os.path.join(torch_dir, "lib", "libtorch.so")
ctypes.CDLL(libtorch_path, mode=ctypes.RTLD_GLOBAL)

from multipy.runtime.build.multipy_pybind import InterpreterManager  # noqa: F401
