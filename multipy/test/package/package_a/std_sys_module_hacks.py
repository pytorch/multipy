# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os  # noqa: F401
import os.path  # noqa: F401
import typing  # noqa: F401
import typing.io  # noqa: F401
import typing.re  # noqa: F401

import torch


class Module(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        return os.path.abspath("test")
