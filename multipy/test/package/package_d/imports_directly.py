# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from .subpackage_0.subsubpackage_0 import important_string


class ImportsDirectlyFromSubSubPackage(torch.nn.Module):

    key = important_string

    def forward(self, inp):
        return torch.sum(inp)
