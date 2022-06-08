# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.fx

try:
    from .some_dependency import a_non_torch_leaf
except ImportError:
    from some_dependency import a_non_torch_leaf


torch.fx.wrap("a_non_torch_leaf")


class SimpleWithLeaf(torch.nn.Module):
    def __init__(self, N, M):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.rand(N, M))

    def forward(self, input):
        output = self.weight + a_non_torch_leaf(1, input)
        return output
