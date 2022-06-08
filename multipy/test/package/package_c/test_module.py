# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Owner(s): ["oncall: package/deploy"]

import torch

try:
    from torchvision.models import resnet18

    class TorchVisionTest(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tvmod = resnet18()

        def forward(self, x):
            x = a_non_torch_leaf(x, x)
            return torch.relu(x + 3.0)

except ImportError:
    pass


def a_non_torch_leaf(a, b):
    return a + b
