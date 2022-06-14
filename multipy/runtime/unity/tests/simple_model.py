# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, X):
        X = self.fc(X)
        X = torch.relu(X)
        X = self.fc2(X)
        X = torch.softmax(X, dim=-1)
        return X
