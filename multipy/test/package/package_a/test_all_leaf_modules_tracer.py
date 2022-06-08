# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Owner(s): ["oncall: package/deploy"]

from torch.fx import Tracer


class TestAllLeafModulesTracer(Tracer):
    def is_leaf_module(self, m, qualname):
        return True
