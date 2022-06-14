# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__import__("subpackage_1", globals(), fromlist=["PackageBSubpackage1Object_0"], level=1)
__import__("subpackage_0.subsubpackage_0", globals(), fromlist=[""], level=1)
__import__("subpackage_2", globals=globals(), locals=locals(), fromlist=["*"], level=1)

result = "package_b"


class PackageBObject:
    __slots__ = ["obj"]

    def __init__(self, obj):
        self.obj = obj

    def return_result(self):
        return result
