# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


def func(*vlist):
    return sum(vlist)


import sys

print("byebye!", file=sys.stderr)
