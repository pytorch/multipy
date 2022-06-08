# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__import__("math", fromlist=[])
__import__("xml.sax.xmlreader")

result = "subpackage_2"


class PackageBSubpackage2Object_0:
    pass


def dynamic_import_test(name: str):
    __import__(name)
