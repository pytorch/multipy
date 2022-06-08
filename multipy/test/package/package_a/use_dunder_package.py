# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

if "__torch_package__" in dir():

    def is_from_package():
        return True

else:

    def is_from_package():
        return False
