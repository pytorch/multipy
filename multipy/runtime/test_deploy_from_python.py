# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import test_deploy_python_ext
from libfb.py import testutil


class TestDeployFromPython(testutil.BaseFacebookTestCase):
    def test_deploy_from_python(self):
        self.assertTrue(test_deploy_python_ext.run())
