# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch


class TestCompat(unittest.TestCase):
    def test_torchvision(self):
        import torchvision  # noqa: F401

    def test_torchaudio(self):
        import torchaudio  # noqa: F401

    def test_pytorch3d(self):
        import pytorch3d  # noqa: F401

    def test_hf_tokenizers(self):
        import tokenizers  # noqa: F401

    @unittest.skip("torch.Library is not supported")
    def test_torchdynamo_eager(self):
        import torchdynamo

        @torchdynamo.optimize("eager")
        def fn(x, y):
            a = torch.cos(x)
            b = torch.sin(y)
            return a + b

        fn(torch.randn(10), torch.randn(10))

    @unittest.skip("torch.Library is not supported")
    def test_torchdynamo_ofi(self):
        import torchdynamo

        torchdynamo.reset()

        @torchdynamo.optimize("ofi")
        def fn(x, y):
            a = torch.cos(x)
            b = torch.sin(y)
            return a + b

        fn(torch.randn(10), torch.randn(10))


if __name__ == "__main__":
    unittest.main()
