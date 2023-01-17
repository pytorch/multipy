# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch._dynamo


class TestCompat(unittest.TestCase):
    def test_torchvision(self):
        import torchvision  # noqa: F401

    def test_torchaudio(self):
        import torchaudio  # noqa: F401

    def test_pytorch3d(self):
        import pytorch3d  # noqa: F401

    def test_hf_tokenizers(self):
        import tokenizers  # noqa: F401

    def test_torchdynamo_eager(self):

        torch._dynamo.reset()

        def fn(x, y):
            a = torch.cos(x)
            b = torch.sin(y)
            return a + b

        c_fn = torch.compile(fn, backend="eager")
        c_fn(torch.randn(10), torch.randn(10))

    def test_torchdynamo_ofi(self):

        torch._dynamo.reset()

        def fn(x, y):
            a = torch.cos(x)
            b = torch.sin(y)
            return a + b

        c_fn = torch.compile(fn, backend="ofi")
        c_fn(torch.randn(10), torch.randn(10))

    def test_torchdynamo_inductor(self):

        torch._dynamo.reset()

        def fn(x, y):
            a = torch.cos(x)
            b = torch.sin(y)
            return a + b

        c_fn = torch.compile(fn)
        c_fn(torch.randn(10), torch.randn(10))


if __name__ == "__main__":
    unittest.main()
