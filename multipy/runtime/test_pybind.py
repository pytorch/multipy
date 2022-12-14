#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from multipy.runtime import InterpreterManager


class TestPybind(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manager = InterpreterManager(1)

    def test_print(self):
        I = self.manager.acquire_one()
        iprint = I.global_("builtins", "print")
        iprint("hello!")

    def test_tensor_passing(self):
        I = self.manager.acquire_one()
        model = I.global_("torch.nn", "Conv2d")(6, 2, 2, 1)
        out = model(torch.ones((1, 6, 6, 6)))
        tensor = out.deref()
        self.assertIsInstance(tensor, torch.Tensor)
        del out
        del model
        del I
        print("exit2")

    def test_multiple(self):
        m = InterpreterManager(2)
        self.assertEqual(len(m), 2)
        for i in range(len(m)):
            I = m[i]
            iprint = I.global_("builtins", "print")
            iprint(f"hello {i}")
            del iprint
            del I


if __name__ == "__main__":
    unittest.main()
