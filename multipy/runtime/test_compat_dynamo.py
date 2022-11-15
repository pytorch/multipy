# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch


class TestCompat(unittest.TestCase):

    def test_torchdynamo(self):
        resnet_path = "multipy/runtime/example/generated/resnet";
        resnet_jit_path = "multipy/runtime/example/generated/resnet_jit";
        importer_dynamo = torch.package.PackageImporter(resnet_path)
        model_dynamo = importer_dynamo.load_pickle("model", "model.pkl")
        eg_dynamo = importer_dynamo.load_pickle("model", "example.pkl")
        model_jit = torch.jit.load(resnet_jit_path)
        model_dynamo.forward(eg_dynamo[0])
        print(model_dynamo(eg_dynamo[0]), model_jit(eg_dynamo[0]))
        self.assertTrue(
            torch.allclose(model_dynamo(eg_dynamo[0]), model_jit(eg_dynamo[0]))
        )

if __name__ == "__main__":
    unittest.main()
