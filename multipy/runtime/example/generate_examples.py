# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Generate the example files that torchpy_test uses.
"""
import argparse
from pathlib import Path

import torch
from torch.fx import symbolic_trace
from torch.package import PackageExporter

try:
    from .examples import (
        BatchedModel,
        load_library,
        multi_return_metadata,
        MultiReturn,
        resnet18,
        Simple,
    )
except ImportError:
    from examples import (
        BatchedModel,
        load_library,
        multi_return_metadata,
        MultiReturn,
        resnet18,
        Simple,
    )

try:
    from .fx.examples import SimpleWithLeaf
except ImportError:
    from fx.examples import SimpleWithLeaf

try:
    from .tensorrt_example import make_trt_module
except ImportError:
    from tensorrt_example import make_trt_module


def generate_fx_example():
    name = "simple_leaf"
    model = SimpleWithLeaf(5, 10)
    graph_module: torch.fx.GraphModule = symbolic_trace(model)
    with PackageExporter(str(p / (name + "_fx"))) as e:
        e.intern("**")
        e.save_pickle("model", "model.pkl", graph_module)

    model_jit = torch.jit.script(model)
    model_jit.save(str(p / (name + "_jit")))


def save(
    name,
    model,
    model_jit=None,
    eg=None,
    featurestore_meta=None,
    text_in_extra_file=None,
    binary_in_extra_file=None,
):
    with PackageExporter(str(p / name)) as e:
        e.mock("iopath.**")
        e.intern("**")
        e.save_pickle("model", "model.pkl", model)
        if eg:
            e.save_pickle("model", "example.pkl", eg)
        if featurestore_meta:
            # TODO(whc) can this name come from buck somehow,
            # so it's consistent with predictor_config_constants::METADATA_FILE_NAME()?
            e.save_text("extra_files", "metadata.json", featurestore_meta)
        if text_in_extra_file:
            e.save_text("extra_files", "text", text_in_extra_file)
        if binary_in_extra_file:
            e.save_binary("extra_files", "binary", binary_in_extra_file)

    if model_jit:
        model_jit.save(str(p / (name + "_jit")))


parser = argparse.ArgumentParser(description="Generate Examples")
parser.add_argument("--install_dir", help="Root directory for all output files")


if __name__ == "__main__":
    args = parser.parse_args()
    if args.install_dir is None:
        p = Path(__file__).parent / "generated"
        p.mkdir(exist_ok=True)
    else:
        p = Path(args.install_dir)

    resnet = resnet18()
    resnet.eval()
    resnet_eg = torch.rand(1, 3, 224, 224)
    resnet_traced = torch.jit.trace(resnet, resnet_eg)
    save("resnet", resnet, resnet_traced, (resnet_eg,))

    simple = Simple(10, 20)
    save(
        name="simple",
        model=simple,
        model_jit=torch.jit.script(simple),
        eg=(torch.rand(10, 20),),
        text_in_extra_file="hello",
        binary_in_extra_file=b"hello",
    )

    multi_return = MultiReturn()
    save(
        "multi_return",
        multi_return,
        torch.jit.script(multi_return),
        (torch.rand(10, 20),),
        multi_return_metadata,
    )

    # used for torch deploy/package tests in predictor
    batched_model = BatchedModel()
    save("batched_model", batched_model)

    with PackageExporter(str(p / "load_library")) as e:
        e.mock("iopath.**")
        e.intern("**")
        e.save_pickle("fn", "fn.pkl", load_library)

    generate_fx_example()

    with PackageExporter(p / "uses_distributed") as e:
        e.save_source_string(
            "uses_distributed",
            "import torch.distributed; assert torch.distributed.is_available()",
        )

    with PackageExporter(p / "uses_cuda") as e:
        e.save_source_string(
            "uses_cuda",
            "import torch; assert torch.cuda.is_available()",
        )

    with PackageExporter(str(p / "make_trt_module")) as e:
        e.extern("tensorrt")
        e.add_dependency("tensorrt")
        e.mock("iopath.**")
        e.intern("**")
        e.save_pickle("make_trt_module", "model.pkl", make_trt_module)
