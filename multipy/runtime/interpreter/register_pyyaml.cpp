// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <Python.h>
#include <multipy/runtime/interpreter/builtin_registry.h>

extern "C" struct _frozen _PyImport_FrozenModules_pyyaml[];

REGISTER_TORCH_DEPLOY_BUILTIN(frozen_pyyaml, _PyImport_FrozenModules_pyyaml);
