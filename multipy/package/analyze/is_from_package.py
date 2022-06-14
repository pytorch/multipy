# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import ModuleType
from typing import Any

from .._mangling import is_mangled


def is_from_package(obj: Any) -> bool:
    """
    Return whether an object was loaded from a package.

    Note: packaged objects from externed modules will return ``False``.
    """
    if type(obj) == ModuleType:
        return is_mangled(obj.__name__)
    else:
        return is_mangled(type(obj).__module__)
