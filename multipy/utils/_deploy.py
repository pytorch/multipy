# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io

import torch

import torch.package
from torch.package import Importer, OrderedImporter, PackageImporter, sys_importer
from torch.package._package_pickler import create_pickler
from torch.package._package_unpickler import PackageUnpickler
from torch.serialization import _maybe_decode_ascii

# For < pytorch 1.13 compatibility. We can likely delete this after it's release.
try:
    STORAGE_TYPE = torch.storage._TypedStorage
except:
    STORAGE_TYPE = torch.storage.TypedStorage


def _save_storages(importer, obj):
    serialized_storages = []
    serialized_dtypes = []

    importer = importer if isinstance(importer, torch.package.PackageImporter) else None
    importers: Importer
    if importer is not None:
        importers = OrderedImporter(importer, sys_importer)
    else:
        importers = sys_importer

    def persistent_id(obj):
        if torch.is_storage(obj) or isinstance(obj, STORAGE_TYPE):
            if isinstance(obj, STORAGE_TYPE):
                # TODO: Once we decide to break serialization FC, we can
                # remove this case
                storage = obj._storage
                dtype = obj.dtype
            else:
                storage = obj
                dtype = torch.uint8

            serialized_storages.append(obj)
            serialized_dtypes.append(dtype)
            return ("storage", len(serialized_storages) - 1)

        if hasattr(obj, "__reduce_deploy__"):
            if _serialized_reduces.get(id(obj)) is None:
                _serialized_reduces[id(obj)] = (
                    "reduce_deploy",
                    id(obj),
                    *obj.__reduce_deploy__(importers),
                )
            return _serialized_reduces[id(obj)]

        return None

    # Write the pickle data for `obj`
    data_buf = io.BytesIO()
    pickler = create_pickler(data_buf, importers)
    pickler.persistent_id = persistent_id
    pickler.dump(obj)
    data_value = data_buf.getvalue()
    return (
        data_value,
        serialized_storages,
        serialized_dtypes,
        importer.zip_reader if importer else None,
    )


def _load_storages(id, zip_reader, obj_bytes, serialized_storages, serialized_dtypes):
    def persistent_load(saved_id):
        assert isinstance(saved_id, tuple)
        typename = _maybe_decode_ascii(saved_id[0])
        data = saved_id[1:]

        if typename == "storage":
            # TODO: Once we decide to break serialization FC, we can
            # stop wrapping with _TypedStorage
            storage = serialized_storages[data[0]]
            dtype = serialized_dtypes[data[0]]

            # For < pytorch 1.13 compatibility. We can likely remove after it's release.
            try:
                storage_untyped = storage.untyped
            except:
                storage_untyped = storage._untyped

            return STORAGE_TYPE(wrap_storage=storage_untyped(), dtype=dtype)

        if typename == "reduce_deploy":
            reduce_id, func, args = data
            if reduce_id not in _loaded_reduces:
                _loaded_reduces[reduce_id] = func(_raw_packages[zip_reader], *args)
            return _loaded_reduces[reduce_id]

        return None

    importer: Importer
    if zip_reader is not None:
        importer = OrderedImporter(_get_package(zip_reader), sys_importer)
    else:
        importer = sys_importer

    unpickler = PackageUnpickler(importer, io.BytesIO(obj_bytes))
    unpickler.persistent_load = persistent_load  # type: ignore[assignment]
    result = _deploy_objects[id] = unpickler.load()
    return result


def _get_package(zip_reader):
    if zip_reader not in _raw_packages:
        _raw_packages[zip_reader] = PackageImporter(zip_reader)
    return _raw_packages[zip_reader]


_raw_packages: dict = {}
_deploy_objects: dict = {}
_serialized_reduces: dict = {}
_loaded_reduces: dict = {}
