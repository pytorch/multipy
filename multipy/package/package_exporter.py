# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import BinaryIO, cast, Sequence, Union

import torch
from torch.serialization import location_tag, normalize_storage_type
from torch.types import Storage

from ._zip_file_torchscript import TorchScriptPackageZipFileWriter
from .importer import Importer, sys_importer
from .package_exporter_no_torch import PackageExporter as DefaultPackageExporter


# To deal with torch.storage._TypedStorage => torch.storage.TypedStorage renaming
try:
    TORCH_STORAGE_CLASS = torch.storage._TypedStorage
except:
    TORCH_STORAGE_CLASS = torch.storage.TypedStorage


# TODO: fix pytorch master to use PackagingError from the base class then delete below line
# from .package_exporter_no_torch import PackagingError, EmptyMatchError  # noqa


class PackageExporter(DefaultPackageExporter):
    """
    A package exporter for specialized functionality for torch. Specifically it uses the optimizations
    of torch's storage in order to not duplicate tensors.
    """

    def __init__(
        self,
        f: Union[str, Path, BinaryIO],
        importer: Union[Importer, Sequence[Importer]] = sys_importer,
    ):

        super(PackageExporter, self).__init__(
            f, importer, zip_file_writer_type=TorchScriptPackageZipFileWriter
        )
        # TODO: Delete these lines after we update pytorch to call directly from the zipfile
        self.script_module_serializer = self.zip_file.script_module_serializer
        self.storage_context = self.zip_file.storage_context

    def persistent_id(self, obj):
        assert isinstance(self.zip_file, TorchScriptPackageZipFileWriter)
        # needed for 'storage' typename which is a way in which torch models are saved
        if torch.is_storage(obj) or isinstance(obj, TORCH_STORAGE_CLASS):

            if isinstance(obj, TORCH_STORAGE_CLASS):
                # TODO: Once we decide to break serialization FC, we can
                # remove this case
                storage = obj._storage
                storage_type_str = obj.pickle_storage_type()
                storage_type = getattr(torch, storage_type_str)
                dtype = obj.dtype
                storage_numel = obj.size()

            else:
                storage = obj
                storage_type = normalize_storage_type(type(storage))
                dtype = torch.uint8
                storage_numel = storage.nbytes()

            storage = cast(Storage, storage)
            location = location_tag(storage)

            # serialize storage if not already written
            storage_present = self.zip_file.storage_context.has_storage(storage)
            storage_id = self.zip_file.storage_context.get_or_add_storage(storage)
            if not storage_present:
                if storage.device.type != "cpu":
                    storage = storage.cpu()
                num_bytes = storage.nbytes()
                self.zip_file.write_record(
                    f".data/{storage_id}.storage", storage.data_ptr(), num_bytes
                )
            return ("storage", storage_type, storage_id, location, storage_numel)
        return None
