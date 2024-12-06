#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
from typing import List, Tuple

COPYRIGHT_NOTICE_PYTHON: str = """
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
""".strip()

COPYRIGHT_NOTICE_C: str = """
// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
""".strip()


def get_all_files_with_extension(directories: List[str], extensions: Tuple[str]):
    filelist = []
    for directory in directories:
        for path, _, files in os.walk(directory):
            for file in files:
                if file.endswith(extensions):
                    filelist.append(path + os.sep + file)
    return filelist


def apply_copywrite(
    directories: List[str], COPYRIGHT_NOTICE: str, extensions: Tuple[str]
) -> None:
    missing = []
    filelist = get_all_files_with_extension(directories, extensions)

    for path in filelist:
        with open(path, "rt") as f:
            data = f.read()

        data = data.strip()
        # don't lint empty files
        if len(data) == 0:
            continue

        # skip the interpreter command and formatting
        while data.startswith("#!") or data.startswith("# -*-"):
            data = data[data.index("\n") + 1 :]

        if not data.startswith(COPYRIGHT_NOTICE):
            missing.append(path)

    for filename in missing:
        try:
            inbuffer = open(filename, "U").readlines()
            outbuffer = [COPYRIGHT_NOTICE] + ["\n\n"] + inbuffer

            open(filename, "w").write("".join([str(line) for line in outbuffer]))
        except IOError:
            print(
                f"Please check the files, there was an error when trying to open {filename}..."
            )
            raise


if __name__ == "__main__":
    files = sys.argv[1:]
    apply_copywrite(files, COPYRIGHT_NOTICE_PYTHON, (".py"))
    apply_copywrite(files, COPYRIGHT_NOTICE_C, (".c", ".cpp", ".hpp", ".h"))
