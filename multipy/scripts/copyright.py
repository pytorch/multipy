#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
from typing import List

COPYRIGHT_NOTICE: str = """
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
""".strip()


def main(directories: List[str]) -> None:
    missing = []
    filelist = []
    for directory in directories:
        for path, dir, files in os.walk(directory):
            for file in files:
                if file.endswith(".py"):
                    filelist.append(path + os.sep + file)

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
        # try:
        inbuffer = open(filename, "U").readlines()
        outbuffer = [COPYRIGHT_NOTICE] + inbuffer

        open(filename, "w").write("".join([str(line) for line in outbuffer]))
        # except IOError:
        #     print(
        #         f"Please check the files, there was an error when trying to open {filename}..."
        #     )
        # except:
        #     print(f"Unexpected error ocurred while processing files...")

    sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
