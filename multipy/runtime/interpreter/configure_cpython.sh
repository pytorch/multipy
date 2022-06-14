#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex
wget https://www.openssl.org/source/openssl-1.1.1k.tar.gz
tar xf openssl-1.1.1k.tar.gz
(cd openssl-1.1.1k && ./config --prefix="$PYTHON_INSTALL_DIR" && make -j32 && make install)
CFLAGS=-fPIC CPPFLAGS=-fPIC ./configure --prefix "$PYTHON_INSTALL_DIR" --with-openssl="$PYTHON_INSTALL_DIR"
