#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
import subprocess
import sys
from datetime import date

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


class MultipyRuntimeExtension(Extension):
    def __init__(self, name):
        Extension.__init__(self, name, sources=[])


def get_cmake_version():
    output = subprocess.check_output(["cmake", "--version"]).decode("utf-8")
    return output.splitlines()[0].split()[2]


class MultipyRuntimeBuild(build_ext):
    def run(self):
        CMAKE_OFF_ARG = "--cmake-off"
        ABI_CXX_ARG = "--abi-cxx"
        abi_cxx_flag = "OFF"
        if CMAKE_OFF_ARG in sys.argv:
            sys.argv.pop(sys.argv.index(CMAKE_OFF_ARG))
            return
        if ABI_CXX_ARG in sys.argv:
           sys.argv.pop(sys.argv.index(CMAKE_OFF_ARG))
           abi_cxx_flag="ON"

        try:
            cmake_version_comps = get_cmake_version().split(".")
            if cmake_version_comps[0] < "3" or cmake_version_comps[1] < "19":
                raise RuntimeError(
                    "CMake 3.19 or later required for multipy runtime installation."
                )
        except OSError:
            raise RuntimeError(
                "Error fetching cmake version. Please ensure cmake is installed correctly."
            ) from None
        base_dir = os.path.abspath(os.path.dirname(__file__))
        build_dir = "multipy/runtime/build3"
        build_dir_abs = base_dir + "/" + build_dir
        if not os.path.exists(build_dir_abs):
            os.makedirs(build_dir_abs)
        legacy_python_cmake_flag = "OFF" if sys.version_info.minor > 7 else "ON"

        print(f"-- Running multipy runtime makefile in dir {build_dir_abs}")
        try:
            subprocess.run(
                [f"cmake -DLEGACY_PYTHON_PRE_3_8={legacy_python_cmake_flag} -DABI_EQUALS_1={abi_cxx_flag} .."],
                cwd=build_dir_abs,
                shell=True,
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(e.output) from None

        print(f"-- Running multipy runtime build in dir {build_dir_abs}")
        try:
            subprocess.run(
                ["cmake --build . --config Release"],
                cwd=build_dir_abs,
                shell=True,
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(e.output) from None

        print(f"-- Running multipy runtime install in dir {build_dir_abs}")
        try:
            subprocess.check_call(
                ["cmake", "--install", ".", "--prefix", '"."'],
                cwd=build_dir_abs,
                shell=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(e.output) from None
        # TODO
        # followups: gen examples, copy .so out.


ext_modules = [
    MultipyRuntimeExtension("multipy.so"),
]


def get_version():
    # get version string from version.py
    # TODO: ideally the version.py should be generated when setup is run
    version_file = os.path.join(os.path.dirname(__file__), "multipy/version.py")
    version_regex = r"__version__ = ['\"]([^'\"]*)['\"]"
    with open(version_file, "r") as f:
        version = re.search(version_regex, f.read(), re.M).group(1)
        return version


def get_nightly_version():
    today = date.today()
    return f"{today.year}.{today.month}.{today.day}"


if __name__ == "__main__":
    if sys.version_info < (3, 7):
        sys.exit("python >= 3.7 required for multipy")

    name = "multipy"
    NAME_ARG = "--override-name"
    if NAME_ARG in sys.argv:
        idx = sys.argv.index(NAME_ARG)
        name = sys.argv.pop(idx + 1)
        sys.argv.pop(idx)
    is_nightly = "nightly" in name

    with open("README.md", encoding="utf8") as f:
        readme = f.read()

    with open("requirements.txt") as f:
        reqs = f.read()

    with open("dev-requirements.txt") as f:
        dev_reqs = f.read()

    version = get_nightly_version() if is_nightly else get_version()
    print(f"-- {name} building version: {version}")

    setup(
        # Metadata
        name=name,
        version=version,
        author="MultiPy Devs",
        # TODO: @sahanp create email for MultiPy
        author_email="sahanp@fb.com",
        description="package + torch::deploy",
        long_description=readme,
        long_description_content_type="text/markdown",
        url="https://github.com/pytorch/multipy",
        license="BSD-3",
        keywords=["pytorch", "machine learning"],
        python_requires=">=3.7",
        install_requires=reqs.strip().split("\n"),
        include_package_data=True,
        packages=find_packages(exclude=()),
        extras_require={
            "dev": dev_reqs,
            ':python_version < "3.8"': [
                # latest numpy doesn't support 3.7
                "numpy<=1.21.6",
            ],
        },
        # Cmake invocation for runtime build.
        ext_modules=ext_modules,
        cmdclass=dict(build_ext=MultipyRuntimeBuild),
        # PyPI package information.
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: BSD License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    )
