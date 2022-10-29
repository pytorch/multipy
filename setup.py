#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import os.path
import re
import shutil
import subprocess
import sys
from datetime import date
from distutils.command.clean import clean

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.develop import develop
from setuptools.command.install import install


class MultipyRuntimeExtension(Extension):
    def __init__(self, name):
        Extension.__init__(self, name, sources=[])


def get_cmake_version():
    output = subprocess.check_output(["cmake", "--version"]).decode("utf-8")
    return output.splitlines()[0].split()[2]


class MultipyRuntimeCmake(object):
    user_options = [
        ("cmakeoff", None, None),
        ("cudatests", None, None),
        ("abicxx", None, None),
    ]


class MultipyRuntimeDevelop(MultipyRuntimeCmake, develop):
    user_options = develop.user_options + MultipyRuntimeCmake.user_options

    def initialize_options(self):
        develop.initialize_options(self)
        self.cmakeoff = None

        # TODO(tristanr): remove once unused
        self.abicxx = None

        self.cudatests = None

    def finalize_options(self):
        develop.finalize_options(self)
        if self.cmakeoff is not None:
            self.distribution.get_command_obj("build_ext").cmake_off = True
        if self.cudatests is not None:
            self.distribution.get_command_obj("build_ext").cuda_tests_flag = "ON"


class MultipyRuntimeBuild(MultipyRuntimeCmake, build_ext):
    user_options = build_ext.user_options + MultipyRuntimeCmake.user_options
    cmake_off = False
    cuda_tests_flag = "OFF"

    def run(self):
        if self.cmake_off:
            return
        try:
            cmake_version_comps = get_cmake_version().split(".")
            if cmake_version_comps[0] < "3" or cmake_version_comps[1] < "12":
                raise RuntimeError(
                    "CMake 3.12 or later required for multipy runtime installation."
                )
        except OSError:
            raise RuntimeError(
                "Error fetching cmake version. Please ensure cmake is installed correctly."
            ) from None
        base_dir = os.path.abspath(os.path.dirname(__file__))
        build_dir = "multipy/runtime/build"
        build_dir_abs = base_dir + "/" + build_dir
        if not os.path.exists(build_dir_abs):
            os.makedirs(build_dir_abs)

        print(f"build_lib {self.build_lib}")
        print(f"-- Running multipy runtime makefile in dir {build_dir_abs}")
        try:
            subprocess.run(
                [f"cmake -DBUILD_CUDA_TESTS={self.cuda_tests_flag} .."],
                cwd=build_dir_abs,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(e.output.decode("utf-8")) from None

        print(f"-- Running multipy runtime build in dir {build_dir_abs}")
        try:
            subprocess.run(
                ["cmake --build . --config Release"],
                cwd=build_dir_abs,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(e.output.decode("utf-8")) from None

        print(f"-- Running multipy runtime install in dir {build_dir_abs}")
        try:
            subprocess.run(
                ['cmake --install . --prefix "."'],
                cwd=build_dir_abs,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(e.output.decode("utf-8")) from None

        print("-- Copying build outputs")
        paths = [
            "multipy/runtime/build/libtorch_deploy.a",
            "multipy/runtime/build/interactive_embedded_interpreter",
            "multipy/runtime/build/test_deploy",
        ]
        for path in paths:
            target = os.path.join(self.build_lib, path)
            target_dir = os.path.dirname(target)
            if not os.path.exists(target_dir):
                print(f"creating dir {target_dir}")
                os.makedirs(target_dir)
            print(f"copying {path} -> {target}")
            shutil.copy2(path, target)


class MultipyRuntimeClean(clean):
    def run(self):
        paths = [
            "multipy/runtime/build",
        ]
        for path in paths:
            if os.path.exists(path):
                print(f"removing: {path}")
                shutil.rmtree(path)
        super().run()


class MultipyRuntimeInstall(MultipyRuntimeCmake, install):
    user_options = install.user_options + MultipyRuntimeCmake.user_options

    def initialize_options(self):
        install.initialize_options(self)
        self.cmakeoff = None

    def finalize_options(self):
        install.finalize_options(self)
        if self.cmakeoff is not None:
            self.distribution.get_command_obj("build_ext").cmake_off = True

    def run(self):
        # Setuptools/setup.py on docker image has some interesting behavior, in that the
        # optional "--cmakeoff" flag gets applied to dependencies specified in
        # requirements.txt as well (installed using "install-requires" argument of setup()).
        # Since we obviously don't want things like "pip install numpy --install-option=--cmakeoff",
        # we install these deps directly in this overridden install command without
        # spurious options, instead of using "install-requires".
        base_dir = os.path.abspath(os.path.dirname(__file__))
        try:
            reqs_filename = "requirements.txt"
            subprocess.run(
                [f"pip install -r {reqs_filename}"],
                cwd=base_dir,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(e.output.decode("utf-8")) from None
        install.run(self)


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

    name = "torchdeploy"
    NAME_ARG = "--override-name"
    if NAME_ARG in sys.argv:
        idx = sys.argv.index(NAME_ARG)
        name = sys.argv.pop(idx + 1)
        sys.argv.pop(idx)
    is_nightly = "nightly" in name

    with open("README.md", encoding="utf8") as f:
        readme = f.read()

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
        description="torch::deploy (multipy) is a C++ library that makes it easier to run eager PyTorch models in production by using independent python interpreters to avoid the GIL.",
        long_description=readme,
        long_description_content_type="text/markdown",
        url="https://github.com/pytorch/multipy",
        license="BSD-3",
        keywords=["pytorch", "machine learning", "inference"],
        python_requires=">=3.7",
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
        cmdclass={
            "build_ext": MultipyRuntimeBuild,
            "develop": MultipyRuntimeDevelop,
            "install": MultipyRuntimeInstall,
            "clean": MultipyRuntimeClean,
        },
        package_data={
            "multipy": [
                "runtime/*",
                "runtime/example/*",
                "runtime/example/fx/*",
                "runtime/interpreter/*",
                "runtime/third-party/fmt/*",
                "runtime/third-party/fmt/include/fmt/*",
                "runtime/third-party/fmt/src/*",
                "runtime/third-party/fmt/support/cmake/*",
            ]
        },
        data_files=[
            (
                "",
                [
                    "requirements.txt",
                    "dev-requirements.txt",
                ],
            )
        ],
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
