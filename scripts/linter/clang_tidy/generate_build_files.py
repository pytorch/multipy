import os
import subprocess
import sys
from typing import List


def run_cmd(cmd: List[str]) -> None:
    print(f"Running: {cmd}")
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = (
        result.stdout.decode("utf-8").strip(),
        result.stderr.decode("utf-8").strip(),
    )
    print(stdout)
    print(stderr)
    if result.returncode != 0:
        print(f"Failed to run {cmd}")
        exit(1)


def run_timed_cmd(cmd: List[str]) -> None:
    run_cmd(["time"] + cmd)


def update_submodules() -> None:
    run_cmd(["git", "submodule", "update", "--init", "--recursive"])


def gen_compile_commands() -> None:
    os.environ["USE_NCCL"] = "0"
    os.environ["CC"] = "clang"
    os.environ["CXX"] = "clang++"
    run_timed_cmd([sys.executable, "setup.py", "--cmake-only", "build"])


def run_autogen() -> None:
    run_timed_cmd(
        [
            sys.executable,
            "-m",
            "torchgen.gen",
            "-s",
            "aten/src/ATen",
            "-d",
            "build/aten/src/ATen",
            "--per-operator-headers",
        ]
    )

    run_timed_cmd(
        [
            sys.executable,
            "tools/setup_helpers/generate_code.py",
            "--native-functions-path",
            "aten/src/ATen/native/native_functions.yaml",
            "--tags-path",
            "aten/src/ATen/native/tags.yaml",
            "--gen_lazy_ts_backend",
        ]
    )


def generate_build_files() -> None:
    update_submodules()
    gen_compile_commands()
    run_autogen()


if __name__ == "__main__":
    generate_build_files()
