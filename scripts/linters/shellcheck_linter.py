import argparse
import json
import logging
import shutil
import subprocess
import time
from enum import Enum
from typing import List, NamedTuple, Optional


LINTER_CODE = "SHELLCHECK"


class LintSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"


class LintMessage(NamedTuple):
    path: Optional[str]
    line: Optional[int]
    char: Optional[int]
    code: str
    severity: LintSeverity
    name: str
    original: Optional[str]
    replacement: Optional[str]
    description: Optional[str]


def run_command(
    args: List[str],
) -> "subprocess.CompletedProcess[bytes]":
    logging.debug("$ %s", " ".join(args))
    start_time = time.monotonic()
    try:
        return subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    finally:
        end_time = time.monotonic()
        logging.debug("took %dms", (end_time - start_time) * 1000)


def check_files(
    files: List[str],
) -> List[LintMessage]:
    try:
        proc = run_command(
            ["shellcheck", "--external-sources", "--format=json1"] + files
        )
    except OSError as err:
        return [
            LintMessage(
                path=None,
                line=None,
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.ERROR,
                name="command-failed",
                original=None,
                replacement=None,
                description=(f"Failed due to {err.__class__.__name__}:\n{err}"),
            )
        ]
    stdout = str(proc.stdout, "utf-8").strip()
    results = json.loads(stdout)["comments"]
    return [
        LintMessage(
            path=result["file"],
            name=f"SC{result['code']}",
            description=result["message"],
            line=result["line"],
            char=result["column"],
            code=LINTER_CODE,
            severity=LintSeverity.ERROR,
            original=None,
            replacement=None,
        )
        for result in results
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="shellcheck runner",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )

    if shutil.which("shellcheck") is None:
        err_msg = LintMessage(
            path="<none>",
            line=None,
            char=None,
            code=LINTER_CODE,
            severity=LintSeverity.ERROR,
            name="command-failed",
            original=None,
            replacement=None,
            description="shellcheck is not installed, did you forget to run `lintrunner init`?",
        )
        print(json.dumps(err_msg._asdict()), flush=True)
        exit(0)

    args = parser.parse_args()

    lint_messages = check_files(args.filenames)
    for lint_message in lint_messages:
        print(json.dumps(lint_message._asdict()), flush=True)
