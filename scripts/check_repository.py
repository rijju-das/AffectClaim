#!/usr/bin/env python3
"""Fail when tracked files look like accidental data/model artefacts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MAX_TRACKED_BYTES = 5 * 1024 * 1024
FORBIDDEN_SUFFIXES = {
    ".7z", ".avi", ".bin", ".ckpt", ".flac", ".h5", ".hdf5", ".joblib",
    ".mkv", ".mov", ".mp4", ".npy", ".npz", ".onnx", ".parquet", ".pickle",
    ".pkl", ".pt", ".pth", ".rar", ".safetensors", ".tar", ".tgz", ".wav", ".zip",
}


def repository_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    )
    return [REPOSITORY_ROOT / item.decode() for item in result.stdout.split(b"\0") if item]


def main() -> int:
    problems: list[str] = []
    for path in repository_files():
        if not path.exists():
            continue
        relative = path.relative_to(REPOSITORY_ROOT)
        if path.stat().st_size > MAX_TRACKED_BYTES:
            problems.append(f"tracked file exceeds 5 MiB: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            problems.append(f"tracked binary/data format: {relative}")
    if problems:
        print("Repository safety check failed:", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        return 1
    print("Repository safety check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
