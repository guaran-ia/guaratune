#!/usr/bin/env python3
"""Compile pip requirement locks while keeping PyTorch platform-specific."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


GPU_PACKAGES = ('torch', 'torchaudio', 'torchvision')


def is_gpu_requirement(line: str) -> bool:
    """Return whether a lock-file requirement line names a GPU-specific package."""
    normalized = line.lower()
    return any(normalized.startswith(f'{package}==') for package in GPU_PACKAGES)


def remove_gpu_requirements(path: Path) -> None:
    """Remove PyTorch package blocks from a pip-compile lock file.

    PyTorch wheel indexes and build tags depend on the target accelerator and platform.
    Those packages are therefore installed first from a dedicated requirements file.
    """
    result = []
    skip_block = False
    for line in path.read_text(encoding='utf-8').splitlines(keepends=True):
        if is_gpu_requirement(line):
            skip_block = True
            continue
        if skip_block and (line.startswith((' ', '\t', '#')) or not line.strip()):
            continue

        skip_block = False
        result.append(line)

    path.write_text(''.join(result), encoding='utf-8')


def compile_lock(source: str, output: str) -> None:
    """Compile and normalize one requirement lock file."""
    subprocess.run(
        [
            sys.executable,
            '-m',
            'piptools',
            'compile',
            '--allow-unsafe',
            '--strip-extras',
            '--output-file',
            output,
            source,
        ],
        check=True,
    )
    remove_gpu_requirements(Path(output))


def main() -> None:
    """Compile runtime and development locks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--dev-only',
        action='store_true',
        help='Compile only requirements-dev.txt.',
    )
    args = parser.parse_args()

    if not args.dev_only:
        compile_lock('requirements.in', 'requirements.txt')
    compile_lock('requirements-dev.in', 'requirements-dev.txt')


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
