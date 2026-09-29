#!/usr/bin/env python3
"""Check the local runtime dependencies required by the CPT framework."""

from __future__ import annotations

import argparse
import importlib
import shutil
import sys


REQUIRED_MODULES = (
    'datasets',
    'lm_eval',
    'llamafactory',
    'pyarrow',
    'sacrebleu',
    'transformers',
    'yaml',
)


def module_version(module_name: str) -> str:
    """Return a module version when exposed, otherwise an unavailable marker."""
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        return 'missing'

    return str(getattr(module, '__version__', 'installed'))


def main() -> None:
    """Print runtime availability and exit nonzero when requirements are missing."""
    parser = argparse.ArgumentParser(
        description='Check dependencies required by Guarania CPT training and evaluation.'
    )
    parser.add_argument(
        '--require-cuda',
        action='store_true',
        help='Fail unless PyTorch can access CUDA.',
    )
    args = parser.parse_args()

    print(f'Python: {sys.version.split()[0]}')
    missing = []
    for module_name in REQUIRED_MODULES:
        version = module_version(module_name)
        print(f'{module_name}: {version}')
        if version == 'missing':
            missing.append(module_name)

    print(f'llamafactory-cli: {shutil.which("llamafactory-cli") or "missing"}')
    if shutil.which('llamafactory-cli') is None:
        missing.append('llamafactory-cli')

    try:
        torch = importlib.import_module('torch')
    except ModuleNotFoundError:
        torch = None
        print('torch: missing')
        if args.require_cuda:
            missing.append('torch')
    else:
        cuda_available = bool(torch.cuda.is_available())
        print(f'torch: {torch.__version__}')
        print(f'CUDA available: {cuda_available}')
        if args.require_cuda and not cuda_available:
            missing.append('CUDA')

    if missing:
        print(f'[error] Missing requirements: {", ".join(sorted(set(missing)))}', file=sys.stderr)
        raise SystemExit(1)

    print('[ok] Environment is ready.')


if __name__ == '__main__':
    main()
