#!/usr/bin/env python3
"""Run a generated lm-evaluation-harness evaluation profile."""

from __future__ import annotations

import click
import os
import subprocess
import sys


def generated_configs(profile: str) -> list[str]:
    """List generated configs for one profile.

    Args:
        profile: Generated profile name.

    Returns:
        Sorted list of YAML config paths.

    Raises:
        FileNotFoundError: If the generated profile directory is missing.
    """
    config_dir = os.path.join('configs', 'evaluation', 'generated', profile)
    if not os.path.isdir(config_dir):
        raise FileNotFoundError(
            f'Generated evaluation profile not found: {config_dir}\n'
            'Generate configs with:\n'
            '  python -m src.generate_eval_configs --matrix configs/evaluation/gemma4_eval_matrix.yaml --overwrite'
        )

    paths = []
    for root, _, filenames in os.walk(config_dir):
        for filename in filenames:
            if filename.endswith('.yaml'):
                paths.append(os.path.join(root, filename))

    return sorted(paths)


@click.command(
    context_settings={'show_default': True},
    help='Run every generated evaluation config in one profile.',
)
@click.argument('profile', required=False, default='experiments')
def main(profile: str) -> None:
    """Run all evaluation configs in one profile.

    Args:
        profile: Generated profile name.

    Returns:
        None.
    """
    try:
        configs = generated_configs(profile)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    for config_path in configs:
        print(f'[run] {config_path}', flush=True)
        result = subprocess.run(
            [
                sys.executable,
                '-m',
                'src.eval_config',
                config_path,
            ],
            check=False,
        )
        if result.returncode != 0:
            raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
