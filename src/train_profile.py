#!/usr/bin/env python3
"""Run a generated LLaMA Factory training profile."""

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
    config_dir = os.path.join('configs', 'train', 'generated', profile)
    if not os.path.isdir(config_dir):
        raise FileNotFoundError(
            f'Generated config profile not found: {config_dir}\n'
            'Generate configs with:\n'
            '  python -m src.generate_train_configs --matrix configs/train/gemma4_cpt_matrix.yaml --overwrite'
        )

    paths = []
    for root, _, filenames in os.walk(config_dir):
        for filename in filenames:
            if filename.endswith('.yaml'):
                paths.append(os.path.join(root, filename))

    return sorted(paths)


@click.command(
    context_settings={'show_default': True},
    help='Run every generated training config in one profile.',
)
@click.argument('profile', required=False, default='experiments')
@click.option(
    '--env-file',
    type=click.Path(dir_okay=False),
    default='.env',
    help='Local secret env file forwarded to each training run.',
)
def main(profile: str, env_file: str) -> None:
    """Run all configs in one profile.

    Args:
        profile: Generated profile name.
        env_file: Local secret env file path.

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
                'src.train_config',
                config_path,
                '--env-file',
                env_file,
            ],
            check=False,
        )
        if result.returncode != 0:
            raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
