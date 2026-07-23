#!/usr/bin/env python3
"""Run one generated LLaMA Factory training config."""

from __future__ import annotations

import click
import os
import shlex
import subprocess
import sys
import yaml

from typing import Any


DEFAULT_WANDB_ENV_FILE = 'configs/train/generated/wandb.env'
DEFAULT_SECRET_ENV_FILE = '.env'
HF_TOKEN_ENV_KEYS = ('HF_TOKEN', 'HF_ACCESS_TOKEN')


def load_yaml(path: str) -> dict[str, Any]:
    """Load a YAML file as a dictionary.

    Args:
        path: YAML file path.

    Returns:
        Parsed YAML dictionary.

    Raises:
        ValueError: If the YAML file does not contain a mapping.
    """
    with open(path, 'r', encoding='utf-8') as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError(f'YAML file must contain a mapping: {path}')

    return data


def uses_wandb(config_path: str) -> bool:
    """Check whether a LLaMA Factory config reports to W&B.

    Args:
        config_path: Training config path.

    Returns:
        True when the config has report_to set to wandb.
    """
    config = load_yaml(config_path)
    report_to = config.get('report_to')
    if isinstance(report_to, str):
        return report_to == 'wandb'
    if isinstance(report_to, list):
        return 'wandb' in report_to

    return False


def parse_env_assignment(line: str) -> tuple[str, str] | None:
    """Parse one generated shell env assignment.

    Args:
        line: Env file line, optionally prefixed with export.

    Returns:
        Environment key/value pair, or None for comments and blank lines.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith('#'):
        return None

    parts = shlex.split(stripped, posix=True)
    if not parts:
        return None
    if parts[0] == 'export':
        parts = parts[1:]
    if len(parts) != 1 or '=' not in parts[0]:
        raise ValueError(f'Invalid env assignment: {line.rstrip()}')

    key, value = parts[0].split('=', 1)
    return key, value


def load_env_file(path: str, env: dict[str, str]) -> None:
    """Load environment values into an environment mapping.

    Existing process environment values take precedence over generated defaults.

    Args:
        path: Env file path.
        env: Environment mapping to update.

    Returns:
        None.
    """
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            assignment = parse_env_assignment(line)
            if assignment is None:
                continue

            key, value = assignment
            if key not in os.environ:
                env[key] = value


def normalize_hf_token_env(env: dict[str, str]) -> None:
    """Normalize supported Hugging Face token environment variable names.

    Args:
        env: Environment mapping to update.

    Returns:
        None.
    """
    if env.get('HF_TOKEN'):
        return
    if env.get('HF_ACCESS_TOKEN'):
        env['HF_TOKEN'] = env['HF_ACCESS_TOKEN']


def require_training_inputs(config_path: str) -> None:
    """Validate files required before launching training.

    Args:
        config_path: Training config path.

    Returns:
        None.

    Raises:
        FileNotFoundError: If a required file or directory is missing.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f'Training config not found: {config_path}')
    if not os.path.isfile('data/dataset_info.json'):
        raise FileNotFoundError(
            'Missing data/dataset_info.json. Run dataset preparation or reconstruction first.'
        )
    if not os.path.isdir('data/processed'):
        raise FileNotFoundError(
            'Missing data/processed/. Reconstruct generated corpora with:\n'
            '  python -m src.prepare_data --reconstruct --config configs/data/gemma4_cpt.yaml --overwrite'
        )


def prepare_environment(config_path: str, env_file: str) -> dict[str, str]:
    """Prepare environment variables for a training launch.

    Args:
        config_path: Training config path.
        env_file: Local secret env file path.

    Returns:
        Environment mapping for the subprocess.
    """
    env = os.environ.copy()
    if os.path.isfile(env_file):
        load_env_file(env_file, env)

    normalize_hf_token_env(env)
    env.setdefault('TOKENIZERS_PARALLELISM', 'false')

    if not any(env.get(key) for key in HF_TOKEN_ENV_KEYS):
        print(
            'HF_TOKEN is not set. Set HF_TOKEN or HF_ACCESS_TOKEN if the Gemma model '
            'requires gated Hugging Face access.',
            file=sys.stderr,
        )

    if uses_wandb(config_path):
        wandb_env_file = env.get('WANDB_ENV_FILE', DEFAULT_WANDB_ENV_FILE)
        if os.path.isfile(wandb_env_file):
            load_env_file(wandb_env_file, env)
        else:
            print(f'W&B env file not found: {wandb_env_file}', file=sys.stderr)
            print(
                'Generate it with:\n'
                '  python -m src.generate_train_configs --matrix configs/train/gemma4_cpt_matrix.yaml --overwrite',
                file=sys.stderr,
            )

        if env.get('WANDB_DIR'):
            os.makedirs(env['WANDB_DIR'], exist_ok=True)

        if (
            env.get('WANDB_MODE', 'online') == 'online'
            and 'WANDB_API_KEY' not in env
            and not os.path.isfile(os.path.expanduser('~/.netrc'))
        ):
            print(
                'W&B online reporting is enabled, but WANDB_API_KEY is not set and no ~/.netrc was found.',
                file=sys.stderr,
            )
            print(
                'Run "wandb login" or set WANDB_API_KEY on the VM before training.',
                file=sys.stderr,
            )

        print(
            'W&B reporting: '
            f'project={env.get("WANDB_PROJECT", "wandb-default")} '
            f'mode={env.get("WANDB_MODE", "wandb-default")} '
            f'tags={env.get("WANDB_TAGS", "unset")}'
        )

    return env


@click.command(
    context_settings={'show_default': True},
    help='Run one generated LLaMA Factory training config.',
)
@click.argument('config_path', type=click.Path(exists=True, dir_okay=False))
@click.option(
    '--env-file',
    type=click.Path(dir_okay=False),
    default=DEFAULT_SECRET_ENV_FILE,
    help='Local secret env file containing values such as HF_TOKEN and WANDB_API_KEY.',
)
def main(config_path: str, env_file: str) -> None:
    """Run one training config through LLaMA Factory.

    Args:
        config_path: Generated LLaMA Factory training config path.
        env_file: Local secret env file path.

    Returns:
        None.
    """
    try:
        require_training_inputs(config_path)
        env = prepare_environment(config_path, env_file)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    result = subprocess.run(['llamafactory-cli', 'train', config_path], env=env, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
