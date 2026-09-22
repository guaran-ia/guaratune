#!/usr/bin/env python3
"""Run one generated LLaMA Factory training config."""

from __future__ import annotations

import click
import glob
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


def default_wandb_env_file(config_path: str) -> str:
    """Infer the default generated W&B env file for a training config.

    Args:
        config_path: Generated training config path.

    Returns:
        Model-specific W&B env path when the config is model-scoped, otherwise the
        legacy generated W&B env path.
    """
    normalized = os.path.normpath(config_path)
    parts = normalized.split(os.sep)
    generated_parts = ['configs', 'train', 'generated']
    for index in range(0, len(parts) - len(generated_parts) + 1):
        if parts[index:index + len(generated_parts)] != generated_parts:
            continue

        remainder = parts[index + len(generated_parts):]
        if len(remainder) >= 3:
            model_key = remainder[1]
            return os.path.join('configs', 'train', 'generated', model_key, 'wandb.env')

    return DEFAULT_WANDB_ENV_FILE


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
    if not os.path.isdir('data/train'):
        raise FileNotFoundError(
            'Missing data/train/. Reconstruct generated corpora with:\n'
            '  python -m src.prepare_data --reconstruct --config configs/data/gemma4_cpt.yaml --overwrite'
        )


def resolve_config_paths(config_patterns: tuple[str, ...]) -> list[str]:
    """Resolve config paths and glob patterns into YAML files.

    Args:
        config_patterns: Explicit config paths or glob patterns.

    Returns:
        Sorted unique config file paths.

    Raises:
        FileNotFoundError: If a pattern does not resolve to any file.
        ValueError: If an input resolves to a non-YAML file.
    """
    resolved = []
    seen = set()
    for pattern in config_patterns:
        matches = sorted(glob.glob(pattern)) if glob.has_magic(pattern) else [pattern]
        file_matches = [path for path in matches if os.path.isfile(path)]
        if not file_matches:
            raise FileNotFoundError(f'Training config pattern matched no files: {pattern}')

        for path in file_matches:
            if not path.endswith(('.yaml', '.yml')):
                raise ValueError(f'Training config must be a YAML file: {path}')
            normalized = os.path.normpath(path)
            if normalized not in seen:
                resolved.append(normalized)
                seen.add(normalized)

    return resolved


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
        wandb_env_file = env.get('WANDB_ENV_FILE', default_wandb_env_file(config_path))
        if os.path.isfile(wandb_env_file):
            load_env_file(wandb_env_file, env)
        elif wandb_env_file != DEFAULT_WANDB_ENV_FILE and os.path.isfile(DEFAULT_WANDB_ENV_FILE):
            load_env_file(DEFAULT_WANDB_ENV_FILE, env)
        else:
            print(f'W&B env file not found: {wandb_env_file}', file=sys.stderr)
            print(
                'Generate it with:\n'
                '  python -m src.generate_train_configs --matrix configs/train/gemma4-12_cpt_matrix.yaml --overwrite',
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


def cleanup_optimizers(output_dir: str, enabled: bool = True) -> None:
    """Remove optimizer.pt files from training output directory.

    Safe to call after training; frees disk space by removing optimizer state files.
    Model weights and other essential files are preserved.

    Args:
        output_dir: Training output directory path.
        enabled: Whether to perform cleanup. If False, function returns immediately.

    Returns:
        None.
    """
    if not enabled:
        return

    if not os.path.isdir(output_dir):
        print(f'[cleanup] Output directory not found: {output_dir}', file=sys.stderr)
        return

    optimizer_files = []
    total_size = 0

    try:
        for root, _, filenames in os.walk(output_dir):
            for filename in filenames:
                if filename == 'optimizer.pt':
                    filepath = os.path.join(root, filename)
                    try:
                        file_size = os.path.getsize(filepath)
                        optimizer_files.append((filepath, file_size))
                        total_size += file_size
                    except OSError as e:
                        print(f'[cleanup] Warning: Could not stat {filepath}: {e}', file=sys.stderr)

        if not optimizer_files:
            print('[cleanup] No optimizer.pt files found to remove.', flush=True)
            return

        for filepath, file_size in optimizer_files:
            try:
                os.remove(filepath)
                size_gb = file_size / (1024 ** 3)
                print(f'[cleanup] Removed {filepath} ({size_gb:.2f} GB)', flush=True)
            except OSError as e:
                print(f'[cleanup] Warning: Could not remove {filepath}: {e}', file=sys.stderr)

        total_gb = total_size / (1024 ** 3)
        print(
            f'[cleanup] Successfully removed {len(optimizer_files)} optimizer file(s) '
            f'({total_gb:.2f} GB freed)',
            flush=True,
        )
    except Exception as e:
        print(f'[cleanup] Error during cleanup: {e}', file=sys.stderr)


@click.command(
    context_settings={'show_default': True},
    help='Run one or more generated LLaMA Factory training configs.',
)
@click.argument('config_patterns', nargs=-1, required=True)
@click.option(
    '--env-file',
    type=click.Path(dir_okay=False),
    default=DEFAULT_SECRET_ENV_FILE,
    help='Local secret env file containing values such as HF_TOKEN and WANDB_API_KEY.',
)
@click.option(
    '--cleanup-optimizers',
    'cleanup_optimizer_files',
    type=bool,
    default=True,
    help='Remove optimizer.pt files after training completes to save disk space.',
)
def main(
    config_patterns: tuple[str, ...], env_file: str, cleanup_optimizer_files: bool
) -> None:
    """Run training configs through LLaMA Factory.

    Args:
        config_patterns: Generated LLaMA Factory training config paths or glob patterns.
        env_file: Local secret env file path.
        cleanup_optimizer_files: Whether to remove optimizer.pt files after training.

    Returns:
        None.
    """
    try:
        config_paths = resolve_config_paths(config_patterns)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    for config_path in config_paths:
        try:
            require_training_inputs(config_path)
            env = prepare_environment(config_path, env_file)
            config = load_yaml(config_path)
        except Exception as exc:
            raise click.ClickException(str(exc)) from exc

        print(f'[run] {config_path}', flush=True)
        result = subprocess.run(['llamafactory-cli', 'train', config_path], env=env, check=False)
        if result.returncode != 0:
            raise SystemExit(result.returncode)

        output_dir = config.get('output_dir')
        if output_dir:
            cleanup_optimizers(output_dir, enabled=cleanup_optimizer_files)  # type:ignore


if __name__ == '__main__':
    main()
