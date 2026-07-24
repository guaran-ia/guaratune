#!/usr/bin/env python3
"""Run a generated lm-evaluation-harness evaluation profile."""

from __future__ import annotations

import click
import fnmatch
import os
import subprocess
import sys


def list_yaml_files(config_dir: str) -> list[str]:
    """List YAML files below one generated config directory.

    Args:
        config_dir: Directory to scan recursively.

    Returns:
        Sorted YAML file paths.
    """
    paths = []
    for root, _, filenames in os.walk(config_dir):
        for filename in filenames:
            if filename.endswith('.yaml'):
                paths.append(os.path.join(root, filename))

    return sorted(paths)


def model_config_dirs(profile_dir: str) -> list[str]:
    """List model-scoped generated config directories.

    Args:
        profile_dir: Generated profile directory.

    Returns:
        Sorted model directory names.
    """
    models = []
    for name in os.listdir(profile_dir):
        path = os.path.join(profile_dir, name)
        if os.path.isdir(path) and list_yaml_files(path):
            models.append(name)

    return sorted(models)


def generated_configs(profile: str, model: str | None) -> list[str]:
    """List generated configs for one profile.

    Args:
        profile: Generated profile name.
        model: Optional model key used to select a model-scoped profile.

    Returns:
        Sorted list of YAML config paths.

    Raises:
        FileNotFoundError: If the generated profile directory is missing.
    """
    profile_dir = os.path.join('configs', 'evaluation', 'generated', profile)
    if not os.path.isdir(profile_dir):
        raise FileNotFoundError(
            f'Generated evaluation profile not found: {profile_dir}\n'
            'Generate configs with:\n'
            '  python -m src.generate_eval_configs --matrix configs/evaluation/gemma4-12_eval_matrix.yaml --overwrite'
        )

    if model is not None:
        config_dir = os.path.join(profile_dir, model)
        if not os.path.isdir(config_dir):
            raise FileNotFoundError(f'Generated evaluation model profile not found: {config_dir}')

        return list_yaml_files(config_dir)

    root_configs = [
        os.path.join(profile_dir, filename)
        for filename in os.listdir(profile_dir)
        if filename.endswith('.yaml')
    ]
    models = model_config_dirs(profile_dir)
    if len(models) == 1 and not root_configs:
        return list_yaml_files(os.path.join(profile_dir, models[0]))
    if root_configs and not models:
        return sorted(root_configs)

    available = ', '.join(models) if models else 'none'
    raise ValueError(
        f'Profile {profile} contains multiple or mixed model config sets. '
        f'Specify --model. Available model directories: {available}'
    )


def is_excluded(config_path: str, exclude_patterns: tuple[str, ...]) -> bool:
    """Check whether a config path matches any exclusion pattern.

    Args:
        config_path: Generated config path.
        exclude_patterns: Filename, stem, path, or glob patterns to exclude.

    Returns:
        True when the config should be skipped.
    """
    if not exclude_patterns:
        return False

    normalized = os.path.normpath(config_path)
    basename = os.path.basename(normalized)
    stem = os.path.splitext(basename)[0]
    candidates = (normalized, basename, stem)
    for pattern in exclude_patterns:
        normalized_pattern = os.path.normpath(pattern)
        for candidate in candidates:
            if candidate == normalized_pattern or fnmatch.fnmatch(candidate, normalized_pattern):
                return True

    return False


def apply_exclusions(configs: list[str], exclude_patterns: tuple[str, ...]) -> list[str]:
    """Filter generated config paths using exclusion patterns.

    Args:
        configs: Generated config paths.
        exclude_patterns: Filename, stem, path, or glob patterns to exclude.

    Returns:
        Config paths that do not match the exclusion patterns.
    """
    return [
        config_path
        for config_path in configs
        if not is_excluded(config_path, exclude_patterns)
    ]


@click.command(
    context_settings={'show_default': True},
    help='Run every generated evaluation config in one profile.',
)
@click.argument('profile', required=False, default='experiments')
@click.option(
    '--model',
    'model_key',
    help='Model key to run inside the generated profile, for example gemma4_12b.',
)
@click.option(
    '--exclude',
    'exclude_patterns',
    multiple=True,
    help='Config filename, stem, path, or glob pattern to skip. Repeat for multiple exclusions.',
)
def main(profile: str, model_key: str | None, exclude_patterns: tuple[str, ...]) -> None:
    """Run all evaluation configs in one profile.

    Args:
        profile: Generated profile name.
        model_key: Optional model key to run.
        exclude_patterns: Config exclusion patterns.

    Returns:
        None.
    """
    try:
        configs = generated_configs(profile, model_key)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    filtered_configs = apply_exclusions(configs, exclude_patterns)
    skipped = len(configs) - len(filtered_configs)
    if skipped:
        print(f'[skip] excluded {skipped} config(s)', flush=True)

    for config_path in filtered_configs:
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
