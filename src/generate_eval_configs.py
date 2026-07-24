#!/usr/bin/env python3
"""Generate lm-evaluation-harness configs from an evaluation matrix."""

from __future__ import annotations

import click
import os
import yaml

from typing import Any, Iterator


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


def require_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    """Read a required mapping from a dictionary.

    Args:
        data: Dictionary to inspect.
        key: Required mapping key.

    Returns:
        Mapping value.

    Raises:
        ValueError: If the key is missing or is not a mapping.
    """
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f'Matrix key must be a mapping: {key}')

    return value


def merge_dicts(*items: dict[str, Any]) -> dict[str, Any]:
    """Merge dictionaries left to right, skipping None values.

    Args:
        items: Dictionaries to merge.

    Returns:
        Merged dictionary.
    """
    merged = {}
    for item in items:
        for key, value in item.items():
            if value is not None:
                merged[key] = value

    return merged


def write_yaml(path: str, data: dict[str, Any], overwrite: bool) -> None:
    """Write one YAML file.

    Args:
        path: Destination YAML path.
        data: Data to serialize.
        overwrite: Whether an existing file can be replaced.

    Returns:
        None.

    Raises:
        FileExistsError: If the file exists and overwrite is false.
    """
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f'Refusing to overwrite existing config: {path}')

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def run_name(model_key: str, variant: str, suite_name: str) -> str:
    """Build a stable evaluation run name.

    Args:
        model_key: Short model identifier.
        variant: Model variant name.
        suite_name: Evaluation suite name.

    Returns:
        Stable run name used for YAML filenames and output directories.
    """
    return f'{model_key}_{variant}_{suite_name}'


def training_run_name(model_key: str, method: str, corpus: str, rank: int | None) -> str:
    """Build the training run name that produced an evaluated checkpoint.

    Args:
        model_key: Short model identifier.
        method: Training method.
        corpus: Corpus configuration name.
        rank: Optional LoRA rank.

    Returns:
        Training run name matching `src.generate_train_configs`.
    """
    if method == 'lora':
        if rank is None:
            raise ValueError('LoRA training run names require a rank.')
        return f'{model_key}_lora_r{rank}_{corpus}'

    return f'{model_key}_{method}_{corpus}'


def training_output_dir(
    matrix: dict[str, Any], method: str, corpus: str, rank: int | None
) -> str:
    """Build the expected training output directory for one variant.

    Args:
        matrix: Loaded evaluation matrix.
        method: Training method.
        corpus: Corpus configuration name.
        rank: Optional LoRA rank.

    Returns:
        Expected checkpoint or adapter directory path.
    """
    model = require_mapping(matrix, 'model')
    root = matrix['training_outputs_root']
    if method == 'lora':
        return os.path.join(root, method, f'rank_{rank}', corpus)

    name = training_run_name(model['key'], method, corpus, rank)
    return os.path.join(root, method, corpus, name)


def model_args_for_variant(
    matrix: dict[str, Any],
    variant_kind: str,
    corpus: str | None,
    rank: int | None,
) -> dict[str, Any]:
    """Build lm-eval Hugging Face model arguments for one variant.

    Args:
        matrix: Loaded evaluation matrix.
        variant_kind: Variant type: base, full, or lora.
        corpus: Corpus configuration name for trained variants.
        rank: Optional LoRA rank.

    Returns:
        Model arguments passed to lm-eval.
    """
    model = require_mapping(matrix, 'model')
    args = {
        'pretrained': model['model_name_or_path'],
        'revision': model.get('model_revision', 'main'),
        'dtype': model.get('dtype'),
    }

    if variant_kind == 'base':
        return args
    if corpus is None:
        raise ValueError(f'{variant_kind} variants require a corpus.')
    if variant_kind == 'full':
        args['pretrained'] = training_output_dir(matrix, 'full', corpus, None)
        args['tokenizer'] = model['model_name_or_path']
        args.pop('revision', None)
        return args
    if variant_kind == 'lora':
        args['peft'] = training_output_dir(matrix, 'lora', corpus, rank)
        return args

    raise ValueError(f'Unsupported variant kind: {variant_kind}')


def variant_name(
    variant_kind: str, corpus: str | None, rank: int | None
) -> str:
    """Build a stable model variant name.

    Args:
        variant_kind: Variant type: base, full, or lora.
        corpus: Corpus configuration name for trained variants.
        rank: Optional LoRA rank.

    Returns:
        Stable variant name.
    """
    if variant_kind == 'base':
        return 'base'
    if variant_kind == 'full':
        return f'full_{corpus}'
    if variant_kind == 'lora':
        return f'lora_r{rank}_{corpus}'

    raise ValueError(f'Unsupported variant kind: {variant_kind}')


def iter_variants(
    profile: dict[str, Any]
) -> Iterator[tuple[str, str | None, int | None]]:
    """Yield model variants requested by a profile.

    Args:
        profile: Evaluation profile mapping.

    Returns:
        Iterator of variant kind, corpus, and optional LoRA rank tuples.
    """
    if bool(profile.get('include_base', False)):
        yield 'base', None, None

    corpora = profile.get('corpora', [])
    if bool(profile.get('include_full', False)):
        for corpus in corpora:
            yield 'full', corpus, None

    for rank in profile.get('lora_ranks', []):
        for corpus in corpora:
            yield 'lora', corpus, int(rank)


def evaluation_config(
    matrix: dict[str, Any],
    profile_name: str,
    suite_name: str,
    variant_kind: str,
    corpus: str | None,
    rank: int | None,
) -> dict[str, Any]:
    """Build one lm-evaluation-harness config.

    Args:
        matrix: Loaded evaluation matrix.
        profile_name: Profile being generated.
        suite_name: Evaluation suite name.
        variant_kind: Variant type: base, full, or lora.
        corpus: Corpus configuration name for trained variants.
        rank: Optional LoRA rank.

    Returns:
        lm-evaluation-harness configuration dictionary.
    """
    model = require_mapping(matrix, 'model')
    defaults = require_mapping(matrix, 'defaults')
    profiles = require_mapping(matrix, 'profiles')
    profile = require_mapping(profiles, profile_name)
    suites = require_mapping(matrix, 'suites')
    suite = require_mapping(suites, suite_name)
    profile_overrides = profile.get('overrides', {})
    if not isinstance(profile_overrides, dict):
        raise ValueError(f'Profile overrides must be a mapping: {profile_name}')

    variant = variant_name(variant_kind, corpus, rank)
    name = run_name(model['key'], variant, suite_name)
    output_path = os.path.join(
        matrix['outputs_root'], profile_name, model['key'], variant, suite_name
    )

    return merge_dicts(
        defaults,
        {
            'model_args': model_args_for_variant(matrix, variant_kind, corpus, rank),
            'tasks': suite['tasks'],
            'include_path': matrix['task_include_path'],
            'output_path': output_path,
            'device': model.get('device'),
            'batch_size': model.get('batch_size'),
            'max_batch_size': model.get('max_batch_size'),
            'trust_remote_code': bool(model.get('trust_remote_code', False)),
            'seed': matrix.get('seed'),
            'metadata': {
                'run_name': name,
                'model_key': model['key'],
                'variant': variant,
                'suite': suite_name,
                'profile': profile_name,
                'corpus': corpus,
                'lora_rank': rank,
            },
        },
        suite,
        profile_overrides,
    )


def iter_configs(
    matrix: dict[str, Any], profile_name: str
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield generated config paths and payloads for one profile.

    Args:
        matrix: Loaded evaluation matrix.
        profile_name: Profile to generate.

    Returns:
        Iterator of config path and config dictionary tuples.
    """
    profiles = require_mapping(matrix, 'profiles')
    profile = require_mapping(profiles, profile_name)
    output_dir = matrix.get('generated_config_dir', 'configs/evaluation/generated')
    model = require_mapping(matrix, 'model')

    for variant_kind, corpus, rank in iter_variants(profile):
        variant = variant_name(variant_kind, corpus, rank)
        for suite_name in profile['suites']:
            name = run_name(model['key'], variant, suite_name)
            path = os.path.join(output_dir, profile_name, model['key'], f'{name}.yaml')
            yield path, evaluation_config(
                matrix, profile_name, suite_name, variant_kind, corpus, rank
            )


@click.command(
    context_settings={'show_default': True},
    help='Generate lm-evaluation-harness YAML configs from an evaluation matrix.',
)
@click.option(
    '--matrix',
    'matrix_path',
    type=click.Path(exists=True, dir_okay=False),
    default='configs/evaluation/gemma4-12_eval_matrix.yaml',
    help='Evaluation matrix YAML file.',
)
@click.option(
    '--profile',
    'profile_names',
    multiple=True,
    help='Profile to generate. Repeat to generate multiple profiles. Defaults to all profiles.',
)
@click.option('--overwrite', is_flag=True, help='Overwrite generated YAML files.')
def main(matrix_path: str, profile_names: tuple[str, ...], overwrite: bool) -> None:
    """Generate lm-evaluation-harness YAML files.

    Args:
        matrix_path: Evaluation matrix YAML file.
        profile_names: Profile names to generate.
        overwrite: Whether existing YAML files can be replaced.

    Returns:
        None.
    """
    matrix = load_yaml(matrix_path)
    profiles = require_mapping(matrix, 'profiles')
    selected_profiles = profile_names or tuple(profiles.keys())
    written = 0

    for profile_name in selected_profiles:
        if profile_name not in profiles:
            raise ValueError(f'Unknown profile: {profile_name}')

        for path, config in iter_configs(matrix, profile_name):
            write_yaml(path, config, overwrite)
            print(path)
            written += 1

    print(f'[done] wrote {written} config files')


if __name__ == '__main__':
    main()
