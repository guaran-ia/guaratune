#!/usr/bin/env python3
"""Run one generated lm-evaluation-harness config."""

from __future__ import annotations

import click
import importlib
import importlib.util
import json
import os
import yaml

from src.train_config import DEFAULT_SECRET_ENV_FILE, load_env_file, normalize_hf_token_env
from typing import Any


if importlib.util.find_spec('lm_eval') is not None:
    lm_eval = importlib.import_module('lm_eval')
    TaskManager = importlib.import_module('lm_eval.tasks').TaskManager
    handle_non_serializable = importlib.import_module(
        'lm_eval.utils'
    ).handle_non_serializable
else:
    lm_eval = None
    TaskManager = None
    handle_non_serializable = None


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


def looks_like_local_path(path: str) -> bool:
    """Return whether a model argument points to a local project path.

    Args:
        path: Model argument value.

    Returns:
        True when the value should exist on the local filesystem.
    """
    return path.startswith(('/', './', '../', 'outputs/'))


def require_lm_eval() -> None:
    """Validate that lm-evaluation-harness is importable.

    Args:
        None.

    Returns:
        None.

    Raises:
        ModuleNotFoundError: If lm-evaluation-harness is not installed.
    """
    if lm_eval is None or TaskManager is None or handle_non_serializable is None:
        raise ModuleNotFoundError(
            'lm-evaluation-harness is not installed. Install it on the GPU VM with:\n'
            '  pip install "lm_eval[hf]"'
        )


def require_evaluation_inputs(config_path: str) -> None:
    """Validate files required before launching evaluation.

    Args:
        config_path: Evaluation config path.

    Returns:
        None.

    Raises:
        FileNotFoundError: If a required file or directory is missing.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f'Evaluation config not found: {config_path}')

    config = load_yaml(config_path)
    include_path = config.get('include_path')
    if include_path and not os.path.isdir(str(include_path)):
        raise FileNotFoundError(f'lm-eval include_path not found: {include_path}')

    output_path = config.get('output_path')
    if output_path:
        os.makedirs(str(output_path), exist_ok=True)

    model_args = config.get('model_args', {})
    if not isinstance(model_args, dict):
        raise ValueError(f'model_args must be a mapping: {config_path}')

    for key in ('pretrained', 'tokenizer', 'peft'):
        value = model_args.get(key)
        if isinstance(value, str) and looks_like_local_path(value) and not os.path.exists(value):
            raise FileNotFoundError(f'Model argument {key} does not exist: {value}')


def prepare_environment(config: dict[str, Any]) -> None:
    """Prepare environment variables for an evaluation run.

    Args:
        config: Evaluation config mapping.

    Returns:
        None.
    """
    env_file = str(config.get('env_file') or DEFAULT_SECRET_ENV_FILE)
    env = os.environ.copy()
    if os.path.isfile(env_file):
        load_env_file(env_file, env)

    normalize_hf_token_env(env)
    env.setdefault(
        'LM_HARNESS_CACHE_PATH',
        str(config.get('cache_path') or 'outputs/evaluation/cache/requests'),
    )
    env.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.makedirs(env['LM_HARNESS_CACHE_PATH'], exist_ok=True)
    os.environ.update(env)


def serializable_results(results: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of lm-eval results without per-sample records.

    Args:
        results: Raw lm-eval result dictionary.

    Returns:
        Result dictionary suitable for summary storage.
    """
    return {key: value for key, value in results.items() if key != 'samples'}


def write_results(output_path: str, results: dict[str, Any]) -> None:
    """Write lm-eval results under the configured output directory.

    Args:
        output_path: Directory where evaluation artifacts are stored.
        results: Raw lm-eval result dictionary.

    Returns:
        None.
    """
    os.makedirs(output_path, exist_ok=True)
    summary_path = os.path.join(output_path, 'results.json')
    with open(summary_path, 'w', encoding='utf-8') as handle:
        json.dump(
            serializable_results(results),
            handle,
            default=handle_non_serializable,
            ensure_ascii=False,
            indent=2,
        )

    samples = results.get('samples')
    if samples is not None:
        samples_path = os.path.join(output_path, 'samples.json')
        with open(samples_path, 'w', encoding='utf-8') as handle:
            json.dump(
                samples,
                handle,
                default=handle_non_serializable,
                ensure_ascii=False,
                indent=2,
            )


def run_evaluation(config: dict[str, Any]) -> None:
    """Run one lm-evaluation-harness config through the Python API.

    Args:
        config: Evaluation config mapping.

    Returns:
        None.
    """
    task_manager = TaskManager(include_path=config.get('include_path'))
    seed = config.get('seed')
    results = lm_eval.simple_evaluate(
        model=config['model'],
        model_args=config.get('model_args'),
        tasks=config['tasks'],
        num_fewshot=config.get('num_fewshot'),
        batch_size=config.get('batch_size'),
        max_batch_size=config.get('max_batch_size'),
        device=config.get('device'),
        cache_requests=bool(config.get('cache_requests', False)),
        rewrite_requests_cache=bool(config.get('rewrite_requests_cache', False)),
        delete_requests_cache=bool(config.get('delete_requests_cache', False)),
        limit=config.get('limit'),
        bootstrap_iters=int(config.get('bootstrap_iters', 100000)),
        check_integrity=bool(config.get('check_integrity', False)),
        write_out=bool(config.get('write_out', False)),
        log_samples=bool(config.get('log_samples', False)),
        system_instruction=config.get('system_instruction'),
        apply_chat_template=config.get('apply_chat_template', False),
        fewshot_as_multiturn=bool(config.get('fewshot_as_multiturn', True)),
        gen_kwargs=config.get('gen_kwargs'),
        task_manager=task_manager,
        predict_only=bool(config.get('predict_only', False)),
        random_seed=seed,
        numpy_random_seed=seed,
        torch_random_seed=seed,
        fewshot_random_seed=seed,
        confirm_run_unsafe_code=bool(config.get('confirm_run_unsafe_code', False)),
        metadata=config.get('metadata'),
    )

    if results is not None:
        write_results(str(config['output_path']), results)


@click.command(
    context_settings={'show_default': True},
    help='Run one generated lm-evaluation-harness config.',
)
@click.argument('config_path', type=click.Path(exists=True, dir_okay=False))
def main(config_path: str) -> None:
    """Run one evaluation config through lm-evaluation-harness.

    Args:
        config_path: Generated lm-evaluation-harness config path.

    Returns:
        None.
    """
    try:
        require_lm_eval()
        require_evaluation_inputs(config_path)
        config = load_yaml(config_path)
        prepare_environment(config)
        run_evaluation(config)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


if __name__ == '__main__':
    main()
