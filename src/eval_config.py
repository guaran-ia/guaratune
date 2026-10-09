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


def load_lm_eval_task_yaml(path: str) -> dict[str, Any]:
    """Load an lm-eval task YAML file.

    Args:
        path: Task YAML file path.

    Returns:
        Parsed task configuration.
    """
    class Loader(yaml.SafeLoader):
        pass

    Loader.add_constructor(
        '!function', lambda loader, node: loader.construct_scalar(node)
    )
    with open(path, 'r', encoding='utf-8') as handle:
        data = yaml.load(handle, Loader=Loader)

    if not isinstance(data, dict):
        raise ValueError(f'Task YAML file must contain a mapping: {path}')

    return data


def looks_like_local_path(path: str) -> bool:
    """Return whether a model argument points to a local project path.

    Args:
        path: Model argument value.

    Returns:
        True when the value should exist on the local filesystem.
    """
    return path.startswith(('/', './', '../', 'outputs/'))


def iter_data_files(value: Any) -> list[str]:
    """Return dataset file paths from an lm-eval data_files value.

    Args:
        value: data_files config value.

    Returns:
        Flat list of configured dataset file paths.
    """
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        paths = []
        for item in value:
            paths.extend(iter_data_files(item))
        return paths
    if isinstance(value, dict):
        paths = []
        for item in value.values():
            paths.extend(iter_data_files(item))
        return paths

    return []


def missing_local_json_files(task_name: str, include_path: Any) -> list[str]:
    """Return missing local JSON files required by a task.

    Args:
        task_name: lm-eval task name.
        include_path: Local task include directory from the evaluation config.

    Returns:
        Missing local data file paths. Empty when the task is not a local JSON task
        or all required files exist.
    """
    if not include_path:
        return []

    task_path = os.path.join(str(include_path), f'{task_name}.yaml')
    if not os.path.isfile(task_path):
        return []

    task_config = load_lm_eval_task_yaml(task_path)
    if task_config.get('dataset_path') != 'json':
        return []

    dataset_kwargs = task_config.get('dataset_kwargs', {})
    if not isinstance(dataset_kwargs, dict):
        return []

    return [
        path
        for path in iter_data_files(dataset_kwargs.get('data_files'))
        if not os.path.exists(path)
    ]


def available_tasks(config: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    """Filter tasks whose required private local data is unavailable.

    Args:
        config: Evaluation config mapping.

    Returns:
        Task names that can be passed to lm-evaluation-harness and structured
        records for skipped tasks.
    """
    tasks = config.get('tasks', [])
    if not isinstance(tasks, list):
        raise ValueError('tasks must be a list.')

    available = []
    skipped = []
    for task in tasks:
        task_name = str(task)
        missing = missing_local_json_files(task_name, config.get('include_path'))
        if missing:
            missing_text = ', '.join(missing)
            print(
                f'[skip] {task_name}: missing local evaluation data: {missing_text}',
                flush=True,
            )
            skipped.append(
                {
                    'task': task_name,
                    'reason': 'missing_local_evaluation_data',
                    'paths': missing,
                }
            )
            continue

        available.append(task_name)

    return available, skipped


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


def write_results(
    output_path: str,
    results: dict[str, Any],
    *,
    executed_tasks: list[str],
    skipped_tasks: list[dict[str, Any]],
) -> None:
    """Write lm-eval results under the configured output directory.

    Args:
        output_path: Directory where evaluation artifacts are stored.
        results: Raw lm-eval result dictionary.
        executed_tasks: Tasks submitted to lm-evaluation-harness.
        skipped_tasks: Tasks skipped before evaluation with their reasons.

    Returns:
        None.
    """
    os.makedirs(output_path, exist_ok=True)
    summary_path = os.path.join(output_path, 'results.json')
    summary = serializable_results(results)
    summary['guarania_evaluation'] = {
        'status': 'incomplete' if skipped_tasks or not executed_tasks else 'complete',
        'executed_tasks': executed_tasks,
        'skipped_tasks': skipped_tasks,
    }
    with open(summary_path, 'w', encoding='utf-8') as handle:
        json.dump(
            summary,
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


def task_specs_for_evaluation(
    tasks: list[str], task_overrides: Any
) -> list[str | dict[str, Any]]:
    """Attach per-task settings to the task specs passed to lm-eval."""
    if task_overrides is None:
        task_overrides = {}
    if not isinstance(task_overrides, dict):
        raise ValueError("task_overrides must be a mapping.")

    specs: list[str | dict[str, Any]] = []
    for task_name in tasks:
        overrides = task_overrides.get(task_name)
        if overrides is None:
            specs.append(task_name)
            continue
        if not isinstance(overrides, dict):
            raise ValueError(f"task_overrides for {task_name} must be a mapping.")
        invalid = set(overrides) - {
            "num_fewshot",
            "fewshot_split",
            "test_split",
            "metrics",
        }
        if invalid:
            raise ValueError(
                f"Unsupported task overrides for {task_name}: {', '.join(sorted(invalid))}"
            )
        num_fewshot = overrides.get("num_fewshot")
        if num_fewshot is not None and (
            not isinstance(num_fewshot, int)
            or isinstance(num_fewshot, bool)
            or num_fewshot < 0
        ):
            raise ValueError(
                f"num_fewshot for {task_name} must be a non-negative integer."
            )
        metrics = overrides.get("metrics")
        if metrics is not None and (
            not isinstance(metrics, list)
            or not metrics
            or any(not isinstance(metric, str) or not metric for metric in metrics)
            or len(set(metrics)) != len(metrics)
        ):
            raise ValueError(
                f"metrics for {task_name} must be a non-empty list of unique names."
            )
        specs.append({"task": task_name, **overrides})
    return specs


def resolve_task_specs_for_evaluation(
    tasks: list[str], task_overrides: Any, task_manager: Any
) -> list[Any]:
    """Load named tasks and apply matrix overrides to their task objects."""
    task_specs = task_specs_for_evaluation(tasks, task_overrides)
    resolved = []
    for spec in task_specs:
        if isinstance(spec, str):
            resolved.append(spec)
            continue

        task_name = str(spec['task'])
        task = task_manager.load(task_name)['tasks'][task_name]
        if 'num_fewshot' in spec:
            task.set_config('num_fewshot', spec['num_fewshot'])
        if 'test_split' in spec:
            task.set_config('test_split', spec['test_split'])
        if 'fewshot_split' in spec:
            task.set_config('fewshot_split', spec['fewshot_split'])
            task.fewshot_cfg.split = spec['fewshot_split']
        if 'metrics' in spec:
            _restrict_task_metrics(task, spec['metrics'])
        resolved.append(task)
    return resolved


def _restrict_task_metrics(task: Any, selected_metrics: list[str]) -> None:
    """Keep only the configured metrics on an already loaded lm-eval task."""
    metric_maps = (
        '_metric_fn_list',
        '_metric_fn_kwargs',
        '_aggregation_list',
        '_higher_is_better',
    )
    available = set(getattr(task, '_metric_fn_list', {}))
    missing = set(selected_metrics) - available
    if missing:
        raise ValueError(
            f"Configured metrics for {task.config.task} are unavailable: "
            f"{', '.join(sorted(missing))}. Available metrics: "
            f"{', '.join(sorted(available))}."
        )

    for attribute in metric_maps:
        values = getattr(task, attribute)
        setattr(
            task,
            attribute,
            {name: values[name] for name in selected_metrics if name in values},
        )

    configured_metrics = task.config.metric_list
    if configured_metrics is None:
        task.set_config(
            'metric_list', [{'metric': metric} for metric in selected_metrics]
        )
    else:
        def metric_config_name(config: dict[str, Any]) -> str:
            name = config['metric']
            return name if isinstance(name, str) else name.__name__

        task.set_config(
            'metric_list',
            [
                config
                for config in configured_metrics
                if metric_config_name(config) in selected_metrics
            ],
        )



def do_run_evaluation(config: dict[str, Any]) -> None:
    """Run one lm-evaluation-harness config through the Python API.

    Args:
        config: Evaluation config mapping.

    Returns:
        None.
    """
    task_manager = TaskManager(include_path=config.get('include_path'))
    seed = config.get('seed')
    tasks, skipped_tasks = available_tasks(config)
    task_specs = resolve_task_specs_for_evaluation(
        tasks, config.get('task_overrides'), task_manager
    )
    if not tasks:
        print('[skip] no evaluation tasks configured', flush=True)
        write_results(
            str(config['output_path']),
            {
                'results': {},
                'groups': {},
                'metadata': config.get('metadata', {}),
            },
            executed_tasks=[],
            skipped_tasks=skipped_tasks,
        )
        return

    results = lm_eval.simple_evaluate(
        model=config['model'],
        model_args=config.get('model_args'),
        tasks=task_specs,
        num_fewshot=None,
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
        write_results(
            str(config['output_path']),
            results,
            executed_tasks=tasks,
            skipped_tasks=skipped_tasks,
        )


def run_evaluation(config_path: str) -> None:
    """Run one lm-evaluation-harness config through the Python API.

    Args:
        config_path: Path to the evaluation config file.
    """
    require_lm_eval()
    require_evaluation_inputs(config_path)
    config = load_yaml(config_path)
    prepare_environment(config)
    do_run_evaluation(config)


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
        do_run_evaluation(config)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


if __name__ == '__main__':
    main()
