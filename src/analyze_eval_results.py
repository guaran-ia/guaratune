#!/usr/bin/env python3
"""Summarize generated lm-evaluation-harness results."""

from __future__ import annotations

import click
import csv
import json
import math
import os

from collections import defaultdict
from typing import Any


DEFAULT_EVALUATION_ROOT = 'outputs/evaluation'
DEFAULT_RESULTS_DIR = 'results'
SUMMARY_FILE_PREFIX = 'evaluation_summary'
PERPLEXITY_TASK_PREFIX = 'guarani_cpt_perplexity_'
IGNORED_METRIC_SUFFIXES = (
    '_stderr',
    '_stderr,none',
    '_stderr,bootstrap',
)
IGNORED_METRIC_NAMES = (
    'sample_len',
)
LOWER_IS_BETTER_TOKENS = (
    'perplexity',
    'bits_per_byte',
    'loss',
)
IDENTITY_COLUMNS = (
    'model_key',
    'variant',
    'training_method',
    'corpus',
    'eval_corpus',
    'lora_rank',
    'profiles',
)


def load_json(path: str) -> dict[str, Any]:
    """Load a JSON file as a dictionary.

    Args:
        path: JSON file path.

    Returns:
        Parsed JSON dictionary.

    Raises:
        ValueError: If the JSON file does not contain a mapping.
    """
    with open(path, 'r', encoding='utf-8') as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f'JSON file must contain a mapping: {path}')

    return data


def is_number(value: Any) -> bool:
    """Return whether a value is a finite numeric metric.

    Args:
        value: Value to inspect.

    Returns:
        True when the value is a finite int or float.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def is_metric_key(key: str, value: Any) -> bool:
    """Return whether a result key should be included as a metric.

    Args:
        key: Raw metric key from lm-eval.
        value: Raw metric value.

    Returns:
        True when the key/value pair should become a summary column.
    """
    if not is_number(value):
        return False

    if clean_metric_name(key) in IGNORED_METRIC_NAMES:
        return False

    return not any(key.endswith(suffix) for suffix in IGNORED_METRIC_SUFFIXES)


def clean_metric_name(metric_name: str) -> str:
    """Normalize an lm-eval metric name for table columns.

    Args:
        metric_name: Raw lm-eval metric name.

    Returns:
        Clean metric name.
    """
    return metric_name.split(',', 1)[0]


def task_suffix(task_name: str) -> str:
    """Normalize task names for compact metric columns.

    Args:
        task_name: Raw lm-eval task name.

    Returns:
        Compact task name used in metric columns.
    """
    if task_name.startswith(PERPLEXITY_TASK_PREFIX):
        return task_name[len(PERPLEXITY_TASK_PREFIX):]
    if task_name == 'guarani_global_mmlu_lite':
        return 'global_mmlu_lite'

    return task_name


def metric_column(task_name: str, metric_name: str) -> str:
    """Build a summary metric column name.

    Args:
        task_name: Raw lm-eval task name.
        metric_name: Raw lm-eval metric name.

    Returns:
        Summary table column name.
    """
    return f'{task_suffix(task_name)}_{clean_metric_name(metric_name)}'


def normalize_corpus(value: Any) -> str | None:
    """Normalize a corpus/configuration name for matching.

    Args:
        value: Corpus/configuration value.

    Returns:
        Lowercase corpus name, or None for missing values.
    """
    if value is None:
        return None

    normalized = str(value).strip().lower()
    return normalized or None


def is_perplexity_task(task_name: str) -> bool:
    """Return whether a task is one of the held-out perplexity tasks.

    Args:
        task_name: Raw lm-eval task name.

    Returns:
        True when the task is a CPT perplexity task.
    """
    return task_name.startswith(PERPLEXITY_TASK_PREFIX)


def result_metric_column(task_name: str, metric_name: str) -> str:
    """Build the visible summary column for one task metric.

    Args:
        task_name: Raw lm-eval task name.
        metric_name: Clean metric name.

    Returns:
        Summary column name.
    """
    if is_perplexity_task(task_name):
        return metric_name

    return metric_column(task_name, metric_name)


def should_include_perplexity_task(task_name: str, row_corpus: str | int | None) -> bool:
    """Return whether a perplexity task belongs in a result row.

    Args:
        task_name: Raw lm-eval task name.
        row_corpus: Training corpus/configuration for the current row.

    Returns:
        True when the task should be included in the summary.
    """
    normalized_row_corpus = normalize_corpus(row_corpus)
    if normalized_row_corpus is None:
        return True

    return normalized_row_corpus == normalize_corpus(task_suffix(task_name))


def base_result_row(
    model_key: str, variant: str, profile: str, parsed_variant: dict[str, str | int | None]
) -> dict[str, Any]:
    """Build common identity fields for an evaluation result.

    Args:
        model_key: Model key from result metadata or output path.
        variant: Model variant from result metadata or output path.
        profile: Evaluation profile from result metadata or output path.
        parsed_variant: Parsed variant fields.

    Returns:
        Result row with identity fields populated.
    """
    return {
        'model_key': model_key,
        'variant': variant,
        'profiles': profile,
        'training_method': parsed_variant['training_method'],
        'corpus': parsed_variant['corpus'],
        'eval_corpus': None,
        'lora_rank': parsed_variant['lora_rank'],
    }


def discover_result_paths(
    evaluation_root: str, model_keys: tuple[str, ...], profile_names: tuple[str, ...]
) -> list[str]:
    """Find result files for requested model keys.

    Args:
        evaluation_root: Evaluation output root directory.
        model_keys: Model keys to include.
        profile_names: Optional profiles to include.

    Returns:
        Sorted result JSON paths.
    """
    paths = []
    if not os.path.isdir(evaluation_root):
        return paths

    requested = set(model_keys)
    requested_profiles = set(profile_names)
    for root, _, filenames in os.walk(evaluation_root):
        if 'results.json' not in filenames:
            continue

        result_path = os.path.join(root, 'results.json')
        rel_parts = os.path.relpath(result_path, evaluation_root).split(os.sep)
        if len(rel_parts) < 4 or rel_parts[1] not in requested:
            continue
        if requested_profiles and rel_parts[0] not in requested_profiles:
            continue

        if len(rel_parts) >= 4:
            paths.append(result_path)

    return sorted(paths)


def metadata_from_path(result_path: str, evaluation_root: str) -> dict[str, str | None]:
    """Extract profile, model, variant, and suite from a result path.

    Args:
        result_path: Result JSON path.
        evaluation_root: Evaluation output root directory.

    Returns:
        Metadata inferred from the output path.
    """
    rel_parts = os.path.relpath(result_path, evaluation_root).split(os.sep)
    metadata = {
        'profile': None,
        'model_key': None,
        'variant': None,
        'suite': None,
    }
    if len(rel_parts) >= 5:
        metadata['profile'] = rel_parts[0]
        metadata['model_key'] = rel_parts[1]
        metadata['variant'] = rel_parts[2]
        metadata['suite'] = rel_parts[3]

    return metadata


def parse_variant(variant: str | None) -> dict[str, str | int | None]:
    """Parse training method, corpus, and LoRA rank from a variant name.

    Args:
        variant: Variant name such as base, full_C1_kuatia, or lora_r64_C1_kuatia.

    Returns:
        Parsed variant fields.
    """
    if variant is None or variant == 'base':
        return {
            'training_method': 'base',
            'corpus': None,
            'lora_rank': None,
        }
    if variant.startswith('full_'):
        return {
            'training_method': 'full',
            'corpus': variant[len('full_'):],
            'lora_rank': None,
        }
    if variant.startswith('lora_r'):
        parts = variant.split('_', 2)
        if len(parts) == 3 and parts[1].startswith('r'):
            return {
                'training_method': 'lora',
                'corpus': parts[2],
                'lora_rank': int(parts[1][1:]),
            }

    return {
        'training_method': variant,
        'corpus': None,
        'lora_rank': None,
    }


def extract_result_records(result_path: str, evaluation_root: str) -> list[dict[str, Any]]:
    """Extract partial rows from an lm-eval result file.

    Args:
        result_path: Result JSON path.
        evaluation_root: Evaluation output root directory.

    Returns:
        Partial rows containing identity fields and metrics.
    """
    data = load_json(result_path)
    path_metadata = metadata_from_path(result_path, evaluation_root)
    result_metadata = data.get('metadata', {})
    if not isinstance(result_metadata, dict):
        result_metadata = {}

    model_key = str(result_metadata.get('model_key') or path_metadata['model_key'])
    variant = str(result_metadata.get('variant') or path_metadata['variant'])
    profile = str(result_metadata.get('profile') or path_metadata['profile'])
    parsed_variant = parse_variant(variant)
    row = base_result_row(model_key, variant, profile, parsed_variant)

    results = data.get('results', {})
    if not isinstance(results, dict):
        return [row]

    global_metrics = {}
    perplexity_rows: dict[str, dict[str, Any]] = {}
    for task_name, task_metrics in results.items():
        if not isinstance(task_metrics, dict):
            continue

        task_name = str(task_name)
        if is_perplexity_task(task_name):
            if not should_include_perplexity_task(task_name, row.get('corpus')):
                continue

            eval_corpus = task_suffix(task_name)
            task_row = perplexity_rows.setdefault(eval_corpus, row | {'eval_corpus': eval_corpus})
        else:
            task_row = global_metrics

        for raw_metric, value in task_metrics.items():
            if is_metric_key(raw_metric, value):
                metric_name = clean_metric_name(raw_metric)
                task_row[result_metric_column(task_name, metric_name)] = value

    if perplexity_rows:
        for task_row in perplexity_rows.values():
            task_row.update(global_metrics)

        return list(perplexity_rows.values())

    row.update(global_metrics)
    return [row]


def row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Build the grouping key for a model variant row.

    Args:
        row: Partial result row.

    Returns:
        Tuple key identifying a model variant.
    """
    return (
        row.get('model_key'),
        row.get('variant'),
        row.get('training_method'),
        row.get('corpus'),
        row.get('eval_corpus'),
        row.get('lora_rank'),
    )


def variant_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Build the grouping key for all rows from one model variant.

    Args:
        row: Summary row.

    Returns:
        Tuple key identifying a model variant across evaluation corpora.
    """
    return (
        row.get('model_key'),
        row.get('variant'),
        row.get('training_method'),
        row.get('corpus'),
        row.get('lora_rank'),
    )


def metric_values(row: dict[str, Any]) -> dict[str, Any]:
    """Return metric values from a row.

    Args:
        row: Summary row.

    Returns:
        Mapping of metric columns to values.
    """
    return {
        column: value
        for column, value in row.items()
        if column not in IDENTITY_COLUMNS
    }


def merge_profile_values(*values: Any) -> str:
    """Merge comma-separated profile values.

    Args:
        values: Profile values to merge.

    Returns:
        Sorted, comma-separated profile names.
    """
    profiles = set()
    for value in values:
        if value is None:
            continue

        profiles.update(part for part in str(value).split(',') if part)

    return ','.join(sorted(profiles))


def collapse_global_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy global metrics onto corpus rows and remove redundant global rows.

    Args:
        rows: Merged rows.

    Returns:
        Rows with global-only entries collapsed where corpus rows exist.
    """
    global_by_variant = {}
    variants_with_eval_corpus = set()
    for row in rows:
        key = variant_key(row)
        if row.get('eval_corpus') is None:
            global_by_variant[key] = row
        else:
            variants_with_eval_corpus.add(key)

    collapsed = []
    for row in rows:
        key = variant_key(row)
        if row.get('eval_corpus') is None and key in variants_with_eval_corpus:
            continue

        global_row = global_by_variant.get(key)
        if row.get('eval_corpus') is not None and global_row is not None:
            for column, value in metric_values(global_row).items():
                row.setdefault(column, value)

            row['profiles'] = merge_profile_values(row.get('profiles'), global_row.get('profiles'))

        collapsed.append(row)

    return collapsed


def merge_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge partial suite rows into one row per model variant.

    Args:
        rows: Partial result rows.

    Returns:
        Merged rows.
    """
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    profiles_by_key: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    for row in rows:
        key = row_key(row)
        if key not in grouped:
            grouped[key] = {column: row.get(column) for column in IDENTITY_COLUMNS}

        profile = row.get('profiles')
        if profile:
            profiles_by_key[key].add(str(profile))

        for column, value in row.items():
            if column in IDENTITY_COLUMNS:
                continue
            grouped[key].setdefault(column, value)

    merged = []
    for key, row in grouped.items():
        row['profiles'] = ','.join(sorted(profiles_by_key[key]))
        merged.append(row)

    merged = collapse_global_rows(merged)
    return sorted(
        merged,
        key=lambda item: (
            str(item.get('model_key') or ''),
            str(item.get('training_method') or ''),
            str(item.get('lora_rank') or ''),
            str(item.get('corpus') or ''),
            str(item.get('eval_corpus') or ''),
            str(item.get('variant') or ''),
        ),
    )


def metric_columns(rows: list[dict[str, Any]]) -> list[str]:
    """List metric columns found in result rows.

    Args:
        rows: Summary rows.

    Returns:
        Sorted metric column names.
    """
    columns = set()
    for row in rows:
        columns.update(
            column
            for column in row.keys()
            if column not in IDENTITY_COLUMNS
        )

    return sorted(columns)


def higher_is_better(metric_name: str) -> bool:
    """Return whether larger values indicate improvement for a metric.

    Args:
        metric_name: Summary metric column name.

    Returns:
        True for accuracy-like metrics, false for perplexity/loss-like metrics.
    """
    lowered = metric_name.lower()
    return not any(token in lowered for token in LOWER_IS_BETTER_TOKENS)


def percentage_improvement(value: Any, base_value: Any, metric_name: str) -> float | None:
    """Compute percentage improvement versus a base value.

    Args:
        value: Variant metric value.
        base_value: Base model metric value.
        metric_name: Metric column name.

    Returns:
        Percentage improvement, or None when it cannot be computed.
    """
    if not is_number(value) or not is_number(base_value) or base_value == 0:
        return None
    if higher_is_better(metric_name):
        return ((float(value) - float(base_value)) / float(base_value)) * 100

    return ((float(base_value) - float(value)) / float(base_value)) * 100


def base_metric_value(
    base_rows: list[dict[str, Any]], row: dict[str, Any], metric: str
) -> Any:
    """Find the relevant base metric value for a row.

    Args:
        base_rows: Base rows for the same model.
        row: Variant row.
        metric: Visible metric column.

    Returns:
        Base metric value when available.
    """
    preferred_eval_corpus = row.get('eval_corpus')
    for base in base_rows:
        if base.get('eval_corpus') == preferred_eval_corpus and is_number(base.get(metric)):
            return base.get(metric)

    for base in base_rows:
        if is_number(base.get(metric)):
            return base.get(metric)

    return None


def add_improvement_columns(rows: list[dict[str, Any]], metrics: list[str]) -> list[str]:
    """Add base-relative improvement columns to summary rows.

    Args:
        rows: Summary rows modified in place.
        metrics: Metric columns.

    Returns:
        Improvement column names.
    """
    base_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get('variant') == 'base':
            base_by_model[str(row['model_key'])].append(row)

    improvement_columns = [f'{metric}_improvement_vs_base_pct' for metric in metrics]
    for row in rows:
        base_rows = base_by_model.get(str(row.get('model_key')))
        for metric in metrics:
            column = f'{metric}_improvement_vs_base_pct'
            if not base_rows or row.get('variant') == 'base':
                row[column] = None
                continue

            row[column] = percentage_improvement(
                row.get(metric), base_metric_value(base_rows, row, metric), metric
            )

    return improvement_columns


def format_value(value: Any) -> str:
    """Format a value for markdown output.

    Args:
        value: Raw table value.

    Returns:
        Markdown-safe string representation.
    """
    if value is None:
        return ''
    if isinstance(value, float):
        return f'{value:.6g}'

    return str(value)


def write_csv(path: str, rows: list[dict[str, Any]], columns: list[str]) -> None:
    """Write summary rows to CSV.

    Args:
        path: Destination CSV path.
        rows: Summary rows.
        columns: Output columns.

    Returns:
        None.
    """
    with open(path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def write_markdown(path: str, rows: list[dict[str, Any]], columns: list[str]) -> None:
    """Write summary rows to a markdown table.

    Args:
        path: Destination markdown path.
        rows: Summary rows.
        columns: Output columns.

    Returns:
        None.
    """
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('| ' + ' | '.join(columns) + ' |\n')
        handle.write('| ' + ' | '.join(['---'] * len(columns)) + ' |\n')
        for row in rows:
            values = [format_value(row.get(column)).replace('|', '\\|') for column in columns]
            handle.write('| ' + ' | '.join(values) + ' |\n')


def output_name_suffix(model_keys: tuple[str, ...]) -> str:
    """Build the default output filename suffix for analyzed models.

    Args:
        model_keys: Model keys passed to the analyzer.

    Returns:
        Model key for one model, or a combined model-key label for multiple models.
    """
    if len(model_keys) == 1:
        return model_keys[0]

    return 'combined_' + '_'.join(model_keys)


def default_output_name(model_keys: tuple[str, ...], extension: str) -> str:
    """Build a default output filename.

    Args:
        model_keys: Model keys passed to the analyzer.
        extension: File extension without leading dot.

    Returns:
        Default model-scoped output filename.
    """
    return f'{SUMMARY_FILE_PREFIX}_{output_name_suffix(model_keys)}.{extension}'


def analyze_results(
    evaluation_root: str, model_keys: tuple[str, ...], profile_names: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Analyze evaluation results for requested models.

    Args:
        evaluation_root: Evaluation output root directory.
        model_keys: Model keys to analyze.
        profile_names: Optional profiles to analyze.

    Returns:
        One summary row per model variant and evaluated corpus.
    """
    partial_rows = []
    for path in discover_result_paths(evaluation_root, model_keys, profile_names):
        partial_rows.extend(extract_result_records(path, evaluation_root))

    return merge_rows(partial_rows)


@click.command(
    context_settings={'show_default': True},
    help='Summarize evaluation results into CSV and markdown tables.',
)
@click.option(
    '--model',
    'model_keys',
    multiple=True,
    required=True,
    help='Model key to analyze. Repeat to analyze multiple models.',
)
@click.option(
    '--evaluation-root',
    default=DEFAULT_EVALUATION_ROOT,
    type=click.Path(file_okay=False),
    help='Evaluation output root directory.',
)
@click.option(
    '--profile',
    'profile_names',
    multiple=True,
    help='Evaluation profile to include. Repeat to include multiple profiles. Defaults to all profiles.',
)
@click.option(
    '--output-dir',
    default=DEFAULT_RESULTS_DIR,
    type=click.Path(file_okay=False),
    help='Directory where analysis tables are written.',
)
@click.option(
    '--csv-name',
    default=None,
    help='CSV output filename.',
)
@click.option(
    '--markdown-name',
    default=None,
    help='Markdown output filename.',
)
def main(
    model_keys: tuple[str, ...],
    evaluation_root: str,
    profile_names: tuple[str, ...],
    output_dir: str,
    csv_name: str | None,
    markdown_name: str | None,
) -> None:
    """Summarize available evaluation results.

    Args:
        model_keys: Model keys to analyze.
        evaluation_root: Evaluation output root directory.
        profile_names: Evaluation profiles to analyze.
        output_dir: Analysis output directory.
        csv_name: Optional CSV output filename.
        markdown_name: Optional markdown output filename.

    Returns:
        None.
    """
    rows = analyze_results(evaluation_root, model_keys, profile_names)
    if not rows:
        raise click.ClickException(
            f'No evaluation results found for models: {", ".join(model_keys)}'
        )

    metrics = metric_columns(rows)
    improvement_columns = add_improvement_columns(rows, metrics)
    columns = list(IDENTITY_COLUMNS) + metrics + improvement_columns
    os.makedirs(output_dir, exist_ok=True)
    resolved_csv_name = csv_name or default_output_name(model_keys, 'csv')
    resolved_markdown_name = markdown_name or default_output_name(model_keys, 'md')
    csv_path = os.path.join(output_dir, resolved_csv_name)
    markdown_path = os.path.join(output_dir, resolved_markdown_name)
    write_csv(csv_path, rows, columns)
    write_markdown(markdown_path, rows, columns)
    print(csv_path)
    print(markdown_path)
    print(f'[done] wrote {len(rows)} row(s)')


if __name__ == '__main__':
    main()
