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
BENCHMARK_TABLE_FILE_PREFIX = 'evaluation_benchmark_table'
LANGUAGE_AVERAGE_TABLE_FILE_PREFIX = 'evaluation_language_average_table'
PERPLEXITY_TABLE_FILE_PREFIX = 'evaluation_perplexity_table'
PERPLEXITY_TASK_PREFIX = 'guarani_cpt_perplexity_'
BENCHMARK_LANGUAGE_CHOICES = (
    'all',
    'en',
    'es',
    'gn',
)
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
    'evaluation_status',
    'skipped_tasks',
)
BENCHMARK_SCORE_SPECS = (
    {
        'label': 'Global MMLU Lite GN (acc_norm)',
        'languages': ('gn',),
        'metrics': ('global_mmlu_lite_acc_norm', 'global_mmlu_lite_acc'),
        'scale': 100.0,
    },
    {
        'label': 'Global MMLU Lite EN (acc_norm)',
        'languages': ('en',),
        'metrics': ('english_global_mmlu_lite_acc_norm', 'english_global_mmlu_lite_acc'),
        'scale': 100.0,
    },
    {
        'label': 'Global MMLU Lite ES (acc_norm)',
        'languages': ('es',),
        'metrics': ('spanish_global_mmlu_lite_acc_norm', 'spanish_global_mmlu_lite_acc'),
        'scale': 100.0,
    },
    {
        'label': 'Belebele GN (acc_norm)',
        'languages': ('gn',),
        'metrics': ('guarani_2m_belebele_acc_norm', 'guarani_2m_belebele_acc'),
        'scale': 100.0,
    },
    {
        'label': 'Belebele EN (acc_norm)',
        'languages': ('en',),
        'metrics': ('english_2m_belebele_acc_norm', 'english_2m_belebele_acc'),
        'scale': 100.0,
    },
    {
        'label': 'Belebele ES (acc_norm)',
        'languages': ('es',),
        'metrics': ('spanish_2m_belebele_acc_norm', 'spanish_2m_belebele_acc'),
        'scale': 100.0,
    },
    {
        'label': 'MultiWikiQA GN (F1)',
        'languages': ('gn',),
        'metrics': ('guarani_multiwikiqa_f1', 'guarani_multiwikiqa_exact_match'),
        'scale': 100.0,
    },
    {
        'label': 'MultiWikiQA EN (F1)',
        'languages': ('en',),
        'metrics': ('english_multiwikiqa_f1', 'english_multiwikiqa_exact_match'),
        'scale': 100.0,
    },
    {
        'label': 'MultiWikiQA ES (F1)',
        'languages': ('es',),
        'metrics': ('spanish_multiwikiqa_f1', 'spanish_multiwikiqa_exact_match'),
        'scale': 100.0,
    },
    {
        'label': 'FLORES-200 EN->GN (BLEU)',
        'languages': ('en', 'gn'),
        'metrics': ('guarani_flores200_eng_to_grn_bleu',),
        'scale': 1.0,
    },
    {
        'label': 'FLORES-200 EN->GN (chrF++)',
        'languages': ('en', 'gn'),
        'metrics': ('guarani_flores200_eng_to_grn_chrf_plus_plus',),
        'scale': 1.0,
    },
    {
        'label': 'FLORES-200 GN->EN (BLEU)',
        'languages': ('gn', 'en'),
        'metrics': ('guarani_flores200_grn_to_eng_bleu',),
        'scale': 1.0,
    },
    {
        'label': 'FLORES-200 GN->EN (chrF++)',
        'languages': ('gn', 'en'),
        'metrics': ('guarani_flores200_grn_to_eng_chrf_plus_plus',),
        'scale': 1.0,
    },
    {
        'label': 'FLORES-200 ES->GN (BLEU)',
        'languages': ('es', 'gn'),
        'metrics': ('guarani_flores200_spa_to_grn_bleu',),
        'scale': 1.0,
    },
    {
        'label': 'FLORES-200 ES->GN (chrF++)',
        'languages': ('es', 'gn'),
        'metrics': ('guarani_flores200_spa_to_grn_chrf_plus_plus',),
        'scale': 1.0,
    },
    {
        'label': 'FLORES-200 GN->ES (BLEU)',
        'languages': ('gn', 'es'),
        'metrics': ('guarani_flores200_grn_to_spa_bleu',),
        'scale': 1.0,
    },
    {
        'label': 'FLORES-200 GN->ES (chrF++)',
        'languages': ('gn', 'es'),
        'metrics': ('guarani_flores200_grn_to_spa_chrf_plus_plus',),
        'scale': 1.0,
    },
    {
        'label': 'ARC Easy EN (acc_norm)',
        'languages': ('en',),
        'metrics': ('arc_easy_acc_norm', 'arc_easy_acc'),
        'scale': 100.0,
    },
    {
        'label': 'ARC Challenge EN (acc_norm)',
        'languages': ('en',),
        'metrics': ('arc_challenge_acc_norm', 'arc_challenge_acc'),
        'scale': 100.0,
    },
    {
        'label': 'PIQA EN (acc_norm)',
        'languages': ('en',),
        'metrics': ('piqa_acc_norm', 'piqa_acc'),
        'scale': 100.0,
    },
    {
        'label': 'HellaSwag EN (acc_norm)',
        'languages': ('en',),
        'metrics': ('hellaswag_acc_norm', 'hellaswag_acc'),
        'scale': 100.0,
    },
    {
        'label': 'WinoGrande EN (acc)',
        'languages': ('en',),
        'metrics': ('winogrande_acc',),
        'scale': 100.0,
    },
    {
        'label': 'XNLI EN (acc)',
        'languages': ('en',),
        'metrics': ('xnli_en_acc',),
        'scale': 100.0,
    },
    {
        'label': 'XStoryCloze EN (acc)',
        'languages': ('en',),
        'metrics': ('xstorycloze_en_acc',),
        'scale': 100.0,
    },
    {
        'label': 'MGSM EN (exact_match)',
        'languages': ('en',),
        'metrics': ('mgsm_direct_en_exact_match',),
        'scale': 100.0,
    },
    {
        'label': 'BBH EN (exact_match)',
        'languages': ('en',),
        'metrics': ('bbh_exact_match',),
        'scale': 100.0,
    },
    {
        'label': 'GPQA Diamond EN (acc_norm)',
        'languages': ('en',),
        'metrics': ('gpqa_diamond_zeroshot_acc_norm', 'gpqa_diamond_zeroshot_acc'),
        'scale': 100.0,
    },
    {
        'label': 'TruthfulQA-MC1 EN (acc)',
        'languages': ('en',),
        'metrics': ('truthfulqa_mc1_acc',),
        'scale': 100.0,
    },
    {
        'label': 'HumanEval EN (pass@1)',
        'languages': ('en',),
        'metrics': ('humaneval_pass@1',),
        'scale': 100.0,
    },
    {
        'label': 'IFEval EN (prompt strict acc)',
        'languages': ('en',),
        'metrics': ('ifeval_prompt_level_strict_acc',),
        'scale': 100.0,
    },
    {
        'label': 'ARC Easy ES (acc_norm)',
        'languages': ('es',),
        'metrics': ('spanish_arc_easy_acc_norm', 'spanish_arc_easy_acc'),
        'scale': 100.0,
    },
    {
        'label': 'ARC Challenge ES (acc_norm)',
        'languages': ('es',),
        'metrics': ('spanish_arc_challenge_acc_norm', 'spanish_arc_challenge_acc'),
        'scale': 100.0,
    },
    {
        'label': 'PIQA ES (acc_norm)',
        'languages': ('es',),
        'metrics': (
            'global_piqa_nonparallel_cloze_spa_latn_spai_acc_norm',
            'global_piqa_nonparallel_cloze_spa_latn_spai_acc',
        ),
        'scale': 100.0,
    },
    {
        'label': 'HellaSwag ES (acc_norm)',
        'languages': ('es',),
        'metrics': ('hellaswag_es_acc_norm', 'hellaswag_es_acc'),
        'scale': 100.0,
    },
    {
        'label': 'COPA ES (acc)',
        'languages': ('es',),
        'metrics': ('copa_es_acc',),
        'scale': 100.0,
    },
    {
        'label': 'XStoryCloze ES (acc)',
        'languages': ('es',),
        'metrics': ('xstorycloze_es_acc',),
        'scale': 100.0,
    },
    {
        'label': 'TruthfulQA-MC1 ES (acc)',
        'languages': ('es',),
        'metrics': ('truthfulqa_es_mc1_acc',),
        'scale': 100.0,
    },
    {
        'label': 'XNLI ES (acc)',
        'languages': ('es',),
        'metrics': ('xnli_es_acc', 'xnli_es_spanish_bench_acc'),
        'scale': 100.0,
    },
    {
        'label': 'MGSM ES (exact_match)',
        'languages': ('es',),
        'metrics': (
            'mgsm_direct_es_spanish_bench_exact_match',
            'mgsm_direct_es_exact_match',
        ),
        'scale': 100.0,
    },
    {
        'label': 'IFEval ES (prompt strict acc)',
        'languages': ('es',),
        'metrics': ('ifeval_es_prompt_level_strict_acc',),
        'scale': 100.0,
    },
    {
        'label': 'GPQA Diamond ES (acc_norm)',
        'languages': ('es',),
        'metrics': ('spanish_gpqa_diamond_acc_norm', 'spanish_gpqa_diamond_acc'),
        'scale': 100.0,
    },
)
PERPLEXITY_METRICS = (
    ('word_perplexity', 'word perplexity'),
    ('byte_perplexity', 'byte perplexity'),
    ('bits_per_byte', 'bits per byte'),
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
        'evaluation_status': 'complete',
        'skipped_tasks': '',
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
    evaluation = data.get('guarania_evaluation', {})
    if isinstance(evaluation, dict):
        row['evaluation_status'] = str(evaluation.get('status') or 'complete')
        skipped_tasks = evaluation.get('skipped_tasks', [])
        if isinstance(skipped_tasks, list):
            row['skipped_tasks'] = ','.join(
                str(task.get('task'))
                for task in skipped_tasks
                if isinstance(task, dict) and task.get('task')
            )

    results = data.get('results', {})
    groups = data.get('groups', {})
    if not isinstance(results, dict):
        results = {}
    if not isinstance(groups, dict):
        groups = {}
    if not results and not groups:
        return [row]

    global_metrics = {}
    perplexity_rows: dict[str, dict[str, Any]] = {}
    for task_name, task_metrics in (results | groups).items():
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
            if global_row.get('evaluation_status') == 'incomplete':
                row['evaluation_status'] = 'incomplete'
            row['skipped_tasks'] = merge_profile_values(
                row.get('skipped_tasks'), global_row.get('skipped_tasks')
            )

        collapsed.append(row)

    return collapsed


def merge_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge partial suite rows into one row per model variant.

    Args:
        rows: Partial result rows.

    Returns:
        Merged rows.

    Raises:
        ValueError: If results for the same model variant report different values
            for the same metric.
    """
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    profiles_by_key: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    statuses_by_key: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    skipped_tasks_by_key: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    for row in rows:
        key = row_key(row)
        if key not in grouped:
            grouped[key] = {column: row.get(column) for column in IDENTITY_COLUMNS}

        profile = row.get('profiles')
        if profile:
            profiles_by_key[key].add(str(profile))
        status = row.get('evaluation_status')
        if status:
            statuses_by_key[key].add(str(status))
        skipped = row.get('skipped_tasks')
        if skipped:
            skipped_tasks_by_key[key].update(
                task for task in str(skipped).split(',') if task
            )

        for column, value in row.items():
            if column in IDENTITY_COLUMNS:
                continue
            existing = grouped[key].get(column)
            if existing is None:
                grouped[key][column] = value
            elif value is not None and existing != value:
                raise ValueError(
                    f'Conflicting metric {column} for {key}: '
                    f'{existing!r} != {value!r}. Analyze different evaluation '
                    'configurations separately.'
                )

    merged = []
    for key, row in grouped.items():
        row['profiles'] = ','.join(sorted(profiles_by_key[key]))
        row['evaluation_status'] = (
            'incomplete' if 'incomplete' in statuses_by_key[key] else 'complete'
        )
        row['skipped_tasks'] = ','.join(sorted(skipped_tasks_by_key[key]))
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


def model_column_label(row: dict[str, Any]) -> str:
    """Build a display label for one model variant column.

    Args:
        row: Summary row.

    Returns:
        Human-readable model variant label.
    """
    return f'{row.get("model_key")}/{row.get("variant")}'


def first_numeric_metric(row: dict[str, Any], metric_names: tuple[str, ...]) -> Any:
    """Return the first numeric metric value found in a row.

    Args:
        row: Summary row.
        metric_names: Candidate metric columns in preference order.

    Returns:
        First finite numeric value, or None when no candidate is available.
    """
    for metric_name in metric_names:
        value = row.get(metric_name)
        if is_number(value):
            return value

    return None


def unique_variant_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse duplicate rows for the same model variant.

    Perplexity-specific rows can duplicate global benchmark metrics across
    `eval_corpus` values. This function keeps one row per model variant and
    copies the first available value for each metric.

    Args:
        rows: Summary rows.

    Returns:
        One merged row per model variant.
    """
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    order = []
    statuses_by_key: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    skipped_tasks_by_key: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    for row in rows:
        key = variant_key(row)
        if key not in grouped:
            grouped[key] = {column: row.get(column) for column in IDENTITY_COLUMNS}
            order.append(key)

        status = row.get('evaluation_status')
        if status:
            statuses_by_key[key].add(str(status))
        skipped = row.get('skipped_tasks')
        if skipped:
            skipped_tasks_by_key[key].update(
                task for task in str(skipped).split(',') if task
            )

        for column, value in metric_values(row).items():
            if grouped[key].get(column) is None:
                grouped[key][column] = value

    for key in order:
        grouped[key]['evaluation_status'] = (
            'incomplete' if 'incomplete' in statuses_by_key[key] else 'complete'
        )
        grouped[key]['skipped_tasks'] = ','.join(sorted(skipped_tasks_by_key[key]))

    return [grouped[key] for key in order]


def benchmark_score(row: dict[str, Any], spec: dict[str, Any]) -> float | None:
    """Read and scale one benchmark score for a model variant.

    Args:
        row: Summary row.
        spec: Benchmark score specification.

    Returns:
        Score on a 0-100 scale, or None when unavailable.
    """
    value = first_numeric_metric(row, spec['metrics'])
    if value is None:
        return None

    return float(value) * float(spec['scale'])


def format_benchmark_score(value: Any) -> str:
    """Format one benchmark table score.

    Args:
        value: Numeric score or None.

    Returns:
        Score rounded to two decimals, or an empty string.
    """
    if value is None:
        return ''

    return f'{float(value):.2f}'


def benchmark_specs_for_language(language: str) -> list[dict[str, Any]]:
    """Filter benchmark score specs by language.

    Args:
        language: Language code to include, or all.

    Returns:
        Benchmark score specs matching the selected language.
    """
    if language == 'all':
        return list(BENCHMARK_SCORE_SPECS)

    return [
        spec
        for spec in BENCHMARK_SCORE_SPECS
        if language in spec.get('languages', ())
    ]


def benchmark_table(
    rows: list[dict[str, Any]], language: str
) -> tuple[list[dict[str, Any]], list[str]]:
    """Build a benchmark-by-model markdown table.

    Args:
        rows: Summary rows.
        language: Language code to include, or all.

    Returns:
        Table rows and column names.
    """
    variants = unique_variant_rows(rows)
    model_labels = [model_column_label(row) for row in variants]
    table_rows = []
    scores_by_model: dict[str, list[float]] = {label: [] for label in model_labels}

    for spec in benchmark_specs_for_language(language):
        table_row = {'benchmark': spec['label']}
        has_score = False
        for variant_row, model_label in zip(variants, model_labels):
            score = benchmark_score(variant_row, spec)
            table_row[model_label] = score
            if score is not None:
                scores_by_model[model_label].append(score)
                has_score = True

        if has_score:
            table_rows.append(table_row)

    average_row = {'benchmark': 'Average'}
    for model_label, scores in scores_by_model.items():
        average_row[model_label] = sum(scores) / len(scores) if scores else None

    table_rows.append(average_row)
    return table_rows, ['benchmark'] + model_labels


def language_average_score(row: dict[str, Any], language: str) -> float | None:
    """Compute the average benchmark score for one language.

    Args:
        row: Summary row.
        language: Language code to average.

    Returns:
        Average score on a 0-100 scale, or None when unavailable.
    """
    scores = [
        score
        for spec in benchmark_specs_for_language(language)
        if (score := benchmark_score(row, spec)) is not None
    ]
    if not scores:
        return None

    return sum(scores) / len(scores)


def format_gain(value: Any, is_base: bool) -> str:
    """Format one base-relative score gain.

    Args:
        value: Score-point gain.
        is_base: Whether the row is the base variant.

    Returns:
        Formatted gain, `--` for base rows, or empty string for missing values.
    """
    if is_base:
        return '--'
    if value is None:
        return ''

    return f'{float(value):+.2f}'


def language_average_table(
    rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Build a language-average benchmark table by model variant.

    Args:
        rows: Summary rows.

    Returns:
        Table rows and column names.
    """
    variants = unique_variant_rows(rows)
    base_by_model = {
        str(row.get('model_key')): row
        for row in variants
        if row.get('variant') == 'base'
    }
    table_rows = []

    for row in variants:
        model_key = str(row.get('model_key'))
        base_row = base_by_model.get(model_key)
        is_base = row.get('variant') == 'base'
        table_row = {'model': model_column_label(row)}

        for language, prefix in (
            ('gn', 'guarani'),
            ('en', 'english'),
            ('es', 'spanish'),
        ):
            score = language_average_score(row, language)
            base_score = (
                language_average_score(base_row, language)
                if base_row is not None
                else None
            )
            gain = (
                score - base_score
                if score is not None and base_score is not None
                else None
            )
            table_row[f'{prefix}_average'] = score
            table_row[f'{prefix}_gain_vs_base'] = gain
            table_row[f'{prefix}_is_base'] = is_base

        table_rows.append(table_row)

    return table_rows, [
        'model',
        'guarani_average',
        'guarani_gain_vs_base',
        'english_average',
        'english_gain_vs_base',
        'spanish_average',
        'spanish_gain_vs_base',
    ]


def write_language_average_markdown(
    path: str, rows: list[dict[str, Any]], columns: list[str]
) -> None:
    """Write a rounded language-average benchmark markdown table.

    Args:
        path: Destination markdown path.
        rows: Language-average table rows.
        columns: Output columns.

    Returns:
        None.
    """
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('| ' + ' | '.join(columns) + ' |\n')
        handle.write('| ' + ' | '.join(['---'] * len(columns)) + ' |\n')
        for row in rows:
            values = []
            for column in columns:
                if column == 'model':
                    value = str(row.get(column, '')).replace('|', '\\|')
                elif column.endswith('_gain_vs_base'):
                    prefix = column.removesuffix('_gain_vs_base')
                    value = format_gain(row.get(column), bool(row.get(f'{prefix}_is_base')))
                else:
                    value = format_benchmark_score(row.get(column))

                values.append(value)

            handle.write('| ' + ' | '.join(values) + ' |\n')


def write_benchmark_markdown(
    path: str, rows: list[dict[str, Any]], columns: list[str]
) -> None:
    """Write a rounded benchmark-by-model markdown table.

    Args:
        path: Destination markdown path.
        rows: Benchmark table rows.
        columns: Output columns.

    Returns:
        None.
    """
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('| ' + ' | '.join(columns) + ' |\n')
        handle.write('| ' + ' | '.join(['---'] * len(columns)) + ' |\n')
        for row in rows:
            values = []
            for column in columns:
                if column == 'benchmark':
                    value = str(row.get(column, '')).replace('|', '\\|')
                else:
                    value = format_benchmark_score(row.get(column))

                values.append(value)

            handle.write('| ' + ' | '.join(values) + ' |\n')


def sorted_variant_labels(rows: list[dict[str, Any]]) -> list[str]:
    """List model variant labels in stable row order.

    Args:
        rows: Summary rows.

    Returns:
        Unique model variant labels.
    """
    labels = []
    seen = set()
    for row in rows:
        label = model_column_label(row)
        if label not in seen:
            labels.append(label)
            seen.add(label)

    return labels


def perplexity_metric_label(eval_corpus: Any, metric_label: str) -> str:
    """Build the first-column label for a perplexity table row.

    Args:
        eval_corpus: Evaluation corpus name.
        metric_label: Human-readable metric label.

    Returns:
        Perplexity table row label.
    """
    if eval_corpus is None:
        return metric_label

    return f'{eval_corpus} {metric_label}'


def perplexity_table(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Build a perplexity-by-model markdown table.

    Args:
        rows: Summary rows.

    Returns:
        Table rows and column names.
    """
    model_labels = sorted_variant_labels(rows)
    eval_corpora = sorted(
        {
            row.get('eval_corpus')
            for row in rows
            if any(is_number(row.get(metric)) for metric, _ in PERPLEXITY_METRICS)
        },
        key=lambda value: str(value or ''),
    )
    table_rows = []

    for eval_corpus in eval_corpora:
        corpus_rows = [
            row
            for row in rows
            if row.get('eval_corpus') == eval_corpus
        ]
        rows_by_model = {model_column_label(row): row for row in corpus_rows}
        for metric, metric_label in PERPLEXITY_METRICS:
            table_row = {
                'perplexity_metric': perplexity_metric_label(eval_corpus, metric_label)
            }
            has_score = False
            for model_label in model_labels:
                value = rows_by_model.get(model_label, {}).get(metric)
                table_row[model_label] = value
                has_score = has_score or is_number(value)

            if has_score:
                table_rows.append(table_row)

    return table_rows, ['perplexity_metric'] + model_labels


def write_perplexity_markdown(
    path: str, rows: list[dict[str, Any]], columns: list[str]
) -> None:
    """Write a rounded perplexity-by-model markdown table.

    Args:
        path: Destination markdown path.
        rows: Perplexity table rows.
        columns: Output columns.

    Returns:
        None.
    """
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('| ' + ' | '.join(columns) + ' |\n')
        handle.write('| ' + ' | '.join(['---'] * len(columns)) + ' |\n')
        for row in rows:
            values = []
            for column in columns:
                if column == 'perplexity_metric':
                    value = str(row.get(column, '')).replace('|', '\\|')
                else:
                    value = format_benchmark_score(row.get(column))

                values.append(value)

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


def default_benchmark_table_name(language: str) -> str:
    """Build a default benchmark table filename.

    Args:
        language: Benchmark language filter.

    Returns:
        Default benchmark markdown filename.
    """
    if language == 'all':
        return f'{BENCHMARK_TABLE_FILE_PREFIX}.md'

    return f'{BENCHMARK_TABLE_FILE_PREFIX}_{language}.md'


def default_language_average_table_name() -> str:
    """Build a default language-average table filename.

    Returns:
        Default language-average markdown filename.
    """
    return f'{LANGUAGE_AVERAGE_TABLE_FILE_PREFIX}.md'


def default_perplexity_table_name(model_keys: tuple[str, ...]) -> str:
    """Build a default perplexity table filename.

    Args:
        model_keys: Model keys passed to the analyzer.

    Returns:
        Default perplexity markdown filename.
    """
    return f'{PERPLEXITY_TABLE_FILE_PREFIX}.md'


def has_multiple_model_variants(rows: list[dict[str, Any]]) -> bool:
    """Return whether rows contain more than one model or variant.

    Args:
        rows: Summary rows.

    Returns:
        True when at least two distinct model variant columns are present.
    """
    return len({model_column_label(row) for row in rows}) > 1


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
@click.option(
    '--benchmark-markdown-name',
    default=None,
    help='Benchmark-by-model markdown output filename.',
)
@click.option(
    '--benchmark-language',
    default='all',
    type=click.Choice(BENCHMARK_LANGUAGE_CHOICES),
    help='Benchmark language to include in the comparison table.',
)
@click.option(
    '--perplexity-markdown-name',
    default=None,
    help='Perplexity-by-model markdown output filename.',
)
@click.option(
    '--language-average-markdown-name',
    default=None,
    help='Language-average benchmark markdown output filename.',
)
def main(
    model_keys: tuple[str, ...],
    evaluation_root: str,
    profile_names: tuple[str, ...],
    output_dir: str,
    csv_name: str | None,
    markdown_name: str | None,
    benchmark_markdown_name: str | None,
    benchmark_language: str,
    perplexity_markdown_name: str | None,
    language_average_markdown_name: str | None,
) -> None:
    """Summarize available evaluation results.

    Args:
        model_keys: Model keys to analyze.
        evaluation_root: Evaluation output root directory.
        profile_names: Evaluation profiles to analyze.
        output_dir: Analysis output directory.
        csv_name: Optional CSV output filename.
        markdown_name: Optional markdown output filename.
        benchmark_markdown_name: Optional benchmark markdown output filename.
        benchmark_language: Benchmark language to include.
        perplexity_markdown_name: Optional perplexity markdown output filename.
        language_average_markdown_name: Optional language-average output filename.

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
    resolved_benchmark_markdown_name = (
        benchmark_markdown_name or default_benchmark_table_name(benchmark_language)
    )
    csv_path = os.path.join(output_dir, resolved_csv_name)
    markdown_path = os.path.join(output_dir, resolved_markdown_name)
    benchmark_markdown_path = os.path.join(output_dir, resolved_benchmark_markdown_name)
    write_csv(csv_path, rows, columns)
    write_markdown(markdown_path, rows, columns)
    benchmark_rows, benchmark_columns = benchmark_table(rows, benchmark_language)
    write_benchmark_markdown(
        benchmark_markdown_path, benchmark_rows, benchmark_columns
    )
    resolved_language_average_markdown_name = (
        language_average_markdown_name or default_language_average_table_name()
    )
    language_average_markdown_path = os.path.join(
        output_dir, resolved_language_average_markdown_name
    )
    language_average_rows, language_average_columns = language_average_table(rows)
    write_language_average_markdown(
        language_average_markdown_path,
        language_average_rows,
        language_average_columns,
    )
    print(csv_path)
    print(markdown_path)
    print(benchmark_markdown_path)
    print(language_average_markdown_path)
    if has_multiple_model_variants(rows):
        resolved_perplexity_markdown_name = (
            perplexity_markdown_name or default_perplexity_table_name(model_keys)
        )
        perplexity_markdown_path = os.path.join(
            output_dir, resolved_perplexity_markdown_name
        )
        perplexity_rows, perplexity_columns = perplexity_table(rows)
        write_perplexity_markdown(
            perplexity_markdown_path, perplexity_rows, perplexity_columns
        )
        print(perplexity_markdown_path)
    else:
        print('[skip] perplexity comparison table requires multiple model variants')

    print(f'[done] wrote {len(rows)} row(s)')


if __name__ == '__main__':
    main()
