#!/usr/bin/env python3
"""Run one configured benchmark across the variants in a model profile."""

from __future__ import annotations

import os
import tempfile
from typing import Any

import click
import yaml

from src import eval_config, eval_profile, generate_eval_configs


MATRIX_DIR = "configs/evaluation"
DEFAULT_PROFILE = "experiments"

BENCHMARK_ALIASES = {
    "mmlu": "global_mmlu_lite",
    "globalmmlu": "global_mmlu_lite",
    "globalmmlulite": "global_mmlu_lite",
    "belele": "belebele",
    "belebele": "belebele",
    "mgsm": "mgsm",
    "multiwikiqa": "multiwikiqa",
    "arceasy": "arc_easy",
    "arcchallenge": "arc_challenge",
    "piqa": "piqa",
    "hellaswag": "hellaswag",
    "xstorycloze": "xstorycloze",
    "gpqa": "gpqa",
    "wnli": "wnli",
    "copa": "copa",
    "truthfulqa": "truthfulqa",
    "flores": "flores_plus",
}

LANGUAGE_ALIASES = {
    "guarani": "gn",
    "gn": "gn",
    "grn": "gn",
    "spanish": "es",
    "es": "es",
    "spa": "es",
    "english": "en",
    "en": "en",
    "eng": "en",
}

FLORES_TASKS = {
    ("en", "gn"): "guarani_flores_plus_eng_to_grn",
    ("gn", "en"): "guarani_flores_plus_grn_to_eng",
    ("gn", "es"): "guarani_flores_plus_grn_to_spa",
    ("es", "gn"): "guarani_flores_plus_spa_to_grn",
}


def normalize(value: str) -> str:
    """Normalize labels and task names for matching."""
    return "".join(character for character in value.lower() if character.isalnum())


def normalize_language(language: str) -> str:
    """Return the language code accepted by the benchmark matrices."""
    normalized = normalize(language)
    try:
        return LANGUAGE_ALIASES[normalized]
    except KeyError as exc:
        supported = "English, Spanish, Guarani (or en, es, gn)"
        raise ValueError(
            f"Unsupported language {language!r}. Supported languages: {supported}."
        ) from exc


def matrix_paths() -> list[str]:
    """List evaluation matrices available in the project."""
    return sorted(
        os.path.join(MATRIX_DIR, filename)
        for filename in os.listdir(MATRIX_DIR)
        if filename.endswith("_eval_matrix.yaml")
    )


def load_model_matrix(model_key: str) -> dict[str, Any]:
    """Load the matrix belonging to a model key."""
    for path in matrix_paths():
        matrix = generate_eval_configs.load_yaml(path)
        if str(matrix.get("model", {}).get("key")) == model_key:
            return matrix
    available = sorted(
        str(generate_eval_configs.load_yaml(path).get("model", {}).get("key"))
        for path in matrix_paths()
    )
    raise ValueError(
        f"Unknown model key {model_key!r}. Available model keys: {', '.join(available)}."
    )


def task_language(task_name: str, suite_name: str) -> str:
    """Infer the language of a matrix task."""
    name = task_name.lower()
    if name.startswith("guarani_") or "_grn" in name:
        return "gn"
    if name.startswith("spanish_") or any(
        marker in name for marker in ("_es", "spa_latn", "_spa", "spa_to_")
    ):
        return "es"
    if name.startswith("english_") or any(
        marker in name for marker in ("_en", "eng_latn", "_eng", "eng_to_")
    ):
        return "en"
    if suite_name == "spanish_benchmarks":
        return "es"
    if suite_name == "guarani_benchmarks":
        return "gn"
    return "en"


def benchmark_matches(task_name: str, benchmark: str) -> bool:
    """Return whether a task ID corresponds to the requested benchmark."""
    task = normalize(task_name)
    canonical = BENCHMARK_ALIASES.get(normalize(benchmark), normalize(benchmark))
    return normalize(canonical) in task


def find_benchmark_task(
    matrix: dict[str, Any], benchmark: str, language: str, profile: str
) -> tuple[str, str]:
    """Resolve a benchmark and language to its configured task and suite."""
    suites = matrix.get("suites", {})
    profiles = matrix.get("profiles", {})
    if profile not in profiles:
        raise ValueError(f"Matrix for {matrix['model']['key']} has no profile {profile!r}.")
    configured_suites = profiles[profile].get("suites", [])
    matches = []
    for suite_name in configured_suites:
        suite = suites.get(suite_name, {})
        for task_name in suite.get("tasks", []):
            if benchmark_matches(str(task_name), benchmark) and task_language(
                str(task_name), str(suite_name)
            ) == language:
                matches.append((str(task_name), str(suite_name)))

    if not matches:
        available = sorted(
            {
                str(task)
                for suite_name in configured_suites
                for task in suites.get(suite_name, {}).get("tasks", [])
                if task_language(str(task), str(suite_name)) == language
            }
        )
        raise ValueError(
            f"Benchmark {benchmark!r} is not configured for language {language!r}. "
            f"Tasks for this language in profile {profile!r}: {', '.join(available)}."
        )
    if len(matches) > 1:
        task_ids = ", ".join(task for task, _ in matches)
        raise ValueError(
            f"Benchmark {benchmark!r} matched multiple tasks for language {language!r}: "
            f"{task_ids}. Use a more specific benchmark name."
        )
    return matches[0]


def configured_metrics(matrix: dict[str, Any], task_name: str, suite_name: str) -> list[str]:
    """Get configured metric IDs, falling back to the task YAML defaults."""
    overrides = matrix.get("suites", {}).get(suite_name, {}).get("task_overrides", {})
    task_override = overrides.get(task_name, {})
    metrics = task_override.get("metrics")
    if metrics:
        return [str(metric) for metric in metrics]

    task_path = os.path.join(str(matrix.get("task_include_path", "")), f"{task_name}.yaml")
    if not os.path.isfile(task_path):
        raise ValueError(
            f"No metric override is configured for {task_name}, and its task YAML "
            f"was not found at {task_path}."
        )
    task_config = eval_config.load_lm_eval_task_yaml(task_path)
    metric_list = task_config.get("metric_list", [])
    return [
        str(item["metric"] if isinstance(item, dict) else item)
        for item in metric_list
        if isinstance(item, (str, dict)) and (not isinstance(item, dict) or "metric" in item)
    ]


def build_run_configs(
    matrix: dict[str, Any],
    profile: str,
    task_name: str,
    suite_name: str,
    selected_metrics: list[str],
    cpt_method: str | None,
    lora_rank: int | None,
    dataset_configuration: str | None,
) -> list[dict[str, Any]]:
    """Build one-task configs matching the requested variant filters."""
    configs = []
    for _, config in generate_eval_configs.iter_configs(matrix, profile):
        if config.get("metadata", {}).get("suite") != suite_name:
            continue
        if task_name not in config.get("tasks", []):
            continue
        metadata = config.get("metadata", {})
        variant = str(metadata.get("variant", ""))
        variant_method = (
            "base" if variant == "base" else "lora" if variant.startswith("lora_") else "full"
        )
        if cpt_method is not None and variant_method != cpt_method:
            continue
        if lora_rank is not None and metadata.get("lora_rank") != lora_rank:
            continue
        corpus = metadata.get("corpus")
        if dataset_configuration is not None and (
            corpus is None or not str(corpus).startswith(f"{dataset_configuration}_")
        ):
            continue

        config = dict(config)
        config["tasks"] = [task_name]
        task_overrides = dict(config.get("task_overrides", {}))
        task_override = dict(task_overrides.get(task_name, {}))
        task_override["metrics"] = selected_metrics
        task_overrides[task_name] = task_override
        config["task_overrides"] = task_overrides

        metadata = dict(config.get("metadata", {}))
        metadata["benchmark_task"] = task_name
        config["metadata"] = metadata
        suffix = "_".join(selected_metrics)
        config["output_path"] = os.path.join(
            str(config["output_path"]), "single_benchmark", f"{task_name}_{suffix}"
        )
        configs.append(config)

    if not configs:
        raise ValueError(
            f"No run config found for {task_name} matching method={cpt_method or 'all'}, "
            f"dataset={dataset_configuration or 'all'}, and LoRA rank="
            f"{lora_rank if lora_rank is not None else 'all'} "
            f"in profile {profile!r}."
        )
    return configs


def run_configs(configs: list[dict[str, Any]]) -> None:
    """Run configs from temporary files and skip unavailable trained variants."""
    with tempfile.TemporaryDirectory(prefix="eval-benchmark-") as temporary_dir:
        for index, config in enumerate(configs):
            config_path = os.path.join(temporary_dir, f"run-{index}.yaml")
            with open(config_path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
            if eval_profile.exist_trained_variant(config_path) is not False:
                print(f"[run] {config['metadata']['run_name']} | {config['tasks'][0]}", flush=True)
                eval_config.run_evaluation(config_path)


@click.command(context_settings={"show_default": True})
@click.option("--benchmark", required=True, help="Benchmark name, for example mgsm or arc_easy.")
@click.option(
    "--language",
    help="Benchmark language: English, Spanish, or Guarani. If omitted, run all configured languages.",
)
@click.option("--source_language", "source_language", help="FLORES+ source language.")
@click.option("--target_language", "target_language", help="FLORES+ target language.")
@click.option("--metric", help="Metric ID. Defaults to the metric(s) configured in the matrix.")
@click.option("--model", "model_key", required=True, help="Model key, for example gemma4_4b.")
@click.option(
    "--cpt-method",
    "--cpt_method",
    "cpt_method",
    type=click.Choice(["full", "lora"], case_sensitive=False),
    help="Training method to evaluate. If omitted, include all available methods and base.",
)
@click.option(
    "--lora-rank",
    "--lora_rank",
    type=int,
    help="LoRA rank; required with --cpt-method lora. If omitted, include all ranks.",
)
@click.option(
    "--dataset-configuration",
    "--dataset_configuration",
    "--dataset-config",
    "dataset_configuration",
    type=click.Choice([f"c{number}" for number in range(1, 9)], case_sensitive=False),
    help="Training dataset configuration. If omitted, include all available datasets.",
)
@click.option("--profile", default=DEFAULT_PROFILE, help="Evaluation profile to run.")
def main(
    benchmark: str,
    language: str | None,
    source_language: str | None,
    target_language: str | None,
    metric: str | None,
    model_key: str,
    cpt_method: str | None,
    lora_rank: int | None,
    dataset_configuration: str | None,
    profile: str,
) -> None:
    """Run one benchmark across variants selected by the optional filters."""
    try:
        cpt_method = cpt_method.lower() if cpt_method else None
        dataset_configuration = dataset_configuration.upper() if dataset_configuration else None
        if cpt_method == "lora" and lora_rank is None:
            raise ValueError("LoRA evaluation requires --lora-rank. Specify the trained adapter rank.")
        if cpt_method == "full" and lora_rank is not None:
            raise ValueError("--lora-rank can only be used with --cpt-method lora.")
        if cpt_method is None and lora_rank is not None:
            raise ValueError("Specify --cpt-method lora when using --lora-rank.")

        is_flores = (
            BENCHMARK_ALIASES.get(normalize(benchmark), normalize(benchmark))
            == "flores_plus"
        )
        if is_flores:
            if language:
                raise ValueError(
                    "For FLORES+, use --source_language and --target_language "
                    "instead of --language."
                )
            if not source_language or not target_language:
                raise ValueError("FLORES+ requires both --source_language and --target_language.")
            source = normalize_language(source_language)
            target = normalize_language(target_language)
            try:
                task_name = FLORES_TASKS[(source, target)]
            except KeyError as exc:
                directions = ", ".join(f"{src} to {dst}" for src, dst in FLORES_TASKS)
                raise ValueError(
                    f"Unsupported FLORES+ direction. Supported directions: {directions}."
                ) from exc
            language_code = "gn"
            suite_name = "flores_plus"
        else:
            if source_language or target_language:
                raise ValueError(
                    "--source_language and --target_language are only used with FLORES+."
                )

        matrix = load_model_matrix(model_key)
        if is_flores:
            configured = matrix.get("suites", {}).get("flores_plus", {}).get("tasks", [])
            if task_name not in configured:
                raise ValueError(
                    f"FLORES+ direction {source_language} to {target_language} "
                    "is not configured."
                )
            resolved_tasks = [(task_name, suite_name)]
        else:
            language_codes = [normalize_language(language)] if language else ["es", "en", "gn"]
            resolved_tasks = []
            unavailable_languages = []
            for language_code in language_codes:
                try:
                    resolved_tasks.append(
                        find_benchmark_task(matrix, benchmark, language_code, profile)
                    )
                except ValueError as exc:
                    if language or "is not configured for language" not in str(exc):
                        raise
                    unavailable_languages.append(language_code)
            if not resolved_tasks:
                raise ValueError(
                    f"Benchmark {benchmark!r} is not configured for Spanish, English, "
                    f"or Guarani in profile {profile!r}."
                )
            if unavailable_languages:
                print(
                    "[skip] no configured task for language(s): "
                    + ", ".join(unavailable_languages),
                    flush=True,
                )

        configs = []
        task_descriptions = []
        for task_name, suite_name in resolved_tasks:
            default_metrics = configured_metrics(matrix, task_name, suite_name)
            if not default_metrics:
                raise ValueError(f"No metrics are configured for task {task_name}.")
            if metric and metric not in default_metrics:
                raise ValueError(
                    f"Unsupported metric {metric!r} for {benchmark} ({task_name}). "
                    f"Supported metrics: {', '.join(default_metrics)}."
                )
            metrics = [metric] if metric else default_metrics
            task_descriptions.append(f"{task_name} [{', '.join(metrics)}]")
            configs.extend(
                build_run_configs(
                    matrix,
                    profile,
                    task_name,
                    suite_name,
                    metrics,
                    cpt_method,
                    lora_rank,
                    dataset_configuration,
                )
            )
        print(
            f"[benchmark] {benchmark} | tasks: {'; '.join(task_descriptions)} "
            f"| model: {model_key} | method: {cpt_method or 'all'} "
            f"| dataset: {dataset_configuration or 'all'} "
            f"| LoRA rank: {lora_rank if lora_rank is not None else 'all'} "
            f"| variants: {len(configs)}",
            flush=True,
        )
        run_configs(configs)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


if __name__ == "__main__":
    main()
