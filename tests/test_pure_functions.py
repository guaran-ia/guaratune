from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import analyze_eval_results, eval_config, eval_profile, generate_eval_configs, generate_train_configs
from src.train_config import parse_env_assignment


def test_train_run_name() -> None:
    assert generate_train_configs.run_name("gemma4_4b", "full", "C1_kuatia", None) == "gemma4_4b_full_C1_kuatia"
    assert generate_train_configs.run_name("gemma4_4b", "lora", "C1_kuatia", 64) == "gemma4_4b_lora_r64_C1_kuatia"
    with pytest.raises(ValueError, match="LoRA"):
        generate_train_configs.run_name("gemma4_4b", "lora", "C1_kuatia", None)


def test_eval_variant_name() -> None:
    assert generate_eval_configs.variant_name("base", None, None) == "base"
    assert generate_eval_configs.variant_name("full", "C1_kuatia", None) == "full_C1_kuatia"
    assert generate_eval_configs.variant_name("lora", "C1_kuatia", 64) == "lora_r64_C1_kuatia"


def test_instruction_tasks_are_filtered_from_non_instruction_suite() -> None:
    matrix = {
        "suites": {
            "instruction": {"tasks": ["ifeval"]},
            "mixed": {"tasks": ["regular_task", "ifeval"]},
        }
    }
    tasks, include_instruction = generate_eval_configs.filtered_tasks_for_variant(
        matrix,
        {"include_instruction_tasks": False},
        "mixed",
        matrix["suites"]["mixed"],
        {},
    )
    assert tasks == ["regular_task"]
    assert include_instruction is False


def test_eval_config_skips_missing_local_json_task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    include_path = tmp_path / "tasks"
    include_path.mkdir()
    (include_path / "private_task.yaml").write_text(
        "\n".join(
            [
                "task: private_task",
                "dataset_path: json",
                "dataset_kwargs:",
                "  data_files:",
                "    test: private/missing.jsonl",
            ]
        ),
        encoding="utf-8",
    )
    (include_path / "public_task.yaml").write_text(
        "task: public_task\ndataset_path: huggingface/dataset\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    available, skipped = eval_config.available_tasks(
        {"include_path": str(include_path), "tasks": ["private_task", "public_task"]}
    )
    assert available == ["public_task"]
    assert skipped == [
        {
            "task": "private_task",
            "reason": "missing_local_evaluation_data",
            "paths": ["private/missing.jsonl"],
        }
    ]


def test_parse_env_assignment() -> None:
    assert parse_env_assignment("") is None
    assert parse_env_assignment("# comment") is None
    assert parse_env_assignment("export WANDB_PROJECT='guarania models'") == (
        "WANDB_PROJECT",
        "guarania models",
    )
    with pytest.raises(ValueError):
        parse_env_assignment("export A=1 B=2")


def test_parse_variant() -> None:
    assert analyze_eval_results.parse_variant("base") == {
        "training_method": "base",
        "corpus": None,
        "lora_rank": None,
    }
    assert analyze_eval_results.parse_variant("full_C1_kuatia") == {
        "training_method": "full",
        "corpus": "C1_kuatia",
        "lora_rank": None,
    }
    assert analyze_eval_results.parse_variant("lora_r64_C1_kuatia") == {
        "training_method": "lora",
        "corpus": "C1_kuatia",
        "lora_rank": 64,
    }


def test_skipped_tasks_are_persisted_and_mark_analysis_incomplete(tmp_path: Path) -> None:
    evaluation_root = tmp_path / "evaluation"
    output_path = evaluation_root / "experiments" / "gemma4_12b" / "base" / "perplexity"
    eval_config.write_results(
        str(output_path),
        {
            "metadata": {
                "model_key": "gemma4_12b",
                "variant": "base",
                "profile": "experiments",
            },
            "results": {},
            "groups": {},
        },
        executed_tasks=[],
        skipped_tasks=[
            {
                "task": "guarani_coreguapa_perplexity",
                "reason": "missing_local_evaluation_data",
                "paths": ["data/evaluation/coreguapa_identified_all.jsonl"],
            }
        ],
    )

    result_path = output_path / "results.json"
    persisted = json.loads(result_path.read_text(encoding="utf-8"))
    assert persisted["guarania_evaluation"] == {
        "status": "incomplete",
        "executed_tasks": [],
        "skipped_tasks": [
            {
                "task": "guarani_coreguapa_perplexity",
                "reason": "missing_local_evaluation_data",
                "paths": ["data/evaluation/coreguapa_identified_all.jsonl"],
            }
        ],
    }

    rows = analyze_eval_results.analyze_results(
        str(evaluation_root), ("gemma4_12b",), ("experiments",)
    )
    assert rows == [
        {
            "model_key": "gemma4_12b",
            "variant": "base",
            "profiles": "experiments",
            "training_method": "base",
            "corpus": None,
            "eval_corpus": None,
            "lora_rank": None,
            "evaluation_status": "incomplete",
            "skipped_tasks": "guarani_coreguapa_perplexity",
        }
    ]


def test_merge_rows_rejects_conflicting_metrics() -> None:
    row = {
        "model_key": "gemma4_12b",
        "variant": "base",
        "profiles": "experiments",
        "training_method": "base",
        "corpus": None,
        "eval_corpus": None,
        "lora_rank": None,
        "evaluation_status": "complete",
        "skipped_tasks": "",
        "benchmark_acc": 0.5,
    }

    with pytest.raises(ValueError, match="Conflicting metric benchmark_acc"):
        analyze_eval_results.merge_rows([row, row | {"benchmark_acc": 0.6}])


def test_merge_rows_combines_non_overlapping_metrics() -> None:
    row = {
        "model_key": "gemma4_12b",
        "variant": "base",
        "profiles": "experiments",
        "training_method": "base",
        "corpus": None,
        "eval_corpus": None,
        "lora_rank": None,
        "evaluation_status": "complete",
        "skipped_tasks": "",
        "benchmark_acc": 0.5,
    }

    assert analyze_eval_results.merge_rows(
        [row, row | {"benchmark_f1": 0.6}]
    ) == [row | {"benchmark_f1": 0.6}]


def test_benchmark_table_matches_grouped_report_format(tmp_path: Path) -> None:
    base = {
        "model_key": "gemma4_4b",
        "variant": "base",
        "training_method": "base",
        "global_mmlu_lite_acc_norm": 0.3875,
        "global_mmlu_lite_acc_norm_stderr": 0.0125,
        "guarani_2m_belebele_acc_norm": 0.43,
        "guarani_multiwikiqa_f1": 0.2021,
        "spanish_global_mmlu_lite_acc_norm": 0.6325,
        "english_global_mmlu_lite_acc_norm": 0.70,
        "guarani_flores200_eng_to_grn_bleu": 0.67,
        "guarani_flores200_eng_to_grn_chrf_plus_plus": 3.4,
    }
    variant = base | {
        "variant": "full_C1_kuatia",
        "training_method": "full",
        "corpus": "C1_kuatia",
        "global_mmlu_lite_acc_norm": 0.325,
        "global_mmlu_lite_acc_norm_stderr": 0.005,
        "guarani_2m_belebele_acc_norm": 0.4667,
        "guarani_multiwikiqa_f1": 0.2874,
        "spanish_global_mmlu_lite_acc_norm": 0.5625,
        "english_global_mmlu_lite_acc_norm": 0.665,
        "guarani_flores200_eng_to_grn_bleu": 0.89,
        "guarani_flores200_eng_to_grn_chrf_plus_plus": 4.0,
    }

    sample_counts = {
        "global_mmlu_lite_acc_norm": 2,
        "guarani_2m_belebele_acc_norm": 2,
        "guarani_multiwikiqa_f1": 2,
        "spanish_global_mmlu_lite_acc_norm": 2,
        "english_global_mmlu_lite_acc_norm": 2,
        "guarani_flores200_eng_to_grn_bleu": 2,
    }
    rows, columns = analyze_eval_results.benchmark_table(
        [base, variant], "all", sample_counts
    )
    output_path = tmp_path / "evaluation_score_by_benchmark.md"
    analyze_eval_results.write_benchmark_markdown(
        str(output_path), rows, columns, ("gemma4_4b",), [base, variant]
    )
    report = output_path.read_text(encoding="utf-8")

    assert report.startswith("# Gemma4 4B (Full CPT)\n\n")
    assert report.index("| **Guarani** |") < report.index("| Average |")
    assert report.index("| Average |") < report.index("| **Spanish** |")
    assert report.index("| **Spanish** |") < report.index("| **English** |")
    assert report.index("| **English** |") < report.index("| **Translation** |")
    assert "| Global MMLU Lite GN (acc_norm) | **38.75 (36.30, 41.20)** | **32.50 (31.52, 33.48)** | **-6.25** |" in report
    assert "| Belebele GN (acc_norm) | 43.00 | **46.67** | **+3.67** |" in report
    assert "| Average | 33.99 | **35.97** | **+1.98** |" in report
    assert "FLORES+ EN->GN (chrF++)" in report
    assert "FLORES+ EN->GN (BLEU)" not in report
    assert "Average of en benchmarks" not in report
    assert "## Variant reference\n* C1: Kuatia\n" in report
    assert "| Instances |" not in report
    assert "Gain columns show signed score-point differences" in report
    mmlu_row = next(row for row in rows if row.get("benchmark") == "Global MMLU Lite GN (acc_norm)")
    assert mmlu_row["_standard_errors"] == {"base": 1.25, "C1": 0.5}
    assert "standard_errors" not in columns
    assert "_standard_errors" not in report

    average_rows, average_columns = analyze_eval_results.language_average_table(
        [base, variant]
    )
    assert [row['variant'] for row in average_rows] == ['base', 'C1']
    average_path = tmp_path / "evaluation_average_score_by_language.md"
    analyze_eval_results.write_language_average_markdown(
        str(average_path), average_rows, average_columns
    )
    average_report = average_path.read_text(encoding="utf-8")
    assert average_report.startswith("# Gemma4 4B (Full CPT)\n\n")
    assert "| Variant | Guarani Average | Guarani Gain | English Average | English Gain | Spanish Average | Spanish Gain |" in average_report
    assert "| C1 |" in average_report
    assert "full_C1_kuatia" not in average_report
    assert "## Variant reference\n* C1: Kuatia\n" in average_report
    assert "* C8: Kuatia without synthetic + 10% Spanish FineWeb-Edu + 10% FineWeb-Edu" in average_report

    filtered_rows, _ = analyze_eval_results.benchmark_table(
        [base, variant], "es", sample_counts
    )
    filtered_labels = [row["benchmark"] for row in filtered_rows]
    assert "**Spanish**" in filtered_labels
    assert "**Guarani**" not in filtered_labels
    assert "Global MMLU Lite ES (acc_norm)" in filtered_labels
    assert "FLORES+ EN->GN (BLEU)" not in filtered_labels

    assert analyze_eval_results.benchmark_report_heading(
        ("gemma4_4b", "llama3_8b"),
        [base, variant, {"training_method": "lora"}],
    ) == "Gemma4 4B, Llama3 8B (Full CPT, LoRA)"


def test_task_specs_apply_matrix_overrides_per_task() -> None:
    specs = eval_config.task_specs_for_evaluation(
        ["arc_easy", "arc_challenge", "unconfigured_task"],
        {
            "arc_easy": {"num_fewshot": 0, "test_split": "validation"},
            "arc_challenge": {
                "num_fewshot": 25,
                "fewshot_split": "train",
                "test_split": "validation",
            },
        },
    )
    assert specs == [
        {"task": "arc_easy", "num_fewshot": 0, "test_split": "validation"},
        {
            "task": "arc_challenge",
            "num_fewshot": 25,
            "fewshot_split": "train",
            "test_split": "validation",
        },
        "unconfigured_task",
    ]


def test_resolved_task_specs_apply_overrides_to_loaded_task_objects() -> None:
    class FakeTask:
        def __init__(self) -> None:
            self.config = {}
            self.fewshot_cfg = SimpleNamespace(split=None)

        def set_config(self, key: str, value: object) -> None:
            self.config[key] = value

    task = FakeTask()

    class FakeTaskManager:
        def load(self, name: str) -> dict[str, dict[str, FakeTask]]:
            return {"tasks": {name: task}}

    resolved = eval_config.resolve_task_specs_for_evaluation(
        ["local_task", "plain_task"],
        {
            "local_task": {
                "num_fewshot": 5,
                "fewshot_split": "train",
                "test_split": "dev",
            }
        },
        FakeTaskManager(),
    )
    assert resolved == [task, "plain_task"]
    assert task.config == {
        "num_fewshot": 5,
        "fewshot_split": "train",
        "test_split": "dev",
    }
    assert task.fewshot_cfg.split == "train"


def test_evaluation_matrix_generates_per_task_fewshot_overrides() -> None:
    matrix = generate_eval_configs.load_yaml(
        "configs/evaluation/gemma4-4_eval_matrix.yaml"
    )
    config = generate_eval_configs.evaluation_config(
        matrix, "smoke", "english_benchmarks", "base", None, None
    )
    assert config.get("num_fewshot") is None
    assert config["task_overrides"]["arc_easy"]["num_fewshot"] == 0
    assert config["task_overrides"]["arc_challenge"] == {
        "num_fewshot": 25,
        "fewshot_split": "train",
        "test_split": "validation",
    }


def test_eval_task_overrides_reject_invalid_fewshot_values() -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        eval_config.task_specs_for_evaluation(
            ["arc_easy"], {"arc_easy": {"num_fewshot": -1}}
        )


def test_eval_profile_refreshes_configs_from_current_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    written = []
    monkeypatch.setattr(
        generate_eval_configs,
        "write_yaml",
        lambda path, data, overwrite: written.append((path, data, overwrite)),
    )
    count = eval_profile.refresh_profile_configs("smoke", "gemma4_4b")
    assert count > 0
    assert count == len(written)
    assert all(overwrite for _, _, overwrite in written)
    assert any(
        data.get("task_overrides", {}).get("english_global_mmlu_lite", {}).get(
            "num_fewshot"
        ) == 5
        for _, data, _ in written
    )


def test_benchmark_stderr_uses_the_selected_fallback_metric() -> None:
    spec = {
        "metrics": ("primary_acc_norm", "fallback_acc"),
        "scale": 100.0,
        "label": "Example GN",
    }
    row = {
        "fallback_acc": 0.4,
        "fallback_acc_stderr": 0.03,
        "primary_acc_norm_stderr": 0.01,
    }
    assert analyze_eval_results.benchmark_standard_error(row, spec) == 3.0
    assert analyze_eval_results.benchmark_standard_error(
        {"fallback_acc": 0.4, "fallback_acc_stderr": "N/A"}, spec
    ) is None


def test_language_average_standard_error_propagates_available_metrics() -> None:
    row = {
        "global_mmlu_lite_acc_norm": 0.5,
        "global_mmlu_lite_acc_norm_stderr": 0.01,
        "guarani_mgsm_direct_exact_match": 0.6,
        "guarani_mgsm_direct_exact_match_stderr": 0.02,
    }
    assert analyze_eval_results.language_average_standard_error(row, "gn") == pytest.approx(
        (1**2 + 2**2) ** 0.5 / 2
    )
    assert analyze_eval_results.language_average_standard_error(
        row | {"guarani_mgsm_direct_exact_match_stderr": "N/A"}, "gn"
    ) is None
    average_rows, columns = analyze_eval_results.language_average_table([row])
    assert average_rows[0]["guarani_average_stderr"] == pytest.approx(
        (1**2 + 2**2) ** 0.5 / 2
    )
    assert "guarani_average_stderr" not in columns


def test_language_average_includes_flores_chrf_but_not_bleu_and_formats_ci(
    tmp_path: Path,
) -> None:
    row = {
        "model_key": "gemma4_4b",
        "variant": "base",
        "training_method": "base",
        "global_mmlu_lite_acc_norm": 0.5,
        "global_mmlu_lite_acc_norm_stderr": 0.01,
        "guarani_flores200_eng_to_grn_chrf_plus_plus": 50.0,
        "guarani_flores200_eng_to_grn_chrf_plus_plus_stderr": 0.02,
        "guarani_flores200_eng_to_grn_bleu": 70.0,
        "guarani_flores200_eng_to_grn_bleu_stderr": 5.0,
    }
    assert analyze_eval_results.language_average_score(row, "gn") == 50.0
    average_rows, columns = analyze_eval_results.language_average_table([row])
    output_path = tmp_path / "average.md"
    analyze_eval_results.write_language_average_markdown(
        str(output_path), average_rows, columns
    )
    report = output_path.read_text(encoding="utf-8")
    assert "| base | **50.00 (49.02, 50.98)** | **--** | **50.00 (49.96, 50.04)** | **--** |  | -- |" in report
    assert analyze_eval_results.default_language_average_table_name() == (
        "evaluation_average_score_by_language.md"
    )


def test_technical_sheet_lists_task_configuration_metadata(tmp_path: Path) -> None:
    evaluation_root = tmp_path / "evaluation"
    result_dir = evaluation_root / "experiments" / "gemma4_4b" / "full_C1_kuatia" / "guarani_benchmarks"
    result_dir.mkdir(parents=True)
    result = {
        "metadata": {
            "model_key": "gemma4_4b",
            "variant": "full_C1_kuatia",
            "profile": "experiments",
        },
        "results": {"guarani_wnli": {"acc,none": 0.7, "sample_len": 49}},
        "configs": {
            "guarani_wnli": {
                "task": "guarani_wnli",
                "dataset_name": None,
                "num_fewshot": 5,
                "test_split": "dev",
            }
        },
        "n-shot": {"guarani_wnli": 5},
        "n-samples": {"guarani_wnli": {"original": 49, "effective": 49}},
    }
    (result_dir / "results.json").write_text(json.dumps(result), encoding="utf-8")
    benchmark_rows = analyze_eval_results.technical_sheet_rows(
        str(evaluation_root), ("gemma4_4b",), ("experiments",)
    )
    variant_rows = analyze_eval_results.unique_variant_rows(
        [
            {
                "model_key": "gemma4_4b",
                "variant": "full_C1_kuatia",
                "training_method": "full",
                "corpus": "C1_kuatia",
            }
        ]
    )
    output_path = tmp_path / "technical_sheet.md"
    analyze_eval_results.write_technical_sheet(
        str(output_path), variant_rows, benchmark_rows
    )
    report = output_path.read_text(encoding="utf-8")
    assert "Full CPT" in report
    assert "WNLI" in report
    assert "Guarani (GN)" in report
    assert "| 49 | 5 | dev |" in report


def test_perplexity_report_uses_short_variant_headers_and_reference(tmp_path: Path) -> None:
    variants = [
        {
            "model_key": "gemma4_4b",
            "variant": "base",
            "training_method": "base",
            "eval_corpus": "coreguapa",
            "word_perplexity": 100.0,
            "byte_perplexity": 5.0,
            "bits_per_byte": 2.3,
        },
        {
            "model_key": "gemma4_4b",
            "variant": "full_C1_kuatia",
            "training_method": "full",
            "corpus": "C1_kuatia",
            "eval_corpus": "coreguapa",
            "word_perplexity": 10.0,
            "byte_perplexity": 2.0,
            "bits_per_byte": 1.1,
        },
    ]
    table_rows, columns = analyze_eval_results.perplexity_table(variants)
    output_path = tmp_path / "evaluation_perplexity.md"
    analyze_eval_results.write_perplexity_markdown(
        str(output_path), table_rows, columns, variants
    )
    report = output_path.read_text(encoding="utf-8")
    assert report.startswith("# Gemma4 4B (Full CPT)\n\n")
    assert "| Perplexity metric | base | C1 |" in report
    assert "| word perplexity | 100.00 | 10.00 |" in report
    assert "coreguapa" not in report.lower()
    assert "## Variant reference" in report


def test_percentage_by_benchmark_csv_includes_variant_confidence_intervals(
    tmp_path: Path,
) -> None:
    metric = "global_mmlu_lite_acc_norm"
    base = {
        "model_key": "gemma4_4b",
        "variant": "base",
        "training_method": "base",
        metric: 0.5,
        f"{metric}_stderr": 0.01,
    }
    c1 = base | {
        "variant": "full_C1_kuatia",
        "training_method": "full",
        "corpus": "C1_kuatia",
        metric: 0.6,
        f"{metric}_stderr": 0.02,
        f"{metric}_gain_base": 20.0,
    }
    output_path = tmp_path / analyze_eval_results.default_pretty_output_name(
        ("gemma4_4b",)
    )
    analyze_eval_results.pretty_report_csv(
        str(output_path),
        [base, c1],
        [metric, f"{metric}_gain_base"],
        {metric: 400},
    )
    report = output_path.read_text(encoding="utf-8")
    assert output_path.name == "evaluation_percentage_by_benchmark_gemma4_4b.csv"
    assert "Instances" not in report
    assert "Metric,base,CI base,C1,CI C1" in report
    assert '0.500,"[0.480, 0.520]",0.600,"[0.561, 0.639]"' in report
    assert "Gain Global MMLU Lite Normalized Accuracy,--,,20.00%," in report


def test_new_guarani_tasks_use_distinct_fewshot_and_scored_splits() -> None:
    mmlu = eval_config.load_lm_eval_task_yaml(
        "evaluation/lm_eval_tasks/guarani_global_mmlu_lite.yaml"
    )
    mgsm = eval_config.load_lm_eval_task_yaml(
        "evaluation/lm_eval_tasks/guarani_mgsm_direct.yaml"
    )
    wnli = eval_config.load_lm_eval_task_yaml(
        "evaluation/lm_eval_tasks/guarani_wnli.yaml"
    )
    assert set(mmlu["dataset_kwargs"]["data_files"]) == {"dev", "test"}
    assert set(mgsm["dataset_kwargs"]["data_files"]) == {"train", "test"}
    assert set(wnli["dataset_kwargs"]["data_files"]) == {"train", "dev"}

    for model in ("gemma4-2", "gemma4-4", "gemma4-12"):
        matrix = generate_eval_configs.load_yaml(
            f"configs/evaluation/{model}_eval_matrix.yaml"
        )
        suite = matrix["suites"]["guarani_benchmarks"]
        assert suite["tasks"] == ["guarani_mgsm_direct", "guarani_wnli"]
        assert suite["task_overrides"]["guarani_mgsm_direct"] == {
            "num_fewshot": 5,
            "fewshot_split": "train",
            "test_split": "test",
        }
        assert suite["task_overrides"]["guarani_wnli"] == {
            "num_fewshot": 5,
            "fewshot_split": "train",
            "test_split": "dev",
        }
        assert "guarani_benchmarks" in matrix["profiles"]["experiments"]["suites"]
        assert matrix["suites"]["global_mmlu_lite"]["task_overrides"][
            "guarani_global_mmlu_lite"
        ]["fewshot_split"] == "dev"
