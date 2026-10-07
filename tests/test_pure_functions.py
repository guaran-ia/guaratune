from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import analyze_eval_results, eval_config, generate_eval_configs, generate_train_configs
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
        "guarani_2m_belebele_acc_norm": 0.43,
        "guarani_multiwikiqa_f1": 0.2021,
        "spanish_global_mmlu_lite_acc_norm": 0.6325,
        "english_global_mmlu_lite_acc_norm": 0.70,
        "guarani_flores200_eng_to_grn_bleu": 0.67,
    }
    variant = base | {
        "variant": "full_C1_kuatia",
        "training_method": "full",
        "corpus": "C1_kuatia",
        "global_mmlu_lite_acc_norm": 0.325,
        "guarani_2m_belebele_acc_norm": 0.4667,
        "guarani_multiwikiqa_f1": 0.2874,
        "spanish_global_mmlu_lite_acc_norm": 0.5625,
        "english_global_mmlu_lite_acc_norm": 0.665,
        "guarani_flores200_eng_to_grn_bleu": 0.89,
    }

    rows, columns = analyze_eval_results.benchmark_table([base, variant], "all")
    output_path = tmp_path / "evaluation_benchmark_table.md"
    analyze_eval_results.write_benchmark_markdown(
        str(output_path), rows, columns, ("gemma4_4b",), [base, variant]
    )
    report = output_path.read_text(encoding="utf-8")

    assert report.startswith("# Gemma4 4B (Full CPT)\n\n")
    assert report.index("| **Guarani** |") < report.index("| Average |")
    assert report.index("| Average |") < report.index("| **Spanish** |")
    assert report.index("| **Spanish** |") < report.index("| **English** |")
    assert report.index("| **English** |") < report.index("| **Translation** |")
    assert "| Global MMLU Lite GN (acc_norm) | **38.75** | 32.50 (-6.25) |" in report
    assert "| Belebele GN (acc_norm) | 43.00 | **46.67 (+3.67)** |" in report
    assert "| Average | 33.99 | **35.97 (+1.98)** |" in report
    assert "FLORES-200 EN->GN (BLEU)" in report
    assert "Average of en benchmarks" not in report
    assert "## Dataset configurations\n* C1: Kuatia\n" in report

    average_rows, average_columns = analyze_eval_results.language_average_table(
        [base, variant]
    )
    assert [row['variant'] for row in average_rows] == ['base', 'C1']
    average_path = tmp_path / "evaluation_language_average_table.md"
    analyze_eval_results.write_language_average_markdown(
        str(average_path), average_rows, average_columns
    )
    average_report = average_path.read_text(encoding="utf-8")
    assert average_report.startswith("# Gemma4 4B (Full CPT)\n\n")
    assert "| C1 |" in average_report
    assert "full_C1_kuatia" not in average_report
    assert "## Dataset configurations\n* C1: Kuatia\n" in average_report
    assert "* C8: Kuatia without synthetic + 10% Spanish FineWeb-Edu + 10% FineWeb-Edu" in average_report

    filtered_rows, _ = analyze_eval_results.benchmark_table([base, variant], "es")
    filtered_labels = [row["benchmark"] for row in filtered_rows]
    assert "**Spanish**" in filtered_labels
    assert "**Guarani**" not in filtered_labels
    assert "Global MMLU Lite ES (acc_norm)" in filtered_labels
    assert "FLORES-200 EN->GN (BLEU)" not in filtered_labels

    assert analyze_eval_results.benchmark_report_heading(
        ("gemma4_4b", "llama3_8b"),
        [base, variant, {"training_method": "lora"}],
    ) == "Gemma4 4B, Llama3 8B (Full CPT, LoRA)"
