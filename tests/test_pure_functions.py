from __future__ import annotations

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

    assert eval_config.available_tasks(
        {"include_path": str(include_path), "tasks": ["private_task", "public_task"]}
    ) == ["public_task"]


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
