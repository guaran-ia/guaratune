from __future__ import annotations

from pathlib import Path

import yaml

from src import generate_eval_configs, generate_train_configs


def test_generate_train_configs_smoke_to_temp_dir(tmp_path: Path) -> None:
    matrix = generate_train_configs.load_yaml("configs/train/gemma4-12_cpt_matrix.yaml")
    matrix["generated_config_dir"] = str(tmp_path / "train-generated")
    matrix["runs_root"] = str(tmp_path / "train-outputs" / matrix["model"]["key"])

    output_dir = matrix["generated_config_dir"]
    model_key = matrix["model"]["key"]
    env_path = generate_train_configs.write_wandb_env(
        output_dir, matrix["reporting"], model_key, overwrite=True
    )
    written = []
    for method, corpus, rank in generate_train_configs.iter_runs(matrix, "smoke"):
        name = generate_train_configs.run_name(model_key, method, corpus, rank)
        path = generate_train_configs.generated_config_path(output_dir, "smoke", model_key, name)
        config = generate_train_configs.training_config(matrix, "smoke", method, corpus, rank)
        generate_train_configs.write_yaml(path, config, overwrite=True)
        written.append(Path(path))

    assert Path(env_path).is_file()
    assert len(written) == 1
    generated = yaml.safe_load(written[0].read_text(encoding="utf-8"))
    assert generated["dataset"] == "C1_kuatia"
    assert generated["finetuning_type"] == "lora"
    assert generated["output_dir"].startswith(str(tmp_path))


def test_generate_eval_configs_smoke_to_temp_dir(tmp_path: Path) -> None:
    matrix = generate_eval_configs.load_yaml("configs/evaluation/gemma4-12_eval_matrix.yaml")
    matrix["generated_config_dir"] = str(tmp_path / "eval-generated")
    matrix["outputs_root"] = str(tmp_path / "eval-outputs")
    matrix["training_outputs_root"] = str(tmp_path / "train-outputs" / matrix["model"]["key"] / "experiments")

    generated_configs = list(generate_eval_configs.iter_configs(matrix, "smoke"))
    assert generated_configs

    written = []
    for path, config in generated_configs:
        generate_eval_configs.write_yaml(path, config, overwrite=True)
        written.append(Path(path))

    assert all(path.is_file() for path in written)
    assert any("base_global_mmlu_lite" in path.name for path in written)

    loaded = yaml.safe_load(written[0].read_text(encoding="utf-8"))
    assert loaded["include_path"] == "evaluation/lm_eval_tasks"
    assert loaded["output_path"].startswith(str(tmp_path))
    assert loaded["model_args"]["trust_remote_code"] is matrix["model"]["trust_remote_code"]
    assert "trust_remote_code" not in loaded
    assert "show_config" not in loaded
