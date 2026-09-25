from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path

import yaml


PRIVATE_LOCAL_DATA_FILES = {
    "data/evaluation/coreguapa_identified_all.jsonl",
}


class LmEvalTaskLoader(yaml.SafeLoader):
    pass


LmEvalTaskLoader.add_constructor(
    "!function", lambda loader, node: loader.construct_scalar(node)
)


def tracked_files() -> list[Path]:
    output = subprocess.check_output(["git", "ls-files"], text=True)
    return [Path(line) for line in output.splitlines()]


def load_yaml(path: Path) -> object:
    loader = LmEvalTaskLoader if path.parts[:2] == ("evaluation", "lm_eval_tasks") else yaml.SafeLoader
    with path.open("r", encoding="utf-8") as handle:
        return yaml.load(handle, Loader=loader)


def iter_data_files(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        paths: list[str] = []
        for item in value:
            paths.extend(iter_data_files(item))
        return paths
    if isinstance(value, dict):
        paths: list[str] = []
        for item in value.values():
            paths.extend(iter_data_files(item))
        return paths
    return []


def test_tracked_python_json_and_yaml_parse() -> None:
    for path in tracked_files():
        if path.suffix == ".py":
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        elif path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix in {".yaml", ".yml"}:
            load_yaml(path)


def test_matrices_have_required_top_level_sections() -> None:
    for path in Path("configs/train").glob("*_cpt_matrix.y*ml"):
        matrix = load_yaml(path)
        assert isinstance(matrix, dict)
        for key in ("generated_config_dir", "runs_root", "reporting", "model", "defaults", "profiles"):
            assert key in matrix, f"{path} is missing {key}"

    for path in Path("configs/evaluation").glob("*_eval_matrix.y*ml"):
        matrix = load_yaml(path)
        assert isinstance(matrix, dict)
        for key in (
            "generated_config_dir",
            "outputs_root",
            "task_include_path",
            "training_outputs_root",
            "model",
            "defaults",
            "suites",
            "profiles",
        ):
            assert key in matrix, f"{path} is missing {key}"


def test_local_json_tasks_reference_available_or_private_files() -> None:
    missing: set[str] = set()
    for path in Path("evaluation/lm_eval_tasks").glob("*.yaml"):
        task_config = load_yaml(path)
        assert isinstance(task_config, dict)
        if task_config.get("dataset_path") != "json":
            continue

        dataset_kwargs = task_config.get("dataset_kwargs", {})
        assert isinstance(dataset_kwargs, dict)
        for data_file in iter_data_files(dataset_kwargs.get("data_files")):
            if not Path(data_file).exists():
                missing.add(data_file)

    assert missing <= PRIVATE_LOCAL_DATA_FILES
