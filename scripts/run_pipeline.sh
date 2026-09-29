#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_pipeline.sh --model MODEL_KEY [options]

Run the end-to-end CPT pipeline for one model.

Required:
  --model MODEL_KEY          Model key declared in configs/train/*_cpt_matrix.yaml.

Options:
  --profile PROFILE         Training profile to generate and run. Default: experiments.
  --data DATA_CONFIG        Data configuration to run, for example C1 or C1_kuatia.
                            Repeat to run multiple data configurations. Default: all.
  --data-config-file PATH   Data preparation config file. Default: configs/data/gemma4_cpt.yaml.
  --method METHOD           CPT method to run. Supported forms: full, lora, lora64,
                            lora_r64, lora:64. Repeat to run multiple methods.
                            Default: all.
  --eval-matrix PATH        Evaluation matrix file. Default: auto-detect by model key.
  --eval-suite SUITE        Evaluation suite to run, for example perplexity or global_mmlu_lite.
                            Repeat to run multiple suites. Default: all.
  --no-eval-base            Do not include base-model evaluations.
  --skip-evaluation         Skip evaluation config generation and execution.
  --skip-analysis           Skip evaluation result analysis.
  --benchmark-language LANG Benchmark language for the comparison table. Default: all.
  --env-file PATH           Local secret env file forwarded to training. Default: .env.
  --cleanup-checkpoints BOOL Remove checkpoint directories after successful training.
                            Default: true.
  --cleanup-optimizers BOOL Pass optimizer cleanup setting to src.train_config.
                            Default: true.
  --force-data-prep         Run data preparation even if required artifacts already exist.
  --no-overwrite            Do not overwrite generated training configs during prepare.
  --dry-run                 Print selected configs without launching training.
  -h, --help                Show this help.

Examples:
  scripts/run_pipeline.sh --model gemma4_4b
  scripts/run_pipeline.sh --model gemma4_4b --data C1 --method full
  scripts/run_pipeline.sh --model gemma4_4b --data C1 --data C2 --method lora64 --method lora512
EOF
}

repo_root() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "${script_dir}/.." && pwd
}

normalize_method_pattern() {
  local model_key="$1"
  local method="$2"
  local normalized

  normalized="$(printf '%s' "${method}" | tr '[:upper:]' '[:lower:]')"
  normalized="${normalized// /_}"
  normalized="${normalized//-/_}"
  normalized="${normalized//:/_}"

  if [[ "${normalized}" == "full" ]]; then
    printf '%s\n' "${model_key}_full_*.yaml"
    return
  fi

  if [[ "${normalized}" == "lora" ]]; then
    printf '%s\n' "${model_key}_lora_r*_*.yaml"
    return
  fi

  if [[ "${normalized}" =~ ^lora_?r?([0-9]+)$ ]]; then
    printf '%s\n' "${model_key}_lora_r${BASH_REMATCH[1]}_*.yaml"
    return
  fi

  printf 'Unsupported CPT method: %s\n' "${method}" >&2
  exit 2
}

normalize_data_pattern() {
  local data_config="$1"

  if [[ "${data_config}" == *"*"* || "${data_config}" == *"?"* || "${data_config}" == *"["* ]]; then
    printf '%s\n' "*_${data_config}.yaml"
    return
  fi

  if [[ "${data_config}" =~ ^C[0-9]+$ ]]; then
    printf '%s\n' "*_${data_config}_*.yaml"
    return
  fi

  printf '%s\n' "*_${data_config}.yaml"
}

matches_any_pattern() {
  local value="$1"
  shift
  local pattern

  if [[ "$#" -eq 0 ]]; then
    return 0
  fi

  for pattern in "$@"; do
    if [[ "${value}" == ${pattern} ]]; then
      return 0
    fi
  done

  return 1
}

eval_matrix_for_model() {
  local model_key="$1"

  python - "$model_key" <<'PY'
import glob
import sys
import yaml

model_key = sys.argv[1]
matches = []
available = []

for path in sorted(glob.glob('configs/evaluation/*_eval_matrix.y*ml')):
    with open(path, 'r', encoding='utf-8') as handle:
        config = yaml.safe_load(handle)

    model = config.get('model', {}) if isinstance(config, dict) else {}
    current_key = model.get('key') if isinstance(model, dict) else None
    if current_key:
        available.append(current_key)
    if current_key == model_key:
        matches.append(path)

if len(matches) == 1:
    print(matches[0])
elif len(matches) > 1:
    raise SystemExit(
        f'Multiple evaluation matrices declare model key {model_key}: {", ".join(matches)}'
    )
else:
    available_text = ', '.join(sorted(available)) or 'none'
    raise SystemExit(
        f'No evaluation matrix found for model key {model_key}. '
        f'Available evaluation model keys: {available_text}'
    )
PY
}

selected_corpora() {
  local data_config_file="$1"
  shift

  python - "$data_config_file" "$@" <<'PY'
import fnmatch
import sys
import yaml

config_path = sys.argv[1]
filters = sys.argv[2:]

with open(config_path, 'r', encoding='utf-8') as handle:
    config = yaml.safe_load(handle)

corpora = config.get('corpora', {})
if not isinstance(corpora, dict):
    raise SystemExit(f'Data config has no corpora mapping: {config_path}')

names = list(corpora.keys())
selected = []

if not filters:
    selected = names
else:
    for item in filters:
        matches = []
        if any(char in item for char in '*?['):
            matches = [name for name in names if fnmatch.fnmatch(name, item)]
        elif item in corpora:
            matches = [item]
        elif item.startswith('C') and item[1:].isdigit():
            matches = [name for name in names if name == item or name.startswith(f'{item}_')]

        if not matches:
            raise SystemExit(f'Unknown data configuration {item}. Available: {", ".join(names)}')

        for match in matches:
            if match not in selected:
                selected.append(match)

for name in selected:
    print(name)
PY
}

normalize_eval_suite_pattern() {
  local suite="$1"

  if [[ "${suite}" == *"*"* || "${suite}" == *"?"* || "${suite}" == *"["* ]]; then
    printf '%s\n' "${suite}"
    return
  fi

  printf '%s\n' "*_${suite}.yaml"
}

missing_data_artifacts() {
  local data_config_file="$1"
  shift

  python - "$data_config_file" "$@" <<'PY'
import json
import os
import sys
import yaml

config_path = sys.argv[1]
selected = sys.argv[2:]

with open(config_path, 'r', encoding='utf-8') as handle:
    config = yaml.safe_load(handle)

output_dir = config.get('output_dir', 'data')
corpora = config.get('corpora', {})
sources = config.get('sources', {})

missing = []

def require_path(path):
    if path and not os.path.exists(path):
        missing.append(path)

def require_dataset_info_key(path, key):
    if not os.path.exists(path):
        return

    try:
        with open(path, 'r', encoding='utf-8') as handle:
            data = json.load(handle)
    except Exception:
        missing.append(f'{path}::{key}')
        return

    if key not in data:
        missing.append(f'{path}::{key}')

dataset_info_path = os.path.join(output_dir, 'dataset_info.json')
require_path(dataset_info_path)
require_path(os.path.join(output_dir, 'source_revisions.lock.json'))

for name in selected:
    require_path(os.path.join(output_dir, 'train', f'{name}.jsonl'))
    require_path(os.path.join(output_dir, 'validation', f'{name}.jsonl'))
    require_path(os.path.join(output_dir, 'manifests', f'{name}.manifest.json'))
    require_dataset_info_key(dataset_info_path, name)
    require_dataset_info_key(dataset_info_path, f'{name}_validation')

    corpus = corpora.get(name, {})
    if corpus.get('kuatia') is not None:
        kuatia = sources.get('kuatia', {})
        require_path(kuatia.get('selection_path'))

    for augmentation in corpus.get('augmentations', []):
        source_name = augmentation.get('source')
        source = sources.get(source_name, {})
        require_path(source.get('selection_path'))
        require_path(augmentation.get('train_selection_path'))
        require_path(augmentation.get('validation_selection_path'))

for path in missing:
    print(path)
PY
}

print_list() {
  local prefix="$1"
  shift
  local item

  for item in "$@"; do
    printf '%s%s\n' "${prefix}" "${item}"
  done
}

check_or_prepare_data() {
  local data_config_file="$1"
  shift
  local requested_corpora=("$@")
  local missing=()
  local data_command=()

  printf '[step 2/4] Checking required data artifacts\n'
  printf '[data] Config: %s\n' "${data_config_file}"
  printf '[data] Requested corpora: %s\n' "${requested_corpora[*]}"
  printf '[data] Required groups: train files, validation files, manifests, selections, dataset_info, source revisions\n'

  mapfile -t missing < <(missing_data_artifacts "${data_config_file}" "${requested_corpora[@]}")

  if [[ "${force_data_prep}" == "false" && "${#missing[@]}" -eq 0 ]]; then
    printf '[data] All required artifacts already exist. Skipping data preparation.\n'
    return
  fi

  if [[ "${force_data_prep}" == "true" ]]; then
    printf '[data] Forced data preparation requested.\n'
  else
    printf '[data] Missing %s required artifact(s):\n' "${#missing[@]}"
    print_list '  - ' "${missing[@]}"
  fi

  data_command=(python -m src.prepare_data --config "${data_config_file}" --overwrite)
  printf '[data] Running data preparation:\n'
  printf '  %s\n' "${data_command[*]}"
  "${data_command[@]}"

  printf '[data] Re-checking data artifacts after preparation\n'
  mapfile -t missing < <(missing_data_artifacts "${data_config_file}" "${requested_corpora[@]}")
  if [[ "${#missing[@]}" -ne 0 ]]; then
    printf '[data] Data preparation finished, but %s required artifact(s) are still missing:\n' "${#missing[@]}" >&2
    print_list '  - ' "${missing[@]}" >&2
    exit 1
  fi

  printf '[data] Data artifacts are ready.\n'
}

select_generated_configs() {
  local config_dir="$1"
  local include_base="$2"
  local -n output_array="$3"
  shift 3
  local suite_patterns=("$@")
  local config_path
  local config_file

  output_array=()
  while IFS= read -r config_path; do
    config_file="$(basename "${config_path}")"

    if [[ "${config_file}" == "${model_key}_base_"*.yaml ]]; then
      if [[ "${include_base}" == "true" ]] && matches_any_pattern "${config_file}" "${suite_patterns[@]}"; then
        output_array+=("${config_path}")
      fi
      continue
    fi

    if ! matches_any_pattern "${config_file}" "${method_patterns[@]}"; then
      continue
    fi

    if ! matches_any_pattern "${config_file}" "${data_patterns[@]}"; then
      continue
    fi

    if ! matches_any_pattern "${config_file}" "${suite_patterns[@]}"; then
      continue
    fi

    output_array+=("${config_path}")
  done < <(find "${config_dir}" -maxdepth 1 -type f -name '*.yaml' | sort)
}

model_key=""
profile="experiments"
data_config_file="configs/data/gemma4_cpt.yaml"
eval_matrix_file=""
env_file=".env"
cleanup_optimizers="true"
cleanup_checkpoints="true"
overwrite="true"
dry_run="false"
force_data_prep="false"
eval_base="true"
skip_evaluation="false"
skip_analysis="false"
benchmark_language="all"
data_configs=()
methods=()
eval_suites=()

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --model)
      model_key="${2:-}"
      shift 2
      ;;
    --profile)
      profile="${2:-}"
      shift 2
      ;;
    --data|--corpus|--config)
      data_configs+=("${2:-}")
      shift 2
      ;;
    --data-config-file)
      data_config_file="${2:-}"
      shift 2
      ;;
    --method)
      methods+=("${2:-}")
      shift 2
      ;;
    --eval-matrix)
      eval_matrix_file="${2:-}"
      shift 2
      ;;
    --eval-suite)
      eval_suites+=("${2:-}")
      shift 2
      ;;
    --no-eval-base)
      eval_base="false"
      shift
      ;;
    --skip-evaluation)
      skip_evaluation="true"
      shift
      ;;
    --skip-analysis)
      skip_analysis="true"
      shift
      ;;
    --benchmark-language)
      benchmark_language="${2:-}"
      shift 2
      ;;
    --env-file)
      env_file="${2:-}"
      shift 2
      ;;
    --cleanup-checkpoints)
      cleanup_checkpoints="${2:-}"
      shift 2
      ;;
    --cleanup-optimizers)
      cleanup_optimizers="${2:-}"
      shift 2
      ;;
    --no-overwrite)
      overwrite="false"
      shift
      ;;
    --force-data-prep)
      force_data_prep="true"
      shift
      ;;
    --dry-run)
      dry_run="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${model_key}" ]]; then
  printf 'Missing required argument: --model\n\n' >&2
  usage >&2
  exit 2
fi

cd "$(repo_root)"

if [[ ! -f "${data_config_file}" ]]; then
  printf 'Data preparation config not found: %s\n' "${data_config_file}" >&2
  exit 1
fi

mapfile -t requested_corpora < <(selected_corpora "${data_config_file}" "${data_configs[@]}")
printf '[step 1/7] Resolved requested data corpora\n'
print_list '  - ' "${requested_corpora[@]}"
check_or_prepare_data "${data_config_file}" "${requested_corpora[@]}"

prepare_command=(python -m src.generate_train_configs --model "${model_key}" --profile "${profile}")
if [[ "${overwrite}" == "true" ]]; then
  prepare_command+=(--overwrite)
fi

printf '[step 3/7] Preparing CPT configs\n'
printf '[cpt] %s\n' "${prepare_command[*]}"
"${prepare_command[@]}"

config_dir="configs/train/generated/${profile}/${model_key}"
if [[ ! -d "${config_dir}" ]]; then
  printf 'Generated config directory not found: %s\n' "${config_dir}" >&2
  exit 1
fi

method_patterns=()
for method in "${methods[@]}"; do
  method_patterns+=("$(normalize_method_pattern "${model_key}" "${method}")")
done

data_patterns=()
for data_config in "${data_configs[@]}"; do
  data_patterns+=("$(normalize_data_pattern "${data_config}")")
done

selected_configs=()
select_generated_configs "${config_dir}" "false" selected_configs

if [[ "${#selected_configs[@]}" -eq 0 ]]; then
  printf 'No generated training configs matched the requested filters.\n' >&2
  printf 'Model: %s\nProfile: %s\nConfig directory: %s\n' "${model_key}" "${profile}" "${config_dir}" >&2
  exit 1
fi

printf '[cpt] Selected %s training config(s)\n' "${#selected_configs[@]}"
printf '  %s\n' "${selected_configs[@]}"

if [[ "${dry_run}" == "true" ]]; then
  printf '[cpt] Dry run requested. Training, evaluation, and analysis will not be launched.\n'
  exit 0
fi

printf '[step 4/7] Launching CPT training\n'
python -m src.train_config \
  --env-file "${env_file}" \
  --cleanup-optimizers "${cleanup_optimizers}" \
  --cleanup-checkpoints "${cleanup_checkpoints}" \
  "${selected_configs[@]}"

if [[ "${skip_evaluation}" == "true" ]]; then
  printf '[eval] Evaluation skipped by request.\n'
else
  if [[ -z "${eval_matrix_file}" ]]; then
    printf '[step 5/7] Resolving evaluation matrix\n'
    eval_matrix_file="$(eval_matrix_for_model "${model_key}")"
  fi

  if [[ ! -f "${eval_matrix_file}" ]]; then
    printf 'Evaluation matrix not found: %s\n' "${eval_matrix_file}" >&2
    exit 1
  fi

  eval_prepare_command=(python -m src.generate_eval_configs --matrix "${eval_matrix_file}" --profile "${profile}")
  if [[ "${overwrite}" == "true" ]]; then
    eval_prepare_command+=(--overwrite)
  fi

  printf '[step 5/7] Preparing evaluation configs\n'
  printf '[eval] %s\n' "${eval_prepare_command[*]}"
  "${eval_prepare_command[@]}"

  eval_config_dir="configs/evaluation/generated/${profile}/${model_key}"
  if [[ ! -d "${eval_config_dir}" ]]; then
    printf 'Generated evaluation config directory not found: %s\n' "${eval_config_dir}" >&2
    exit 1
  fi

  eval_suite_patterns=()
  for suite in "${eval_suites[@]}"; do
    eval_suite_patterns+=("$(normalize_eval_suite_pattern "${suite}")")
  done

  selected_eval_configs=()
  select_generated_configs "${eval_config_dir}" "${eval_base}" selected_eval_configs "${eval_suite_patterns[@]}"

  if [[ "${#selected_eval_configs[@]}" -eq 0 ]]; then
    printf 'No generated evaluation configs matched the requested filters.\n' >&2
    printf 'Model: %s\nProfile: %s\nConfig directory: %s\n' "${model_key}" "${profile}" "${eval_config_dir}" >&2
    exit 1
  fi

  printf '[eval] Selected %s evaluation config(s)\n' "${#selected_eval_configs[@]}"
  printf '  %s\n' "${selected_eval_configs[@]}"

  printf '[step 6/7] Running evaluations\n'
  for eval_config in "${selected_eval_configs[@]}"; do
    printf '[eval] %s\n' "${eval_config}"
    python -m src.eval_config "${eval_config}"
  done
fi

if [[ "${skip_analysis}" == "true" ]]; then
  printf '[analysis] Evaluation result analysis skipped by request.\n'
else
  analysis_command=(
    python -m src.analyze_eval_results
    --model "${model_key}"
    --profile "${profile}"
    --benchmark-language "${benchmark_language}"
  )
  printf '[step 7/7] Analyzing evaluation results\n'
  printf '[analysis] %s\n' "${analysis_command[*]}"
  "${analysis_command[@]}"
fi
