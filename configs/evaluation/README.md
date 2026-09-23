# Add a New Evaluation Matrix

A new evaluation matrix is useful when adding a new model, changing the set of 
evaluated checkpoints, or testing a different suite/profile layout. Start by 
copying the closest existing matrix:

```bash
cp configs/evaluation/gemma4-12_eval_matrix.yaml configs/evaluation/<new_model>_eval_matrix.yaml
```

Then edit the new matrix. The main sections are:

- `generated_config_dir`: root directory for generated lm-eval YAML files. Keep 
`configs/evaluation/generated` unless generated configs need to be isolated.
- `outputs_root`: root directory for evaluation outputs. Keep `outputs/evaluation` 
for the standard analyzer workflow.
- `task_include_path`: local lm-eval task directory. Keep `evaluation/lm_eval_tasks` 
unless task definitions live elsewhere.
- `training_outputs_root`: model-specific training output directory for full 
checkpoints and LoRA adapters.
- `seed`: evaluation seed.
- `model`: base model key, Hugging Face model path, revision, dtype, device, and 
batch settings.
- `defaults`: settings shared by generated lm-eval configs, including request cache, 
few-shot count, sample logging, instruction-task inclusion, and optional limits.
- `suites`: named groups of lm-eval tasks. Use the optional `instruction` suite 
for instruction-following benchmarks such as IFEval.
- `profiles`: named evaluation grids, usually `smoke` for a short validation run 
and `experiments` for the full evaluation set.

At minimum, update the model-specific fields:

```yaml
training_outputs_root: outputs/train/<model_key>/experiments

model:
  key: <model_key>
  model_name_or_path: organization/model-name
  model_revision: main
  trust_remote_code: true
  dtype: bfloat16
  device: cuda:0
  batch_size: auto
  max_batch_size: null
```

Suites define which lm-eval tasks run together. For example:

```yaml
suites:
  global_mmlu_lite:
    tasks:
      - guarani_global_mmlu_lite
    log_samples: true

  perplexity:
    tasks:
      - task1
      - task2
    log_samples: false

  instruction:
    tasks:
      - ifeval
      - ifeval_es
    log_samples: true
```

Profiles define which model variants and suites are evaluated:

```yaml
profiles:
  smoke:
    suites:
      - global_mmlu_lite
    training_corpora:
      - corpus
    include_base: true
    include_full: false
    include_instruction_tasks: false
    lora_ranks:
      - 64
    overrides:
      limit: 5
      batch_size: 1

  experiments:
    suites:
      - global_mmlu_lite
      - perplexity
      - instruction
    training_corpora:
      - corpus1
      - corpus2
    include_base: true
    include_full: true
    include_instruction_tasks: true
    variant_overrides:
      base:
        include_instruction_tasks: false
    lora_ranks:
      - 64
      - 128
    overrides: {}
```

The profile expands to the base model when `include_base: true`, full CPT variants 
when `include_full: true`, and LoRA variants for every pair of `training_corpora` 
and `lora_ranks`. `training_corpora` names the training dataset configurations that 
produced the checkpoints or adapters; it does not select evaluation datasets. The 
generated config names and output paths include the model key, variant, and suite.

Instruction-following benchmarks should live in `suites.instruction`. Set 
`include_instruction_tasks: true` on profiles that should evaluate 
instruction-following ability, and use `variant_overrides` when a specific 
variant kind should behave differently. For example, set `variant_overrides.base.include_instruction_tasks: false` 
when base models should be evaluated only on non-instruction suites. If a matrix 
has an `instruction` suite, the generator also removes those task IDs from 
non-instruction suites so IFEval-style tasks are not duplicated.

Generate configs from the new matrix:

```bash
python -m src.generate_eval_configs --matrix configs/evaluation/<new_model>_eval_matrix.yaml --overwrite
```

This writes model-scoped configs under `configs/evaluation/generated/<profile>/<model_key>/`. 
