# Add a New LM-Eval Task

Local lm-eval task definitions live under `evaluation/lm_eval_tasks/`, and evaluation 
matrices expose that directory through `task_include_path`. To add a task, 
create a new YAML file whose `task` value is unique and descriptive:

```text
evaluation/lm_eval_tasks/<task_name>.yaml
```

For a local JSONL multiple-choice task, use the `json` dataset loader and point 
`dataset_kwargs.data_files.test` to the evaluation file:

```yaml
task: new_task
dataset_path: json
dataset_name: null
dataset_kwargs:
  data_files:
    test: data/evaluation/new_task.jsonl
test_split: test
output_type: multiple_choice
doc_to_text: "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:"
doc_to_choice:
  - A
  - B
  - C
  - D
doc_to_target: gold
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
  - metric: acc_norm
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
```

If the raw JSONL rows do not already match lm-eval's expected fields, add a 
preprocessing function to `evaluation/lm_eval_tasks/utils.py` and reference it 
from the task file:

```yaml
process_docs: !function utils.process_new_task_docs
```

For a plain perplexity task over a post-training evaluation JSONL file with a 
`text` column, use `loglikelihood_rolling`. The active Guarani perplexity task 
is `guarani_coreguapa_perplexity` and reads `data/evaluation/coreguapa_identified_all.jsonl`.

The Guarani MGSM, Global MMLU Lite, and WNLI tasks prefer their local files under
`data/evaluation/`. If any required file for one of these tasks is missing, the
runner loads that task's complete dataset from its corresponding Guaran-IA
Hugging Face repository.

INCLUDE 2.0 has zero-shot Guarani and Spanish tasks backed by the `guarani` and
`spanish` configs in `epfl-nlp/include-2.0`. They use the paper's country-aware
four-choice prompt and report strict and tolerant exact-match scores. The
`include_2_0` prompt includes the paper-style answer-format instruction inline,
so it also works with tokenizers that do not define a chat template.

```yaml
task: new_perplexity_task
dataset_path: json
dataset_name: null
dataset_kwargs:
  data_files:
    test: data/evaluation/<dataset>.jsonl
test_split: test
output_type: loglikelihood_rolling
doc_to_text: ""
doc_to_target: "{{text}}"
metric_list:
  - metric: word_perplexity
  - metric: byte_perplexity
  - metric: bits_per_byte
metadata:
  version: 1.0
```

After adding the task file, include the task in the relevant evaluation matrix suite:

```yaml
suites:
  new_suite:
    tasks:
      - guarani_new_multiple_choice
    log_samples: true
```

Then add the suite name to any profile that should run it:

```yaml
profiles:
  experiments:
    suites:
      - global_mmlu_lite
      - perplexity
      - new_suite
```

Regenerate evaluation configs from the updated matrix:

```bash
python -m src.generate_eval_configs --matrix configs/evaluation/<name>_matrix.yaml --overwrite
```

Run one generated config first to validate the task before launching a full profile:

```bash
python -m src.eval_config configs/evaluation/generated/smoke/<model_key>/<config_name>.yaml
```

If the new task introduces metrics that are not accuracy or perplexity-like, review 
`src/analyze_eval_results.py` before using percentage improvement columns. The 
analyzer treats metrics containing `perplexity`, `bits_per_byte`, or `loss` as 
lower-is-better and treats other numeric metrics as higher-is-better.

Instruction-following tasks such as IFEval should be placed in the evaluation 
matrix's `instruction` suite. Use `include_instruction_tasks` and `variant_overrides` 
in the matrix to decide which profiles and variant kinds receive those tasks.

Tasks that execute code, such as HumanEval, require `confirm_run_unsafe_code: true` 
in the suite or generated config. Keep that option off for normal multiple-choice, 
QA, translation, and perplexity tasks.
