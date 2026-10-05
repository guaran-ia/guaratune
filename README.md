# GuaraTune

<p align="center">
  <strong>Framework to build the LLMs of the <a href="https://guarania.org" >GuaranIA</a> project.</strong>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#installation">Installation</a> •
   <a href="#quick-start-recommended">Quick start</a> •
   <a href="#1-data-preparation">Data preparation</a> •
   <a href="#2-continued-pre-training-cpt">Continued pre-training (CPT)</a> •
   <a href="#3-evaluation">Evaluation</a> •
   <a href="#citation">Citation</a>
</p>

---

## Overview

This repository contains the code, data, and configurations developed to
conduct both full and LoRA **continued pre-training (CPT)** of state-of-the-art 
open-source base models (check [supported models](#supported-models)) on the 
[Kuatia](https://huggingface.co/datasets/guaran-ia/kuatia) corpus, which is 
a Guarani/Jopara-based corpus created by the [GuaranIA](https://guarania.org) project.

### Pipeline

The full training pipeline consists of four steps:

1. Data preparation
2. CPT execution
3. Model evaluation
4. Performance analysis

Before starting, follow the installation instructions below.

### Supported Models

Currently, the framework supports the training and evaluation of the following 
Gemma 4 variants. 

| Model                        | Parameters | Requiered GPU Memory |
| ---------------------------- | ---------- | ------------------ | 
| google/gemma-4-E2B           | 2B         | ~55 GB (bf16)      |
| google/gemma-4-E4B           | 4B         | ~80 GB (bf16)      |
| google/gemma-4-12B           | 12B        | ~93 GB (bf16)*     |

* Text-only CPT, freezing audio and vision.

> [!Note]
> New models can be added by following the instructions described in 
> [configs/train/README.md](configs/train/README.md).

## Installation

> [!Note]
> The supported runtime is Python 3.12 on Linux with an NVIDIA GPU. The provided GPU
> dependency file targets CUDA 12.8. Install the matching PyTorch wheel set for a
> different CUDA version or platform before installing the project lock. The pipeline
> script requires Bash 4.3 or later; verify it with `bash --version`.

1. Create and activate a fresh Python 3.12 environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

2. Install the CUDA 12.8 PyTorch build:

```bash
python -m pip install -r requirements-gpu-cu128.txt
```

> [!Important]
> Do not use `requirements-gpu-cu128.txt` on macOS or a CPU-only environment. Use the
> [PyTorch installation selector](https://pytorch.org/get-started/locally/) to install
> a compatible wheel set, then continue with the next step.

3. Install the pinned runtime lock:

```bash
python -m pip install -r requirements.txt
```

4. Install development tooling when changing the project:

```bash
python -m pip install -r requirements-dev.txt
```

5. Rename the file `.env.sample` to `.env` and set local secrets:

```bash
HF_TOKEN=...
WANDB_API_KEY=...
```

6. Verify the environment:

```bash
python -m src.check_environment --require-cuda
```

For full-parameter multi-GPU runs that enable DeepSpeed in a training matrix, install
DeepSpeed separately on the GPU VM:

```bash
python -m pip install deepspeed
```

### Dependency Reproducibility

`requirements.in` is the human-maintained list of direct runtime dependencies.
`requirements.txt` is its generated, fully pinned lock. `requirements-dev.in` and
`requirements-dev.txt` extend that lock with development tools. The CUDA-specific
PyTorch packages are intentionally kept in `requirements-gpu-cu128.txt`, because
their wheel index and build tags depend on the target platform.

After changing a `.in` file, regenerate both locks from a development environment:

```bash
python scripts/compile_requirements.py
```

The lock compiler uses `pip-tools==7.5.2` and `pip<26`, as pinned in
`requirements-dev.in`. Do not edit generated `.txt` lock files by hand.

### LlamaFactory and LM-Eval

Training employs [LlamaFactory](https://github.com/hiyouga/LlamaFactory) while the
evaluation is based on [LM-Eval](https://github.com/EleutherAI/lm-evaluation-harness).
The pip dependency locks pin both upstream frameworks and their Python dependencies.

| Upstream project | Pinned commit | Role |
| --- | --- | --- |
| [LlamaFactory](https://github.com/hiyouga/LlamaFactory) | [`ef2d8f9`](https://github.com/hiyouga/LlamaFactory/commit/ef2d8f9da66ef13378dc8068dabc99cfe89a9736) | Continued pre-training |
| [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | [`f4d4b3d`](https://github.com/EleutherAI/lm-evaluation-harness/commit/f4d4b3de3ee6741a7151a9fe74945ee515262f4c) | Model evaluation |

## Quick start (recommended)

Run the full CPT pipeline, including data checks, CPT config generation, training, 
evaluation, and result analysis, by executing the following command

```bash
scripts/run_pipeline.sh --model <model_key>
```

### CLI reference

| Argument | Default | Description |
| --- | --- | --- |
| `--model MODEL_KEY` | Required | Model key declared in `configs/train/*_cpt_matrix.yaml` (e.g., gemma4_4b). |
| `--profile PROFILE` | `experiments` | Training and evaluation profile to generate and run. |
| `--data DATA_CONFIG` | All data configurations | Data configuration to run, for example `C1` or `C1_kuatia`. Repeat to run multiple data configurations. |
| `--corpus DATA_CONFIG` | All data configurations | Alias for `--data`. |
| `--config DATA_CONFIG` | All data configurations | Alias for `--data`. |
| `--data-config-file PATH` | `configs/data/gemma4_cpt.yaml` | Data preparation config file used to resolve corpora and prepare missing data artifacts. |
| `--method METHOD` | All CPT methods | CPT method to run. Supported forms include `full`, `lora`, `lora64`, `lora_r64`, `lora:64`, and quoted forms such as `"lora 64"`. Repeat to run multiple methods. |
| `--eval-matrix PATH` | Auto-detected by model key | Evaluation matrix file. When omitted, the script finds the matrix in `configs/evaluation/*_eval_matrix.yaml` whose `model.key` matches `--model`. |
| `--eval-suite SUITE` | All evaluation suites | Evaluation suite to run, for example `perplexity` or `global_mmlu_lite`. Repeat to run multiple suites. |
| `--no-eval-base` | `False` | Exclude base-model evaluations from the selected evaluation configs. |
| `--skip-evaluation` | `False` | Skip evaluation config generation and execution. |
| `--skip-analysis` | `False` | Skip evaluation result analysis. |
| `--benchmark-language LANG` | `all` | Benchmark language for the comparison table produced by `src.analyze_eval_results`. Supported values are defined by that script. |
| `--env-file PATH` | `.env` | Local secret env file forwarded to training through `src.train_config`. |
| `--cleanup-checkpoints BOOL` | `true` | Remove checkpoint directories after each successful training run; retain the final root model. |
| `--cleanup-optimizers BOOL` | `true` | Pass optimizer cleanup setting to `src.train_config`. |
| `--force-data-prep` | `False` | Run data preparation even if required artifacts already exist. |
| `--no-overwrite` | `False` | Do not overwrite generated training or evaluation configs during preparation. |
| `--dry-run` | `False` | Print selected training configs without launching training, evaluation, or analysis. |
| `-h`, `--help` | `False` | Show script help and exit. |

### Examples

To run only selected configurations:

```bash
scripts/run_pipeline.sh --model <model_key> --data C1 --method full
scripts/run_pipeline.sh --model <model_key> --data C1 --data C2 --method lora64 --method lora512
scripts/run_pipeline.sh --model <model_key> --data C1 --data C2 --method lora64 --method lora512
scripts/run_pipeline.sh --model <model_key> --data C1 --method full --eval-suite perplexity
```

## Step by step execution

If a greater control over each step of the pipeline is preferred, the pipeline 
can also be run step by step.

### 1. Data preparation

To prevent training and validation leakage, the Kuatia train and validation splits were defined manually in `data/kuatia_config.yaml`. Before splitting, we excluded corpora included 
in Kuatia but intended for benchmarking, such as Flores-200, Belebele, and MultiWikiQA, as well as the instruction-tuning corpus Alpaca. We also excluded the AmericasNLP corpora, [Glot500](https://huggingface.co/datasets/cis-lmu/Glot500), and [MaLA-monolingual](https://huggingface.co/datasets/MaLA-LM/mala-monolingual-split) because they are sourced from Flores-200, which provides 
the passages used in the Belebele benchmark.

To avoid "catastrophic forgetting" where the model forgets its old knowledge as 
it learns new things, CPT is proposed to be conducted through several data mixture
configurations that combine Guarani with Spanish and English. A refined 
version of [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), 
the collection of high-quality educational web pages developed by Hugging Face, is 
used to include English in the training process. Similarly, a refined version of the 
Spanish translation of [FineWeb-Edu](https://huggingface.co/datasets/TokenHaven/FineWeb-Edu-Spanish), is used in some configurations of the training set, as shown below

- `C1_kuatia`: Kuatia.
- `C2_kuatia_no_synthetic`: Kuatia but excluding synthetic documents.
- `C3_kuatia_es20`: `C1` plus Spanish FineWeb-Edu at 20% of `C1` tokens.
- `C4_kuatia_no_synthetic_es20`: `C2` plus Spanish FineWeb-Edu at 20% of `C2` tokens.
- `C5_kuatia_en20`: `C1` plus English FineWeb-Edu at 20% of `C1` tokens.
- `C6_kuatia_no_synthetic_en20`: `C2` plus English FineWeb-Edu at 20% of `C2` tokens.
- `C7_kuatia_es10_en10`: `C1` plus Spanish FineWeb-Edu at 10% and English FineWeb-Edu at 10% of `C1` tokens.
- `C8_kuatia_no_synthetic_es10_en10`: `C2` plus Spanish FineWeb-Edu at 10% and English FineWeb-Edu at 10% of `C2` tokens.

FineWeb-Edu samples, both English and Spanish, are fixed, seeded, full-document samples. 
The configured percentages are approximate token proportions: the sampler keeps complete 
documents and stops after reaching the requested proportion, so the final token count can 
be slightly above the requested amount. Train and validation FineWeb-Edu selections are 
stored separately so the same selected document is not used in both splits.

> [!IMPORTANT]  
> The data preparation procedure is config-backed and can be executed by running 
> the following command:

```bash
python -m src.prepare_data --config configs/data/gemma4_cpt.yaml --overwrite
```

This writes:

- `data/train/C1_kuatia.jsonl`
- `data/train/C2_kuatia_no_synthetic.jsonl`
- `data/train/C3_kuatia_es20.jsonl`
- `data/train/C4_kuatia_no_synthetic_es20.jsonl`
- `data/train/C5_kuatia_en20.jsonl`
- `data/train/C6_kuatia_no_synthetic_en20.jsonl`
- `data/train/C7_kuatia_es10_en10.jsonl`
- `data/train/C8_kuatia_no_synthetic_es10_en10.jsonl`
- `data/validation/C1_kuatia.jsonl`
- `data/validation/C2_kuatia_no_synthetic.jsonl`
- `data/validation/C3_kuatia_es20.jsonl`
- `data/validation/C4_kuatia_no_synthetic_es20.jsonl`
- `data/validation/C5_kuatia_en20.jsonl`
- `data/validation/C6_kuatia_no_synthetic_en20.jsonl`
- `data/validation/C7_kuatia_es10_en10.jsonl`
- `data/validation/C8_kuatia_no_synthetic_es10_en10.jsonl`
- `data/source_revisions.lock.json`
- `data/selections/*.selection.jsonl.gz`
- `data/manifests/*.manifest.json`
- `data/dataset_info.json`

Each final corpus is split into a training file under `data/train/` and a validation 
file under `data/validation/`. The source revision lock records the exact Hugging Face 
dataset commits used by a preparation run. Generated manifest files include the 
config path plus train/validation corpus and component accounting for each generated 
JSONL file.

> [!NOTE]
> Token counts are computed with the tokenizer configured in `configs/data/gemma4_cpt.yaml`.

#### CLI reference

| Argument | Default | Description |
| --- | --- | --- |
| `--output-dir` | `None` | Override the output directory defined in the data config. |
| `--overwrite` | `False` | Replace existing generated data artifacts. |
| `--preflight-only` | `False` | Print source and corpus metadata and exit without writing artifacts. |

#### Add new datasets

New datasets can be added to the pre-defined data mixture configuration by following 
the instructions in [data/README.md](data/README.md). Also, a totally new data 
configuration can be included by implementing the steps listed in 
[configs/data/README.md](configs/data/README.md).

### 2. Continued Pre-Training (CPT)

> [!Important]
> To prepare CPT, generate the training configurations from the target model matrix by running:

```bash
python -m src.generate_train_configs --model <model_key> --overwrite
```

This creates:

- `configs/train/generated/smoke/<model_key>/*.yaml`
- `configs/train/generated/experiments/<model_key>/*.yaml`


The smoke profile contains one short LoRA run on `C1_kuatia`, while the experiments profile contains:
- full-parameter CPT over `C1_kuatia` through `C8_kuatia_no_synthetic_es10_en10`
- LoRA CPT over the same corpora with ranks `64`, `128`, `256`, and `512`

> [!Note]
> Training configs report to [Weights & Biases](https://wandb.ai) by default through 
> `report_to: wandb`. W&B defaults live in the matrix `reporting` block and are 
> generated into `configs/train/generated/wandb.env`. The project is intentionally 
> broad, while `WANDB_RUN_GROUP` separates model families. Set `WANDB_API_KEY` in 
> `.env` to authenticate into Weights & Biases before training. Override `WANDB_ENV_FILE` 
> to use a different env file. Use `WANDB_MODE=offline` for disconnected runs, 
> then sync later with `wandb sync`.

##### CLI reference

| Argument | Default | Description |
| --- | --- | --- |
| `--model` | Required | Model key (e.g., gemma4_4b) whose training matrix is stored in `configs/train`. |
| `--profile` | All profiles | Profile to generate. Repeat the argument to generate multiple profiles. |
| `--overwrite` | `False` | Replace existing generated YAML files and W&B environment file. |

>[!Note]
>New training configurations can be added by following the instructions presented 
>in [configs/train/README.md](configs/train/README.md).

#### Run CPT from profile

After preparing the CPT configuration, CPT experiments can be run by model-scoped 
profile. 

> [!Important]
> To execute an entire profile for one model, run

```bash
python -m src.train_profile experiments --model <model_key>
```

Generated profile runs can skip specific configs by indicating the: 

- Filename stem

```bash
python -m src.train_profile experiments --model <model_key> --exclude <model_key>_full_C1_kuatia
```

- Filename

```bash
python -m src.train_profile experiments --model <model_key> --exclude <model_key>_full_C1_kuatia.yaml
```

- Full path

```bash
python -m src.train_profile experiments --model <model_key> --exclude configs/train/generated/experiments/<model_key>/<model_key>_full_C1_kuatia.yaml
```

- Glob pattern:

```bash
python -m src.train_profile experiments --model <model_key> --exclude '<model_key>_full_*'
```

##### CLI reference

| Argument | Default | Description |
| --- | --- | --- |
| `profile` | `experiments` | Generated training profile to run. |
| `--model` | `None` | Model key to run inside the generated profile, for example `gemma4_4b`. |
| `--env-file` | `.env` | Local secret environment file forwarded to each training run. |
| `--exclude` | None | Config filename, stem, path, or glob pattern to skip. Repeat for multiple exclusions. |
| `--cleanup-checkpoints BOOL` | `true` | Remove checkpoint directories after each successful training run; retain the final root model. |
| `--cleanup-optimizers` | `True` | Remove `optimizer.pt` files after each training run to save disk space. |

>[!Note]
>Training wrappers remove `checkpoint-N` directories by default only after the
>training process exits successfully. Failed or interrupted runs retain their
>checkpoints for recovery. The final model, tokenizer, configuration, and results
>in the run's root directory are preserved. To retain checkpoints with their optimizer state, 
>pass both `--cleanup-checkpoints false --cleanup-optimizers false` to `src.train_config`,
>`src.train_profile`, or `scripts/run_pipeline.sh`. Using only
>`--cleanup-checkpoints false` retains checkpoint directories but still applies
>the existing optimizer cleanup.

#### Run CPT from config

Alternatively, an individual CPT experiment can be run by executing:

```bash
python -m src.train_config configs/train/generated/experiments/<model_key>/<model_key>_full_C1_kuatia.yaml
```

Also run a group of configs by passing a quoted glob pattern:

```bash
python -m src.train_config 'configs/train/generated/experiments/<model_key>/<model_key>_lora_r64_*'
```

##### CLI reference

| Argument | Default | Description |
| --- | --- | --- |
| `config_patterns` | Required | One or more generated LLaMA Factory training config paths or glob patterns. |
| `--env-file` | `.env` | Local secret environment file containing values such as `HF_TOKEN` and `WANDB_API_KEY`. |
| `--cleanup-checkpoints BOOL` | `true` | Remove checkpoint directories after each successful training run; retain the final root model. |
| `--cleanup-optimizers` | `True` | Remove `optimizer.pt` files after training completes to save disk space. |

> [!Tip]
> For multi-GPU or DeepSpeed runs, set the LLaMA Factory torchrun environment 
> variables before launching: `CUDA_VISIBLE_DEVICES=0,1,2,3 FORCE_TORCHRUN=1`

> [!Note]
> Tune the target model matrix under `configs/train/` for the actual VM memory 
> before long runs. Set `model.cutoff_len` as the model-level default, or override 
> it per method with `defaults.full.cutoff_len` or `defaults.lora.cutoff_len`. The 
> other likely knobs are `gradient_accumulation_steps`, `flash_attn`, `deepspeed`, 
> and the full/LoRA learning rates.

### 3. Evaluation

Evaluation is based on analyzing the model performance on state-of-the-art downstream
tasks such as: [Flores 200](https://huggingface.co/datasets/facebook/flores), 
[Belele](https://huggingface.co/datasets/facebook/2M-Belebele), 
[MultiWiki-QA](https://huggingface.co/datasets/alexandrainst/multi-wiki-qa), 
Global MMLU Lite, MGSM, and WLNI. For Flores 200, Belele, and MultiWiki-QA the Guarani 
split of the set is used in the evaluation tasks while for Global MMLU Lite, MGSM, 
and WLNI a Guarani version were created by language experts and
are available at `data/evaluation`. 

We also evaluate the models on standard English benchmarks and their Spanish translations, including [ARC](gemma4-4_eval_matrix), [PIQA](https://huggingface.co/datasets/ybisk/piqa), [HellaSwag](https://huggingface.co/datasets/Rowan/hellaswag), and [WinoGrande](https://huggingface.co/datasets/allenai/winogrande). The full list is in `configs/evaluation/gemma4-4_eval_matrix.yaml`.

>[!Warning]
> The evaluation `evaluation/lm_eval_tasks/guarani_coreguapa_perplexity.yaml` 
> depends on a proprietary dataset that cannot be publicly released. The task is 
> ignored if the set is not available at `data/evaluation`.


> [!Important]
> As the first step, generate the evaluation configs from the target model matrix by running:

```bash
python -m src.generate_eval_configs --matrix configs/evaluation/<model_key>_eval_matrix.yaml --overwrite
```

This writes config files under:

- `configs/evaluation/generated/smoke/<model_key>/`
- `configs/evaluation/generated/experiments/<model_key>/`

> [!Note]
> The local lm-eval task definitions live in `evaluation/lm_eval_tasks/`:
> - `guarani_global_mmlu_lite` reads `data/evaluation/gmlgnt.jsonl`
> - `guarani_coreguapa_perplexity` reads `data/evaluation/coreguapa_identified_all.jsonl`
> - Guarani, English, and Spanish benchmark tasks are grouped in the evaluation matrix suites.

Evaluation results, samples, and request caches are stored under `outputs/evaluation/`. 

##### CLI reference

| Argument | Default | Description |
| --- | --- | --- |
| `--matrix` | Required | Evaluation matrix YAML file used to generate lm-evaluation-harness configs. |
| `--profile` | All profiles | Profile to generate. Repeat the argument to generate multiple profiles. |
| `--overwrite` | `False` | Replace existing generated YAML files. |

> [!Important]
> To run an evaluation from a config, execute:

```bash
python -m src.eval_config configs/evaluation/generated/smoke/<model_key>/<model_key>_base_global_mmlu_lite.yaml
```

##### CLI reference

| Argument | Default | Description |
| --- | --- | --- |
| `config_path` | Required | Generated lm-evaluation-harness config YAML file to run. |

> [!Important]
> Alternatively, a full generated profile can be executed:

```bash
python -m src.eval_profile experiments --model <model_key>
```

Generated evaluation profiles also support exclusions:

```bash
python -m src.eval_profile experiments --model <model_key> --exclude '<model_key>_full_C1_kuatia_*'
```

> [!Note]
> The generated experiment profile evaluates the base model, full CPT checkpoints, 
> and LoRA adapters against the benchmark suites listed in 
> `configs/evaluation/<model_key>_matrix.yaml`. 
> 
> Instruction-following tasks live in the `instruction` suite and are controlled 
> from the matrix through `include_instruction_tasks` and `variant_overrides`. 
> 
> The full and LoRA configs expect training outputs under the model-specific 
> `outputs/train/<model_key>/experiments` directory.

##### CLI reference

| Argument | Default | Description |
| --- | --- | --- |
| `profile` | `experiments` | Generated evaluation profile to run. |
| `--model` | `None` | Model key to run inside the generated profile, for example `gemma4_12b`. |
| `--exclude` | None | Config filename, stem, path, or glob pattern to skip. Repeat for multiple exclusions. |


> [!Note]
> Check [configs/evaluation/README.md](configs/evaluation/README.md) for instructions 
> on how to add new evaluation configurations. Also, new LM-Eval tasks can be 
> included by following steps listed in [evaluation/lm_eval_tasks/README.md](evaluation/lm_eval_tasks/README.md).

#### Analyze evaluation results

> [!Important]
> After evaluation runs finish, summarize all available results for one or more models:

```bash
python -m src.analyze_eval_results --model <model_key> --model another_model_key --profile experiments
```

The evaluation profile runner writes per-variant evaluation results; it does not
generate summary tables. Run the analyzer after the profile finishes to produce the
summary and perplexity comparison reports.

The benchmark comparison table can be restricted to one evaluation language:

```bash
python -m src.analyze_eval_results --model <model_key> --profile experiments --benchmark-language en
```

The analyzer produces:

- `results/evaluation_summary_<model_key>.csv` for one model
- `results/evaluation_summary_combined_<model_keys>.csv` for multiple models
- matching `.md` files with the same filename stem
- `results/evaluation_benchmark_table.md`, a benchmark-by-model table with rounded scores and a final average row
- `results/evaluation_benchmark_table_<language>.md` when `--benchmark-language` is set to `en`, `es`, or `gn`
- `results/evaluation_perplexity_table.md` when multiple model variants are available, comparing base and CPT variants for the selected models

##### CLI reference

| Argument | Default | Description |
| --- | --- | --- |
| `--model` | Required | Model key to analyze. Repeat the option to analyze multiple models. |
| `--evaluation-root` | `outputs/evaluation` | Root directory containing evaluation outputs. |
| `--profile` | All profiles | Evaluation profile to include. Repeat the option to include multiple profiles. |
| `--output-dir` | `results` | Directory where analysis tables are written. |
| `--csv-name` | Auto-generated | CSV output filename. |
| `--markdown-name` | Auto-generated | Markdown summary output filename. |
| `--benchmark-markdown-name` | Auto-generated | Benchmark-by-model markdown output filename. |
| `--benchmark-language` | `all` | Benchmark language to include in the comparison table. Supported values: `all`, `en`, `es`, `gn`. |
| `--perplexity-markdown-name` | Auto-generated | Perplexity-by-model markdown output filename. |
| `--language-average-markdown-name` | Auto-generated | Language-average benchmark markdown output filename. |

## License

This project is licensed under the GNU GPLv3 License. Model weights and corpus 
content are subject to their respective licenses.

## Citation

If you use GuaraTune in research, please cite the repository and identify the
commit or release used:

```bibtex
@software{guarania_guaratune,
  author = {{Saldivar, Jorge}},
  title = {GuaraTune: Framework to Build GuaranIA Language Models},
  url = {https://github.com/guaran-ia/models},
  year = {2026}
}
```

Also cite the base models, datasets, benchmarks, and frameworks used in your
work according to their upstream terms and
[third-party notices](THIRD_PARTY_NOTICES.md).
