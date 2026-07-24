# Guarania Models Framework

This repository contains the code, data, configurations, and outputs that resulted
from conducting continual pretraining (CPT) on state-of-the-art base models. Training
uses the framework [LlamaFactory](https://github.com/hiyouga/LlamaFactory), and evaluation uses the framework [LM-Eval](https://github.com/EleutherAI/lm-evaluation-harness).

The full training pipeline consists of four steps:

1. Data preparation
2. CPT preparation
3. CPT execution
4. Model evaluation

Before starting, follow the installation instructions below.

## 0. Installation

Follow the steps below to install the project dependencies.

> [!Note]
> The current development environment uses Python 3.12.

1. Create and activate a fresh environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

2. Install the training framework LlamaFactory.

```bash
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory
pip install -e .
pip install -r requirements/metrics.txt
```

> [!Note]
> The requirements file also includes pinned editable installs. The generated training configs reference `LlamaFactory/examples/deepspeed/ds_z3_config.json`, so keep the local `LlamaFactory/` checkout at the project root.

**Optional.** For full-parameter multi-GPU training with the current configs, install LlamaFactory's DeepSpeed dependencies on the GPU VM:

```bash
cd LlamaFactory
python -m pip install -r requirements/deepspeed.txt
cd ..
```

3. Install the evaluation framework LM-Eval.

```bash
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
pip install "lm_eval[hf]"
```

4. Install the rest of the project dependencies from `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

5. Rename the file `.env.sample` to `.env` and set local secrets:

```bash
HF_TOKEN=...
WANDB_API_KEY=...
```

6. Check the installation:

```bash
llamafactory-cli --help
python -c "import lm_eval; print('lm_eval ok')"
```

## 1. Dataset preparation

Training is based on a composition of three datasets: [Kuatia](https://huggingface.co/datasets/guaran-ia/kuatia), a Guarani/Jopara-based corpus created by the Guarania project; [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), a collection of high-quality educational web pages developed by Hugging Face; and a Spanish version of [FineWeb-Edu](https://huggingface.co/datasets/Helsinki-NLP/fineweb-edu-translated), automatically translated by the [HPLT](https://hplt-project.org) project.

> [!Note]
> The default Spanish source is [Helsinki-NLP](https://huggingface.co/datasets/Helsinki-NLP/fineweb-edu-translated) with config `spa` and reads directly from the Hugging Face Parquet shards. To use a smaller version, like the one produced by [Token Haven](https://huggingface.co/datasets/TokenHaven/FineWeb-Edu-Spanish), edit the Spanish source block in `configs/data/gemma4_cpt.yaml`:

```yaml
fineweb_edu_es:
  name: fineweb_edu_es
  dataset: TokenHaven/FineWeb-Edu-Spanish
  config: null
  split: train
  text_column: text
  id_column: id
  loader: datasets
  revision: null
```

The mentioned corpora are combined using the following configurations:

- Configuration 1 (`C1`): Kuatia only.
- `C2`: 80% Kuatia, 20% Spanish FineWeb-Edu.
- `C3`: 80% Kuatia, 20% English FineWeb-Edu.
- `C4`: 80% Kuatia, 10% Spanish FineWeb-Edu, 10% English FineWeb-Edu.

FineWeb-Edu samples, both English and Spanish, are fixed, seeded, full-document samples, so they may exceed the target token count slightly. `C4` uses deterministic fixed subsamples of the corresponding 20% Spanish and English samples.

> [!IMPORTANT]  
> To create the dataset configurations, run the following command:

```bash
python -m src.prepare_data --config configs/data/gemma4_cpt.yaml --overwrite
```

This writes:

- `data/processed/C1_kuatia.jsonl`
- `data/processed/C2_kuatia_es20.jsonl`
- `data/processed/C3_kuatia_en20.jsonl`
- `data/processed/C4_kuatia_es10_en10.jsonl`
- `data/evaluation/perplexity/C1_kuatia.jsonl`
- `data/evaluation/perplexity/C2_kuatia_es20.jsonl`
- `data/evaluation/perplexity/C3_kuatia_en20.jsonl`
- `data/evaluation/perplexity/C4_kuatia_es10_en10.jsonl`
- `data/source_revisions.lock.json`
- `data/selections/*.selection.jsonl.gz`
- `data/manifests/*.manifest.json`
- `data/dataset_info.json`

Each final corpus is split into a training file under `data/processed/` and a held-out file under `data/evaluation/perplexity/`. The default held-out split is 1% of documents, assigned by a deterministic hash of the corpus name, document id, and `heldout.seed`, so reruns and reconstruction produce the same train/evaluation split.

Stable recipe settings live in `configs/data/gemma4_cpt.yaml`, including tokenizer, seed, held-out split, dataset sources, component sampling targets, and final corpus composition. The source revision lock records the exact Hugging Face dataset commits used by a preparation run. The manifest files include the config path plus train/held-out corpus and component accounting for each generated JSONL file.

> [!NOTE]
> The large generated corpora under `data/processed/` and generated perplexity held-outs under `data/evaluation/perplexity/` are ignored by Git. To reconstruct them from tracked auxiliary files, use:

```bash
python -m src.prepare_data --reconstruct --config configs/data/gemma4_cpt.yaml --overwrite
```

Reconstruction reads `data/source_revisions.lock.json`, `data/selections/*.selection.jsonl.gz`, and `data/manifests/*.manifest.json`. If any locked Hugging Face dataset revision is no longer accessible, reconstruction stops instead of falling back to the latest dataset version. The held-out perplexity files are reconstructed from the same component ledgers and deterministic split settings.

> [!NOTE]
> Token counts are computed with `google/gemma-4-26B-A4B` without special tokens.

The remaining optional CLI flags are operational or debug controls: `--output-dir`, `--overwrite`, `--reconstruct`, `--preflight-only`, `--max-kuatia-docs`, `--target-scale`, and `--allow-incomplete-samples`.

## 2. CPT preparation

In the first version, CPT is focused on Gemma 4 model variants (12B and 26B) through separate model-specific matrices.

> [!Important]
> To prepare CPT, generate the training configurations from the target model matrix by running:

```bash
python -m src.generate_train_configs --matrix configs/train/gemma4-12_cpt_matrix.yaml --overwrite
```

This creates:

- `configs/train/generated/smoke/<model_key>/*.yaml`
- `configs/train/generated/experiments/<model_key>/*.yaml`


The smoke profile contains one short LoRA run on `C1_kuatia`, while the experiments profile contains the full matrix:
- full-parameter CPT over `C1` to `C4`
- LoRA CPT over `C1` to `C4` with ranks `64`, `128`, `256`, and `512`

> [!Note]
> Training configs report to [Weights & Biases](https://wandb.ai) by default through `report_to: wandb`. W&B defaults live in the matrix `reporting` block and are generated into `configs/train/generated/wandb.env`. The project is intentionally broad, while `WANDB_RUN_GROUP` separates model families. Set `WANDB_API_KEY` in `.env` to authenticate into Weights & Biases before training. The launcher sources `configs/train/generated/wandb.env` when a config has `report_to: wandb`. Override `WANDB_ENV_FILE` to use a different env file. Use `WANDB_MODE=offline` for disconnected runs, then sync later with `wandb sync`.

## 3. Run CPT

After preparing the CPT configuration, CPT experiments can be run both individually or by model-scoped profile. 

> [!Important]
> To execute an entire profile for one model, run

```bash
python -m src.train_profile experiments --model gemma4_12b
```

Generated profile runs can skip specific configs by filename stem, filename, path, or glob pattern:

```bash
python -m src.train_profile experiments --model gemma4_12b --exclude gemma4_12b_full_C1_kuatia
```

Alternatively, an individual CPT experiment can be run by executing:

```bash
python -m src.train_config configs/train/generated/experiments/gemma4_12b/gemma4_12b_full_C1_kuatia.yaml
```

> [!Tip]
> For multi-GPU or DeepSpeed runs, set the LLaMA Factory torchrun environment variables before launching: `CUDA_VISIBLE_DEVICES=0,1,2,3 FORCE_TORCHRUN=1`

> [!Note]
> Tune the target model matrix under `configs/train/` for the actual VM memory before long runs. Set `model.cutoff_len` as the model-level default, or override it per method with `defaults.full.cutoff_len` or `defaults.lora.cutoff_len`. The other likely knobs are `gradient_accumulation_steps`, `flash_attn`, `deepspeed`, and the full/LoRA learning rates.

## 4. Evaluation

The performance of the trained models is evaluated using the framework [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

### Prepare evaluation

> [!Important]
> As the first step, generate the evaluation configs from the target model matrix by running:

```bash
python -m src.generate_eval_configs --matrix configs/evaluation/gemma4-12_eval_matrix.yaml --overwrite
```

This writes config files under:

- `configs/evaluation/generated/smoke/<model_key>/`
- `configs/evaluation/generated/experiments/<model_key>/`

> [!Note]
> The local lm-eval task definitions live in `evaluation/lm_eval_tasks/`:
> - `guarani_global_mmlu_lite` reads `data/evaluation/gmlgnt.jsonl`
> - `guarani_cpt_perplexity_*` reads held-out JSONL files from `data/evaluation/perplexity/`

Evaluation results, samples, and request caches are stored under `outputs/evaluation/`. The launchers use the lm-evaluation-harness Python API.

### Run evaluation

> [!Important]
> To run an evaluation from a config, execute:

```bash
python -m src.eval_config configs/evaluation/generated/smoke/gemma4_12b/gemma4_12b_base_global_mmlu_lite.yaml
```

Alternatively, a full generated profile can be executed:

```bash
python -m src.eval_profile experiments --model gemma4_12b
```

Generated evaluation profiles also support exclusions:

```bash
python -m src.eval_profile experiments --model gemma4_12b --exclude 'gemma4_12b_full_C1_kuatia_*'
```

> [!Note]
> The generated experiment profile evaluates the base model, full CPT checkpoints, and LoRA adapters against a Guarani version of Global-MMLU-Lite and the held-out perplexity splits. The full and LoRA configs expect training outputs under the model-specific `outputs/train/<model_key>/experiments` directory.

### Analyze evaluation results

> [!Important]
> After evaluation runs finish, summarize all available results for one or more models:

```bash
python -m src.analyze_eval_results --model gemma4_12b --model gemma4_26b_a4b --profile experiments
```

The analyzer writes:

- `results/evaluation_summary_gemma4_12b.csv` for one model
- `results/evaluation_summary_combined_gemma4_12b_gemma4_26b_a4b.csv` for multiple models
- matching `.md` files with the same filename stem

The summary has one row per model variant, including `base` when available. Metric columns include Global-MMLU-Lite accuracy metrics and held-out perplexity metrics, plus percentage improvement versus the model's base evaluation when base results exist. Corpus-specific perplexity task names are collapsed into generic metric columns, such as `word_perplexity`, using only the held-out corpus that matches each trained variant's corpus configuration.
