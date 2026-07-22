# Guarania Models

This repository contains the code, data, configurations, and outcome that resulted 
from conducting continual pretraining on state-of-the-art base models. The train 
uses the framework [LlamaFactory](https://github.com/hiyouga/LlamaFactory).

## 1. Dataset preparation

Prepare the four frozen CPT corpora with:

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

The corpus proportions are:

- `C1`: Kuatia only.
- `C2`: 80% Kuatia, 20% Spanish FineWeb-Edu.
- `C3`: 80% Kuatia, 20% English FineWeb-Edu.
- `C4`: 80% Kuatia, 10% Spanish FineWeb-Edu, 10% English FineWeb-Edu.

FineWeb samples are fixed, seeded, full-document samples, so they may exceed the target token count slightly. `C4` uses deterministic fixed subsamples of the corresponding 20% Spanish and English samples. Token counts are computed with `google/gemma-4-26B-A4B` without special tokens.

Each final corpus is split into a training file under `data/processed/` and a held-out file under `data/evaluation/perplexity/`. The default held-out split is 1% of documents, assigned by a deterministic hash of the corpus name, document id, and `heldout.seed`, so reruns and reconstruction produce the same train/evaluation split.

Stable recipe settings live in `configs/data/gemma4_cpt.yaml`, including tokenizer, seed, held-out split, dataset sources, component sampling targets, and final corpus composition. The source revision lock records the exact Hugging Face dataset commits used by a preparation run. The manifest files include the config path plus train/held-out corpus and component accounting for each generated JSONL file.

The large generated corpora under `data/processed/` and generated perplexity held-outs under `data/evaluation/perplexity/` are ignored by Git. To reconstruct them from tracked auxiliary files, use:

```bash
python -m src.prepare_data --reconstruct --config configs/data/gemma4_cpt.yaml --overwrite
```

Reconstruction reads `data/source_revisions.lock.json`, `data/selections/*.selection.jsonl.gz`, and `data/manifests/*.manifest.json`. If any locked Hugging Face dataset revision is no longer accessible, reconstruction stops instead of falling back to the latest dataset version. The held-out perplexity files are reconstructed from the same component ledgers and deterministic split settings.

The default Spanish source is `Helsinki-NLP/fineweb-edu-translated` with config `spa`, read directly from the Hugging Face Parquet shards. To use a smaller version, like the `TokenHaven/FineWeb-Edu-Spanish` sample, edit the Spanish source block in `configs/data/gemma4_cpt.yaml`:

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

The remaining CLI flags are operational or debug controls: `--output-dir`, `--overwrite`, `--reconstruct`, `--preflight-only`, `--max-kuatia-docs`, `--target-scale`, and `--allow-incomplete-samples`.

## 2. CPT preparation

Training configs are generated from `configs/train/gemma4_cpt_matrix.yaml`.

Generate or refresh the LLaMA Factory YAML files with:

```bash
python -m src.generate_train_configs --matrix configs/train/gemma4_cpt_matrix.yaml --overwrite
```

This creates:

- `configs/train/generated/smoke/*.yaml`
- `configs/train/generated/experiments/*.yaml`

The smoke profile contains one short Gemma 4 LoRA run on `C1_kuatia`. The experiments profile contains the full Gemma 4 matrix:

- full-parameter CPT over `C1` to `C4`
- LoRA CPT over `C1` to `C4` with ranks `64`, `128`, `256`, and `512`

Training configs report to Weights & Biases by default through `report_to: wandb`. W&B defaults live in the matrix `reporting` block and are generated into `configs/train/generated/wandb.env`. The project is intentionally broad, while `WANDB_RUN_GROUP` separates model families. Set `WANDB_API_KEY` in `.env` to authenticate into Weights & Biases before training.

The launcher sources `configs/train/generated/wandb.env` when a config has `report_to: wandb`. Override `WANDB_ENV_FILE` to use a different env file. Use `WANDB_MODE=offline` for disconnected runs, then sync later with `wandb sync`.

## 3. Run CPT

Launch an individual CPT experiment

```bash
python -m src.train_config configs/train/generated/experiments/gemma4_26b_a4b_full_C1_kuatia.yaml
```

> For multi-GPU or DeepSpeed runs, set the LLaMA Factory torchrun environment variables before launching: `CUDA_VISIBLE_DEVICES=0,1,2,3 FORCE_TORCHRUN=1`

To run a whole profile (i.e., all different experiments)

```bash
python -m src.train_profile experiments
```

> Tune `configs/train/gemma4_cpt_matrix.yaml` for the actual VM memory before long runs. Set `model.cutoff_len` as the model-level default, or override it per method with `defaults.full.cutoff_len` or `defaults.lora.cutoff_len`. The other likely knobs are `gradient_accumulation_steps`, `flash_attn`, `deepspeed`, and the full/LoRA learning rates.

## 4. Evaluation

Evaluations use [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). See the installation section for environment setup.

Generate evaluation configs from `configs/evaluation/gemma4_eval_matrix.yaml`:

```bash
python -m src.generate_eval_configs --matrix configs/evaluation/gemma4_eval_matrix.yaml --overwrite
```

This writes config files under:

- `configs/evaluation/generated/smoke/`
- `configs/evaluation/generated/experiments/`

The local lm-eval task definitions live in `evaluation/lm_eval_tasks/`:

- `guarani_global_mmlu_lite` reads `data/evaluation/gmlgnt.jsonl`
- `guarani_cpt_perplexity_*` reads held-out JSONL files from `data/evaluation/perplexity/`

Evaluation results, samples, and request caches are stored under `outputs/evaluation/`. The launchers use the lm-evaluation-harness Python API rather than shelling out to the `lm-eval` CLI. Environment loading is config-driven: generated configs include `env_file: .env` and `cache_path: outputs/evaluation/cache/requests`.

Run one evaluation config:

```bash
python -m src.eval_config configs/evaluation/generated/smoke/base_global_mmlu_lite.yaml
```

Run a full generated profile:

```bash
python -m src.eval_profile experiments
```

> The generated experiment profile evaluates the base model, full CPT checkpoints, and LoRA adapters against Guarani Global-MMLU-Lite and the held-out perplexity splits. The full and LoRA configs expect training outputs under `outputs/train/gemma4_26b_a4b/experiments`, matching the CPT training matrix.

## 5. Installation

Use Python 3.11, 3.12, or 3.13. The current development environment uses Python 3.12.

Create and activate a fresh environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Install the project dependencies from `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

> The requirements file includes pinned editable installs for both LlamaFactory and lm-evaluation-harness. The generated training configs also reference `LlamaFactory/examples/deepspeed/ds_z3_config.json`, so keep a local `LlamaFactory/` checkout at the repository root:

```bash
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
```

For full-parameter multi-GPU training with the current configs, install LlamaFactory's DeepSpeed dependencies on the GPU VM:

```bash
cd LlamaFactory
python -m pip install -r requirements/deepspeed.txt
cd ..
```

Set local secrets in `.env`; this file is ignored by Git:

```bash
HF_TOKEN=...
WANDB_API_KEY=...
```

Check the installation:

```bash
llamafactory-cli --help
python -c "import lm_eval; print('lm_eval ok')"
```
