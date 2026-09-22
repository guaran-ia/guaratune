# Add a New Train Matrix

A new train matrix is useful when adding a new model family, separating hardware-specific settings, or testing a different CPT experiment grid. Start by copying the closest existing matrix:

```bash
cp configs/train/gemma4-12_cpt_matrix.yaml configs/train/<new_model>_cpt_matrix.yaml
```

Then edit the new matrix. The main sections are:

- `generated_config_dir`: root directory for generated YAML files. Keep `configs/train/generated` unless there is a specific reason to separate generated configs.
- `runs_root`: model-specific training output root. Use `outputs/train/<model_key>` so training artifacts stay separated by model.
- `reporting`: W&B defaults used when generating `wandb.env`.
- `model`: model key, Hugging Face model path, revision, remote-code setting, and default `cutoff_len`.
- `defaults.common`: LLaMA Factory arguments shared by full and LoRA CPT runs, including `dataset_dir`, workers, packing, attention backend, logging, saving, warmup, epochs, and precision.
- `defaults.full`: full-parameter CPT settings such as `finetuning_type`, optimizer, learning rate, weight decay, gradient accumulation, and optional DeepSpeed config.
- `defaults.lora`: LoRA CPT settings such as target modules, rank-derived alpha multiplier, dropout, learning rate, and gradient accumulation.
- `profiles`: named experiment sets, usually `smoke` for a short validation run and `experiments` for the full grid.

At minimum, update the model-specific fields:

```yaml
runs_root: outputs/train/<model_key>

reporting:
  wandb_run_group: <model_key>

model:
  key: <model_key>
  model_name_or_path: organization/model-name
  model_revision: main
  trust_remote_code: true
  cutoff_len: 2048
```

Then review `defaults.common.flash_attn` for the target model and GPU stack. Use `sdpa` when FlashAttention-2 is unsupported by the model head dimension or local CUDA environment. Use `fa2` only after a smoke run confirms it works.

Finally, make sure each profile lists only corpora that exist in `data/dataset_info.json`:

```yaml
profiles:
  smoke:
    methods:
      - lora
    corpora:
      - C1_kuatia
    lora_ranks:
      - 64
    overrides:
      max_steps: 5
      logging_steps: 1
      save_steps: 5
      overwrite_output_dir: true

  experiments:
    methods:
      - full
      - lora
    corpora:
      - C1_kuatia
      - C2_kuatia_no_synthetic
      - C3_kuatia_es20
      - C4_kuatia_no_synthetic_es20
      - C5_kuatia_en20
      - C6_kuatia_no_synthetic_en20
      - C7_kuatia_es10_en10
      - C8_kuatia_no_synthetic_es10_en10
    lora_ranks:
      - 64
      - 128
      - 256
      - 512
```

Generate configs from the new matrix:

```bash
python -m src.generate_train_configs --matrix configs/train/<new_model>_cpt_matrix.yaml --overwrite
```

This writes model-scoped configs under `configs/train/generated/<profile>/<model_key>/`.

Generated configs write training artifacts under `outputs/train/<model_key>/<profile>/`. Full CPT runs use `outputs/train/<model_key>/<profile>/full/<corpus>/<run_name>`, while LoRA runs use `outputs/train/<model_key>/<profile>/lora/rank_<rank>/<corpus>`.
