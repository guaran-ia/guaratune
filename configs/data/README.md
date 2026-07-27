# Add New Data Configuration

A new data configuration can be created by copying the existing config and giving the new name:

```bash
cp configs/data/gemma4_cpt.yaml configs/data/<new_config_name>.yaml
```

Then edit the new YAML file. The main fields to review are:

- `output_dir`: where processed corpora, held-out files, manifests, selections, and `dataset_info.json` are written. Use `data` by default otherwise the train and evaluation configs should be updated to read from that directory.
- `tokenizer`: tokenizer used only for token accounting during data preparation.
- `seed`, `shuffle_buffer_size`, and `heldout`: deterministic sampling and train/held-out split settings.
- `sources`: Hugging Face datasets or Parquet-backed sources.
- `components`: reusable selections built from sources.
- `corpora`: final CPT dataset configurations exposed to training and perplexity evaluation.

After editing the new configuration, run data preparation with the new file:

```bash
python -m src.prepare_data --config configs/data/<new_config_name>.yaml --overwrite
```

If the new configuration writes to `output_dir: data`, the generated `data/dataset_info.json` will expose the new `corpora` names to the framework. Add those corpus names to the relevant training matrix profile, such as `profiles.experiments.corpora` in `configs/train/gemma4-12_cpt_matrix.yaml` or `configs/train/gemma4-26_cpt_matrix.yaml`, and regenerate training configs.

```bash
python -m src.generate_train_configs --matrix configs/train/gemma4-12_cpt_matrix.yaml --overwrite
```

For evaluation, add the same corpus names to the relevant evaluation matrix and regenerate evaluation configs:

```bash
python -m src.generate_eval_configs --matrix configs/evaluation/gemma4-12_eval_matrix.yaml --overwrite
```

The current evaluation task definitions expect held-out files under `data/evaluation/perplexity/`. If the new data configuration uses a different `output_dir`, either make that directory the active `data` directory before evaluation or add matching lm-eval task definitions that point to the alternate held-out paths.