# Add New Data Configuration

A new data configuration can be created by copying the existing config, giving 
it a new name:

```bash
cp configs/data/gemma4_cpt.yaml configs/data/<new_config_name>.yaml
```

Then edit the new YAML file. The main fields to review are:

- `output_dir`: where train corpora, validation files, manifests, selections, and `dataset_info.json` are written. Use `data` by default otherwise the train and evaluation configs should be updated to read from that directory.
- `tokenizer`: tokenizer used only for token accounting during data preparation.
- `seed`: deterministic sampling seed used by source selection routines.
- `sources`: Hugging Face datasets or Parquet-backed sources pinned by `commit_id`.
- `corpora`: final CPT dataset configurations exposed to training and in-training validation. 

If the Kuatia corpus will be include in the new configuration, the field `split_config` 
needs to be set with the Kuatia split policy file, currently `data/kuatia_config.yaml`.

After editing the new configuration, run data preparation with the new file:

```bash
python -m src.prepare_data --config configs/data/<new_config_name>.yaml --overwrite
```

If the new configuration writes to `output_dir: data`, the generated `data/dataset_info.json` 
will expose the new `corpora` names to the framework. Add those corpus names to 
the relevant training matrix profile, and regenerate training configs.

```bash
python -m src.generate_train_configs --model <model_key> --overwrite
```

For evaluation, add the same corpus names to the relevant evaluation matrix and 
regenerate evaluation configs:

```bash
python -m src.generate_eval_configs --matrix configs/evaluation/<model_key>_eval_matrix.yaml --overwrite
```

The data preparation step writes in-training validation files under `data/validation/`. 
Post-training evaluation tasks should use files under `data/evaluation/`, so add 
or update lm-eval task definitions separately when a new evaluation set is introduced.
