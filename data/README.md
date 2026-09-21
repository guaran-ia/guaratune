# Add a Dataset

New source datasets are registered in `configs/data/gemma4_cpt.yaml`. The active data configuration is manifest-backed: each final corpus points to a manifest under `data/manifests/`, and `src.prepare_data` materializes train and validation JSONL files from those manifests. Selection ledgers under `data/selections/` are generated locally during preparation.

The relevant sections are:

- `sources`: Hugging Face datasets or Hugging Face Parquet-backed repositories.
- `corpora`: final CPT dataset configurations, each with a `manifest` path.
- `data/kuatia_config.yaml`: Kuatia corpus split policy, including train, validation, and synthetic corpus membership.
- `data/manifests/*.manifest.json`: reproducible corpus definitions used by data preparation.
- `data/selections/*.selection.jsonl.gz`: local generated document-selection ledgers for Kuatia and FineWeb-Edu additions.

To add a new source, first add it under `sources`:

```yaml
sources:
  new_dataset:
    name: new_dataset
    dataset: organization/dataset-name
    version: null
    loader: datasets
    requested_revision: null
    commit_id: <resolved_hugging_face_commit>
    selection_path: data/selections/new_dataset.selection.jsonl.gz
```

Use `loader: datasets` for sources that work through `datasets.load_dataset(...)`. Use `loader: hf_parquet` when the source should be read directly from Hugging Face Parquet shards under a repository path or language/config directory. `commit_id` pins the exact source revision used for reproducibility.

Then create or update a corpus manifest under `data/manifests/`. The manifest defines:

- `config_name`: final corpus name, for example `C9_kuatia_new20`.
- `config_path`: the split/config file used to create the manifest.
- `tokenizer`: tokenizer used for token accounting.
- `sources`: source metadata, including dataset, version, and commit IDs.
- `splits.train.corpora`: train components to write into `data/train/<config_name>.jsonl`.
- `splits.validation.corpora`: validation components to write into `data/validation/<config_name>.jsonl`.

After the manifest exists, add the corpus to `configs/data/gemma4_cpt.yaml`:

```yaml
corpora:
  C9_kuatia_new20:
    manifest: data/manifests/C9_kuatia_new20.manifest.json
    kuatia:
      include_synthetic: true
    augmentations:
      - source: new_dataset
        ratio: 0.2
        ratio_reference: C1_kuatia
        train_selection_path: data/selections/C9_kuatia_new20.new_dataset.train.selection.jsonl.gz
        validation_selection_path: data/selections/C9_kuatia_new20.new_dataset.validation.selection.jsonl.gz
```

Regenerate the data artifacts:

```bash
python -m src.prepare_data --config configs/data/gemma4_cpt.yaml --overwrite
```

This writes train corpora, validation splits, source component pools under `data/train/components/`, local source revision locks, local selection ledgers, and `data/dataset_info.json`.

The new corpus name must also be added to any training matrix profile that should train on it, for example `profiles.experiments.corpora` in `configs/train/gemma4-12_cpt_matrix.yaml`. For evaluation, add the same corpus name to `profiles.experiments.training_corpora` in `configs/evaluation/gemma4-12_eval_matrix.yaml` so trained variants can be evaluated. Then regenerate train and evaluation configs:

```bash
python -m src.generate_train_configs --matrix configs/train/gemma4-12_cpt_matrix.yaml --overwrite
python -m src.generate_eval_configs --matrix configs/evaluation/gemma4-12_eval_matrix.yaml --overwrite
```
