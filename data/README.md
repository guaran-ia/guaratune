# Add a Dataset

New source datasets are registered in `configs/data/gemma4_cpt.yaml`.

The relevant sections are:

- `sources`: Hugging Face datasets or Hugging Face Parquet-backed repositories.
- `corpora`: final CPT dataset configurations, including Kuatia synthetic-data 
handling and optional augmentation ratios.

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

Use `loader: datasets` for sources that work through `datasets.load_dataset(...)`.
Use `loader: hf_parquet` when the source should be read directly from Hugging Face 
Parquet shards under a repository path or language/config directory. `commit_id` 
pins the exact source revision used for reproducibility.

Then add a corpus entry to `configs/data/gemma4_cpt.yaml`:

```yaml
corpora:
  C9_kuatia_new20:
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

The new corpus name must also be added to any training matrix profile that should 
train on it, for example `profiles.experiments.corpora` in `configs/train/gemma4-12_cpt_matrix.yaml`. 
For evaluation, add the same corpus name to `profiles.experiments.training_corpora` in `configs/evaluation/gemma4-12_eval_matrix.yaml` so trained variants can be evaluated. Then regenerate train and evaluation configs:

```bash
python -m src.generate_train_configs --matrix configs/train/gemma4-12_cpt_matrix.yaml --overwrite
python -m src.generate_eval_configs --matrix configs/evaluation/gemma4-12_eval_matrix.yaml --overwrite
```
