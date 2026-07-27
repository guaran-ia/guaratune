# Add a Dataset

New datasets can be added to the dataset configuration `configs/data/gemma4_cpt.yaml`. The file has three related sections:

- `sources`: Hugging Face datasets or Parquet-backed dataset repositories to read from.
- `components`: reusable document selections built from one source or from another component.
- `corpora`: final CPT dataset configurations written under `data/processed/` and `data/evaluation/perplexity/`.

First, add a source under `sources`. Use `loader: datasets` for normal Hugging Face datasets that work with `datasets.load_dataset(..., streaming=True)`. Use `loader: hf_parquet` when the source should be read directly from Hugging Face Parquet files under a repository path such as a language/config directory.

```yaml
sources:
  new_dataset:
    name: new_dataset
    dataset: organization/dataset-name
    config: null
    split: train
    text_column: text
    id_column: id
    loader: datasets
    revision: null
```

Set `revision` to a branch, tag, or commit if a specific source version should be requested. When `revision: null`, preparation resolves the current default revision and records the immutable commit in `data/source_revisions.lock.json`.

Then add a component under `components`. The common options are:

- `kind: all`: include all documents from a source.
- `kind: sample`: make a deterministic document-level sample from a source.
- `kind: fixed_subsample`: make a deterministic prefix-sized subsample from an existing component.

For a sampled component, `target_ratio_of_kuatia` is relative to the Kuatia token count. For example, `0.25` means the component targets 25% as many tokens as Kuatia. Because the final corpus target is 80% Kuatia and 20% added data, this is the ratio used by the existing `C2` and `C3` components.

```yaml
components:
  new_dataset_20:
    kind: sample
    source: new_dataset
    target_ratio_of_kuatia: 0.25
```

Finally, include the component in a final corpus under `corpora`. If the new corpus should keep Kuatia at 80% of the total, set `kuatia_share_target: 0.8` and add Kuatia plus the new component.

```yaml
corpora:
  C5_kuatia_new20:
    kuatia_share_target: 0.8
    components:
      - kuatia
      - new_dataset_20
```

After editing the config, regenerate the data artifacts:

```bash
python -m src.prepare_data --config configs/data/gemma4_cpt.yaml --overwrite
```

This updates the processed corpora, held-out perplexity splits, selection ledgers, manifests, source revision lock, and `data/dataset_info.json`. The new corpus name must also be added to any training and evaluation matrices that should use it, for example the `profiles.*.corpora` lists in `configs/train/gemma4-12_cpt_matrix.yaml`, `configs/train/gemma4-26_cpt_matrix.yaml`, `configs/evaluation/gemma4-12_eval_matrix.yaml`, and `configs/evaluation/gemma4-26_eval_matrix.yaml`. After updating those matrices, regenerate the train and evaluation configs with `src.generate_train_configs` and `src.generate_eval_configs`.