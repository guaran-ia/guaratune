#!/usr/bin/env python3
"""Prepare frozen CPT corpora for LLaMA Factory.

The script writes JSONL training corpora, document-level selection files,
source revision locks, per-corpus manifests, and a project-local LLaMA
Factory dataset_info.json.
"""

from __future__ import annotations

import click
import gzip
import hashlib
import inspect
import json
import os
import pyarrow.parquet as pq
import random
import yaml

from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, replace
from datasets import get_dataset_config_names, load_dataset, load_dataset_builder
from datetime import UTC, datetime
from huggingface_hub import HfApi
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from typing import Any


SELECTION_SUFFIX = '.selection.jsonl.gz'


@dataclass(frozen=True)
class SourceSpec:
    name: str
    dataset: str
    config: str | None
    split: str
    text_column: str
    id_column: str | None
    loader: str
    revision: str | None


@dataclass
class ComponentStats:
    name: str
    source_name: str
    dataset: str
    config: str | None
    split: str
    loader: str
    revision: str | None
    selection: str
    path: str
    selection_path: str
    target_tokens: int | None
    actual_tokens: int
    documents: int
    seed: int | None


def load_config(config_path: str) -> dict[str, Any]:
    """Load a YAML dataset preparation config.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Parsed config dictionary.

    Raises:
        ValueError: If the config file is empty or not a mapping.
    """
    with open(config_path, 'r', encoding='utf-8') as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f'Config must be a mapping: {config_path}')

    return config


def resolve_config(
    config: dict[str, Any],
    *,
    output_dir: str | None,
    target_scale: float,
    max_kuatia_docs: int | None,
    allow_incomplete_samples: bool,
) -> dict[str, Any]:
    """Apply CLI execution overrides to a loaded config.

    Args:
        config: Loaded YAML config.
        output_dir: Optional output directory override.
        target_scale: Debug multiplier for token targets.
        max_kuatia_docs: Optional debug cap for Kuatia documents.
        allow_incomplete_samples: Whether CLI enables incomplete samples.

    Returns:
        Resolved config dictionary used by the run.
    """
    resolved = json.loads(json.dumps(config))
    resolved['output_dir'] = output_dir or resolved.get('output_dir', 'data')
    resolved['target_scale'] = target_scale
    resolved['max_kuatia_docs'] = max_kuatia_docs
    resolved['allow_incomplete_samples'] = allow_incomplete_samples or bool(
        resolved.get('allow_incomplete_samples', False)
    )
    return resolved


def require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    """Return a required mapping from a config dictionary.

    Args:
        config: Config dictionary to inspect.
        key: Required key whose value must be a mapping.

    Returns:
        Mapping value for the requested key.

    Raises:
        ValueError: If the key is missing or is not a mapping.
    """
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f'Config key must be a mapping: {key}')

    return value


def source_specs_from_config(config: dict[str, Any]) -> dict[str, SourceSpec]:
    """Build source descriptors from config source entries.

    Args:
        config: Resolved config containing a sources mapping.

    Returns:
        Mapping of source keys to SourceSpec instances.
    """
    specs = {}
    for key, source in require_mapping(config, 'sources').items():
        specs[key] = SourceSpec(
            name=source.get('name', key),
            dataset=source['dataset'],
            config=optional_config(source.get('config')),
            split=source.get('split', 'train'),
            text_column=source.get('text_column', 'text'),
            id_column=optional_config(source.get('id_column', 'id')),
            loader=source.get('loader', 'datasets'),
            revision=optional_config(source.get('revision')),
        )

    return specs


def resolve_source_revisions(sources: dict[str, SourceSpec]) -> dict[str, Any]:
    """Resolve source revisions to immutable Hugging Face commit SHAs.

    Args:
        sources: Source descriptors from the resolved config.

    Returns:
        Source revision lock entries keyed by source name.

    Raises:
        RuntimeError: If a source revision cannot be resolved.
    """
    api = HfApi()
    locked_sources = {}
    for spec in sources.values():
        try:
            info = api.dataset_info(spec.dataset, revision=spec.revision)
            resolved_revision = getattr(info, 'sha', None)
        except Exception as exc:
            raise RuntimeError(
                f'Cannot resolve revision for {spec.dataset} '
                f'({spec.revision or "default"}): {type(exc).__name__}: {exc}'
            ) from exc

        if not resolved_revision:
            raise RuntimeError(f'Cannot resolve immutable revision for {spec.dataset}.')

        locked_sources[spec.name] = {
            'dataset': spec.dataset,
            'requested_revision': spec.revision,
            'resolved_revision': resolved_revision,
        }

    return locked_sources


def pin_sources_to_revisions(
    sources: dict[str, SourceSpec], locked_sources: dict[str, Any]
) -> dict[str, SourceSpec]:
    """Return source descriptors pinned to locked revisions.

    Args:
        sources: Source descriptors from the resolved config.
        locked_sources: Lock entries keyed by source name.

    Returns:
        Source descriptors with revision set to the locked commit SHA.

    Raises:
        RuntimeError: If the lock does not match the configured sources.
    """
    pinned = {}
    for key, spec in sources.items():
        locked = locked_sources.get(spec.name)
        if not isinstance(locked, dict):
            raise RuntimeError(f'Missing source revision lock entry: {spec.name}')
        if locked.get('dataset') != spec.dataset:
            raise RuntimeError(
                f'Source lock mismatch for {spec.name}: expected dataset '
                f'{spec.dataset}, found {locked.get("dataset")}.'
            )
        pinned[key] = replace(spec, revision=locked['resolved_revision'])

    return pinned


def write_source_revisions_lock(path: str, locked_sources: dict[str, Any]) -> None:
    """Write the source revision lock file.

    Args:
        path: Destination JSON lock file path.
        locked_sources: Resolved source revisions keyed by source name.

    Returns:
        None.
    """
    lock = {
        'created_at': datetime.now(UTC).isoformat(),
        'sources': locked_sources,
    }
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write(json.dumps(lock, ensure_ascii=False, indent=2) + '\n')


def open_text(path: str, mode: str) -> Any:
    """Open plain or gzip-compressed text files.

    Args:
        path: File path to open.
        mode: Text-mode-compatible open mode.

    Returns:
        Open file handle.
    """
    if path.endswith('.gz'):
        text_mode = mode if 't' in mode else f'{mode}t'
        return gzip.open(path, text_mode, encoding='utf-8')

    return open(path, mode, encoding='utf-8')


def load_source_revisions_lock(path: str) -> dict[str, Any]:
    """Load source revision locks from disk.

    Args:
        path: JSON lock file path.

    Returns:
        Source revision lock entries keyed by source name.

    Raises:
        FileNotFoundError: If the lock file does not exist.
        ValueError: If the lock file has an invalid shape.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'Cannot reconstruct without source revision lock: {path}'
        )

    with open(path, 'r', encoding='utf-8') as handle:
        lock = json.load(handle)

    locked_sources = lock.get('sources') if isinstance(lock, dict) else None
    if not isinstance(locked_sources, dict):
        raise ValueError(f'Invalid source revision lock: {path}')

    return locked_sources


def validate_locked_source_revisions(locked_sources: dict[str, Any]) -> None:
    """Verify that locked Hugging Face dataset revisions are still available.

    Args:
        locked_sources: Source revision lock entries keyed by source name.

    Returns:
        None.

    Raises:
        RuntimeError: If a locked revision cannot be accessed exactly.
    """
    api = HfApi()
    for source_name, locked in locked_sources.items():
        dataset = locked.get('dataset') if isinstance(locked, dict) else None
        expected_revision = (
            locked.get('resolved_revision') if isinstance(locked, dict) else None
        )
        if not dataset or not expected_revision:
            raise RuntimeError(f'Invalid source revision lock entry: {source_name}')

        try:
            info = api.dataset_info(dataset, revision=expected_revision)
            actual_revision = getattr(info, 'sha', None)
        except Exception as exc:
            raise RuntimeError(
                f'Cannot reconstruct because the locked dataset revision is not available: '
                f'{dataset}@{expected_revision}. Original source access may have changed. '
                f'No corpus was written. {type(exc).__name__}: {exc}'
            ) from exc

        if actual_revision != expected_revision:
            raise RuntimeError(
                f'Cannot reconstruct because {dataset}@{expected_revision} resolved to '
                f'{actual_revision}. No corpus was written.'
            )


def output_paths(output_dir: str) -> dict[str, str]:
    """Build standard output paths for dataset artifacts.

    Args:
        output_dir: Root directory for generated dataset artifacts.

    Returns:
        Mapping of logical artifact locations to paths.
    """
    return {
        'processed': os.path.join(output_dir, 'processed'),
        'components': os.path.join(output_dir, 'processed', 'components'),
        'perplexity': os.path.join(output_dir, 'evaluation', 'perplexity'),
        'selections': os.path.join(output_dir, 'selections'),
        'manifests': os.path.join(output_dir, 'manifests'),
        'source_lock': os.path.join(output_dir, 'source_revisions.lock.json'),
    }


def selection_file_path(paths: dict[str, str], component_name: str) -> str:
    """Build the default selection ledger path for a component.

    Args:
        paths: Standard artifact paths for the active output directory.
        component_name: Component name whose selection path is needed.

    Returns:
        Gzip-compressed selection ledger path.
    """
    return os.path.join(paths['selections'], f'{component_name}{SELECTION_SUFFIX}')


def heldout_config_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return normalized held-out split settings.

    Args:
        config: Resolved dataset preparation config.

    Returns:
        Held-out settings containing enabled, ratio, and seed values.

    Raises:
        ValueError: If the held-out ratio is outside the supported range.
    """
    raw_config = config.get('heldout') or {}
    if not isinstance(raw_config, dict):
        raise ValueError('Config key must be a mapping: heldout')

    ratio = float(raw_config.get('ratio', 0.0))
    if ratio < 0.0 or ratio >= 1.0:
        raise ValueError('heldout.ratio must be greater than or equal to 0 and less than 1.')

    return {
        'enabled': bool(raw_config.get('enabled', ratio > 0.0)),
        'ratio': ratio,
        'seed': int(raw_config.get('seed', config['seed'])),
    }


def ensure_writable_output(
    output_dir: str,
    overwrite: bool,
    corpus_names: Iterable[str],
    component_names: Iterable[str],
    *,
    auxiliary: bool,
    heldout_enabled: bool,
) -> dict[str, str]:
    """Create output directories and guard existing generated files.

    Args:
        output_dir: Root directory where generated dataset artifacts are written.
        overwrite: Whether existing generated output files may be replaced.
        corpus_names: Final corpus names expected to be generated.
        component_names: Component names expected to be generated.
        auxiliary: Whether auxiliary files are expected to be written.
        heldout_enabled: Whether held-out perplexity files are generated.

    Returns:
        Mapping of logical directory names to absolute or relative path strings.

    Raises:
        FileExistsError: If generated files already exist and overwrite is false.
    """
    paths = output_paths(output_dir)
    for path in paths.values():
        if os.path.splitext(path)[1]:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        else:
            os.makedirs(path, exist_ok=True)

    generated = [os.path.join(paths['processed'], f'{name}.jsonl') for name in corpus_names]
    if heldout_enabled:
        generated.extend(
            os.path.join(paths['perplexity'], f'{name}.jsonl') for name in corpus_names
        )
    generated.extend(
        os.path.join(paths['components'], f'{name}.jsonl') for name in component_names
    )
    if auxiliary:
        generated.append(os.path.join(output_dir, 'dataset_info.json'))
        generated.extend(
            selection_file_path(paths, name) for name in component_names
        )
        generated.extend(
            os.path.join(paths['selections'], f'{name}.selection.jsonl')
            for name in component_names
        )
        generated.extend(
            os.path.join(paths['manifests'], f'{name}.manifest.json')
            for name in corpus_names
        )
        generated.append(paths['source_lock'])

    existing = [path for path in generated if os.path.exists(path)]
    if existing and not overwrite:
        formatted = '\n'.join(f'  {path}' for path in existing)
        raise FileExistsError(
            f'Refusing to overwrite existing generated files:\n{formatted}'
        )

    return paths


def load_stream(
    spec: SourceSpec, *, shuffle: bool, seed: int, buffer_size: int
) -> Iterable[dict[str, Any]]:
    """Load a source dataset as an iterable stream.

    Args:
        spec: Source dataset descriptor.
        shuffle: Whether to shuffle records before yielding them.
        seed: Random seed used by deterministic shuffling.
        buffer_size: Shuffle buffer size for streaming datasets that support it.

    Returns:
        Iterable of source rows as dictionaries.
    """
    if spec.loader == 'hf_parquet':
        return load_hf_parquet_stream(spec, shuffle=shuffle, seed=seed)

    kwargs: dict[str, Any] = {'split': spec.split, 'streaming': True}
    if spec.config:
        kwargs['name'] = spec.config
    if spec.revision:
        kwargs['revision'] = spec.revision

    dataset = load_dataset(spec.dataset, **kwargs)
    if shuffle:
        shuffle_kwargs: dict[str, Any] = {'seed': seed}
        if 'buffer_size' in inspect.signature(dataset.shuffle).parameters:
            shuffle_kwargs['buffer_size'] = buffer_size

        dataset = dataset.shuffle(**shuffle_kwargs)
    return dataset  # type: ignore


def load_hf_parquet_stream(
    spec: SourceSpec, *, shuffle: bool, seed: int
) -> Iterator[dict[str, Any]]:
    """Stream selected columns from Hugging Face Parquet shards.

    Args:
        spec: Source descriptor whose config names the repo subdirectory.
        shuffle: Whether to shuffle shard, row-group, and row order.
        seed: Random seed used when shuffling.

    Returns:
        Iterator over row dictionaries containing at least text and optional id.

    Raises:
        ValueError: If the source has no config/path prefix.
        FileNotFoundError: If no Parquet shards are found for the source.
    """
    if spec.config is None:
        raise ValueError(
            f'{spec.name} uses hf_parquet loader but has no config/path prefix.'
        )

    api = HfApi()
    files = [
        item.path
        for item in api.list_repo_tree(
            spec.dataset,
            path_in_repo=spec.config,
            repo_type='dataset',
            recursive=True,
            revision=spec.revision,
        )
        if item.path.endswith('.parquet')
    ]
    if not files:
        raise FileNotFoundError(
            f'No parquet files found under {spec.dataset}/{spec.config}.'
        )

    rng = random.Random(seed)
    if shuffle:
        rng.shuffle(files)

    row_index = 0
    columns = [spec.text_column]
    if spec.id_column:
        columns.append(spec.id_column)

    for file_path in files:
        if spec.revision:
            parquet_path = f'hf://datasets/{spec.dataset}@{spec.revision}/{file_path}'
        else:
            parquet_path = f'hf://datasets/{spec.dataset}/{file_path}'
        parquet_file = pq.ParquetFile(parquet_path)
        row_groups = list(range(parquet_file.num_row_groups))
        if shuffle:
            rng.shuffle(row_groups)

        for row_group in row_groups:
            table = parquet_file.read_row_group(row_group, columns=columns)
            rows = table.to_pylist()
            if shuffle:
                rng.shuffle(rows)

            for row in rows:
                row['_row_index'] = row_index
                row_index += 1
                yield row


def clean_text(value: Any) -> str:
    """Normalize a raw text value into stripped text.

    Args:
        value: Raw text-like value from a dataset row.

    Returns:
        Stripped string, or an empty string for missing values.
    """
    return str(value).strip() if value is not None else ''


def optional_config(value: str | None) -> str | None:
    """Convert empty CLI config values into None.

    Args:
        value: Config value supplied by the CLI.

    Returns:
        Normalized config string, or None for empty/null sentinel values.
    """
    if value is None:
        return None
    stripped = value.strip()
    if stripped.lower() in {'', 'none', 'null'}:
        return None
    return stripped


def token_count(tokenizer: PreTrainedTokenizerBase, text: str) -> int:
    """Count tokenizer tokens without adding special tokens.

    Args:
        tokenizer: Tokenizer used for CPT token accounting.
        text: Text to tokenize.

    Returns:
        Number of tokenizer tokens in the text.
    """
    return len(tokenizer.encode(text, add_special_tokens=False))


def row_id(row: dict[str, Any], source: str, id_column: str | None, index: int) -> str:
    """Build a stable source-prefixed document identifier.

    Args:
        row: Source dataset row.
        source: Short source name to prefix the identifier.
        id_column: Optional source column containing stable row ids.
        index: Fallback row index when no source id is available.

    Returns:
        Source-prefixed identifier for the document.
    """
    if id_column and row.get(id_column) not in (None, ''):
        return f'{source}:{row[id_column]}'
    return f'{source}:row-{index}'


def write_jsonl(path: str, rows: Iterable[dict[str, Any]]) -> None:
    """Write dictionaries as UTF-8 JSONL records.

    Args:
        path: Destination JSONL path.
        rows: Dictionaries to serialize one per line.

    Returns:
        None.
    """
    with open_text(path, 'w') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def stream_jsonl(path: str) -> Iterator[dict[str, Any]]:
    """Yield non-empty JSONL records from a local file.

    Args:
        path: Source JSONL path.

    Returns:
        Iterator over decoded JSON records.
    """
    with open_text(path, 'r') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_kuatia_component(
    component_name: str,
    spec: SourceSpec,
    tokenizer: PreTrainedTokenizerBase,
    output_path: str,
    selection_path: str,
    *,
    max_docs: int | None,
) -> ComponentStats:
    """Write the Kuatia component and its document-selection ledger.

    Args:
        component_name: Component name used in outputs and manifests.
        spec: Kuatia source descriptor.
        tokenizer: Tokenizer used to count document tokens.
        output_path: Destination component JSONL path.
        selection_path: Destination selection ledger JSONL path.
        max_docs: Optional development cap on the number of documents written.

    Returns:
        Component statistics for the written Kuatia corpus.
    """
    docs = 0
    tokens = 0

    with (
        open(output_path, 'w', encoding='utf-8') as out,
        open_text(selection_path, 'w') as sel,
    ):
        for index, row in enumerate(
            load_stream(spec, shuffle=False, seed=0, buffer_size=1)
        ):
            if max_docs is not None and docs >= max_docs:
                break

            text = clean_text(row.get(spec.text_column))
            if not text:
                continue

            doc_tokens = token_count(tokenizer, text)
            doc_id = row_id(row, spec.name, spec.id_column, index)
            record = {'id': doc_id, 'source': spec.name, 'text': text}
            selection = {'id': doc_id, 'source': spec.name, 'tokens': doc_tokens}
            if row.get('url'):
                selection['url'] = row['url']
            if row.get('corpus'):
                selection['corpus'] = row['corpus']

            out.write(json.dumps(record, ensure_ascii=False) + '\n')
            sel.write(json.dumps(selection, ensure_ascii=False) + '\n')
            docs += 1
            tokens += doc_tokens

            if docs % 10000 == 0:
                print(f'[kuatia] wrote {docs:,} docs / {tokens:,} tokens', flush=True)

    return ComponentStats(
        name=component_name,
        source_name=spec.name,
        dataset=spec.dataset,
        config=spec.config,
        split=spec.split,
        loader=spec.loader,
        revision=spec.revision,
        selection='all' if max_docs is None else f'first_{max_docs}_documents',
        path=str(output_path),
        selection_path=str(selection_path),
        target_tokens=None,
        actual_tokens=tokens,
        documents=docs,
        seed=None,
    )


def write_sample_component(
    spec: SourceSpec,
    tokenizer: PreTrainedTokenizerBase,
    output_path: str,
    selection_path: str,
    *,
    target_tokens: int,
    seed: int,
    shuffle_buffer_size: int,
    allow_incomplete: bool,
) -> ComponentStats:
    """Sample full documents until a token target is reached.

    Args:
        spec: FineWeb source descriptor.
        tokenizer: Tokenizer used to count document tokens.
        output_path: Destination component JSONL path.
        selection_path: Destination selection ledger JSONL path.
        target_tokens: Minimum token count to sample.
        seed: Random seed for deterministic sampling.
        shuffle_buffer_size: Shuffle buffer size for compatible streaming sources.
        allow_incomplete: Whether to accept samples smaller than target_tokens.

    Returns:
        Component statistics for the sampled corpus.

    Raises:
        RuntimeError: If the source is exhausted before target_tokens and
            allow_incomplete is false.
    """
    docs = 0
    tokens = 0

    with (
        open(output_path, 'w', encoding='utf-8') as out,
        open_text(selection_path, 'w') as sel,
    ):
        for index, row in enumerate(
            load_stream(spec, shuffle=True, seed=seed, buffer_size=shuffle_buffer_size)
        ):
            text = clean_text(row.get(spec.text_column))
            if not text:
                continue

            doc_tokens = token_count(tokenizer, text)
            doc_id = row_id(row, spec.name, spec.id_column, index)
            record = {'id': doc_id, 'source': spec.name, 'text': text}
            selection = {'id': doc_id, 'source': spec.name, 'tokens': doc_tokens}
            raw_metadata = row.get('metadata')
            metadata: dict[str, Any] = (
                raw_metadata if isinstance(raw_metadata, dict) else {}
            )
            url = row.get('url') or metadata.get('url')
            if url:
                selection['url'] = url

            out.write(json.dumps(record, ensure_ascii=False) + '\n')
            sel.write(json.dumps(selection, ensure_ascii=False) + '\n')
            docs += 1
            tokens += doc_tokens

            if docs % 10000 == 0:
                print(
                    f'[{os.path.splitext(os.path.basename(output_path))[0]}] sampled '
                    f'{docs:,} docs / {tokens:,} of {target_tokens:,} target tokens',
                    flush=True,
                )
            if tokens >= target_tokens:
                break

    if tokens < target_tokens and not allow_incomplete:
        raise RuntimeError(
            f'{spec.name} exhausted at {tokens:,} tokens before reaching target {target_tokens:,}. '
            'Use a larger source or rerun with --allow-incomplete-samples for development only.'
        )

    return ComponentStats(
        name=os.path.splitext(os.path.basename(output_path))[0],
        source_name=spec.name,
        dataset=spec.dataset,
        config=spec.config,
        split=spec.split,
        loader=spec.loader,
        revision=spec.revision,
        selection='sampled_document_level',
        path=str(output_path),
        selection_path=str(selection_path),
        target_tokens=target_tokens,
        actual_tokens=tokens,
        documents=docs,
        seed=seed,
    )


def write_fixed_subsample(
    source_stats: ComponentStats,
    source_path: str,
    source_selection_path: str,
    output_path: str,
    selection_path: str,
    *,
    target_tokens: int,
) -> ComponentStats:
    """Write a deterministic fixed subsample from an existing component.

    Args:
        source_stats: Statistics for the source component being subsampled.
        source_path: JSONL path for the source component.
        source_selection_path: Selection ledger path for the source component.
        output_path: Destination subsample JSONL path.
        selection_path: Destination subsample ledger path.
        target_tokens: Minimum token count to include in the subsample.

    Returns:
        Component statistics for the fixed subsample.
    """
    docs = 0
    tokens = 0
    source_records = stream_jsonl(source_path)
    source_selections = stream_jsonl(source_selection_path)

    with (
        open(output_path, 'w', encoding='utf-8') as out,
        open_text(selection_path, 'w') as sel,
    ):
        for record, selection in zip(source_records, source_selections, strict=True):
            out.write(json.dumps(record, ensure_ascii=False) + '\n')
            sel.write(json.dumps(selection, ensure_ascii=False) + '\n')
            docs += 1
            tokens += int(selection['tokens'])
            if tokens >= target_tokens:
                break

    return ComponentStats(
        name=os.path.splitext(os.path.basename(output_path))[0],
        source_name=source_stats.source_name,
        dataset=source_stats.dataset,
        config=source_stats.config,
        split=source_stats.split,
        loader=source_stats.loader,
        revision=source_stats.revision,
        selection=f'fixed_subsample_of_{source_stats.name}',
        path=str(output_path),
        selection_path=str(selection_path),
        target_tokens=target_tokens,
        actual_tokens=tokens,
        documents=docs,
        seed=source_stats.seed,
    )


def is_heldout_document(
    doc_id: str, *, config_name: str, ratio: float, seed: int
) -> bool:
    """Return whether a document belongs to the deterministic held-out split.

    Args:
        doc_id: Stable document id from the component JSONL.
        config_name: Final corpus configuration name used as split salt.
        ratio: Held-out split ratio.
        seed: Held-out split seed.

    Returns:
        True when the document should be excluded from training and written to held-out.
    """
    if ratio <= 0.0:
        return False

    digest = hashlib.sha256(f'{seed}:{config_name}:{doc_id}'.encode('utf-8')).digest()
    bucket = int.from_bytes(digest[:8], byteorder='big') / float(1 << 64)
    return bucket < ratio


def empty_corpus_summary(config_name: str, output_path: str) -> dict[str, Any]:
    """Build an empty corpus summary for disabled output splits.

    Args:
        config_name: Final corpus configuration name.
        output_path: Destination JSONL path represented by the summary.

    Returns:
        Summary dictionary with zero document and token counts.
    """
    return {
        'name': config_name,
        'path': str(output_path),
        'documents': 0,
        'tokens': 0,
        'components': [],
    }


def write_corpus_splits(
    config_name: str,
    components: list[ComponentStats],
    train_output_path: str,
    heldout_output_path: str,
    heldout_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Write train and held-out JSONL files for a final corpus config.

    Args:
        config_name: Final corpus configuration name.
        components: Component statistics whose records are split and concatenated.
        train_output_path: Destination JSONL path for the train split.
        heldout_output_path: Destination JSONL path for the held-out split.
        heldout_config: Normalized held-out split settings.

    Returns:
        Pair of summary dictionaries for the train and held-out splits.
    """
    train_docs = 0
    train_tokens = 0
    heldout_docs = 0
    heldout_tokens = 0
    heldout_enabled = bool(heldout_config['enabled'])
    heldout_ratio = float(heldout_config['ratio'])
    heldout_seed = int(heldout_config['seed'])

    os.makedirs(os.path.dirname(train_output_path), exist_ok=True)
    if heldout_enabled:
        os.makedirs(os.path.dirname(heldout_output_path), exist_ok=True)

    heldout_out = None
    with open(train_output_path, 'w', encoding='utf-8') as train_out:
        if heldout_enabled:
            heldout_out = open(heldout_output_path, 'w', encoding='utf-8')
        for component in components:
            component_records = stream_jsonl(component.path)
            component_selections = stream_jsonl(component.selection_path)
            for record, selection in zip(
                component_records, component_selections, strict=True
            ):
                doc_tokens = int(selection['tokens'])
                if record['id'] != selection['id']:
                    raise RuntimeError(
                        f'Component {component.name} has mismatched record and selection '
                        f'ids: {record["id"]} != {selection["id"]}.'
                    )

                line = json.dumps(record, ensure_ascii=False) + '\n'
                if heldout_enabled and is_heldout_document(
                    record['id'],
                    config_name=config_name,
                    ratio=heldout_ratio,
                    seed=heldout_seed,
                ):
                    if heldout_out is None:
                        raise RuntimeError('Held-out output is not open.')
                    heldout_out.write(line)
                    heldout_docs += 1
                    heldout_tokens += doc_tokens
                else:
                    train_out.write(line)
                    train_docs += 1
                    train_tokens += doc_tokens
        if heldout_out is not None:
            heldout_out.close()

    train_summary = {
        'name': config_name,
        'path': str(train_output_path),
        'documents': train_docs,
        'tokens': train_tokens,
        'components': [component.name for component in components],
    }
    heldout_summary = {
        'name': config_name,
        'path': str(heldout_output_path),
        'documents': heldout_docs,
        'tokens': heldout_tokens,
        'components': [component.name for component in components],
    }
    if not heldout_enabled:
        heldout_summary = empty_corpus_summary(config_name, heldout_output_path)

    return train_summary, heldout_summary


def write_manifest(
    path: str,
    *,
    config_name: str,
    config_path: str,
    tokenizer: str,
    seed: int,
    corpus: dict[str, Any],
    heldout_corpus: dict[str, Any],
    components: list[ComponentStats],
    extra: dict[str, Any],
) -> None:
    """Write reproducibility metadata for one final corpus config.

    Args:
        path: Destination manifest JSON path.
        config_name: Final corpus configuration name.
        config_path: Config file path used for the run.
        tokenizer: Tokenizer name or path used for token accounting.
        seed: Sampling seed used for fixed FineWeb samples.
        corpus: Final train corpus summary from write_corpus_splits.
        heldout_corpus: Held-out corpus summary from write_corpus_splits.
        components: Component statistics included in the final corpus.
        extra: Additional metadata fields to include in the manifest.

    Returns:
        None.
    """
    manifest = {
        'config_name': config_name,
        'created_at': datetime.now(UTC).isoformat(),
        'config_path': config_path,
        'tokenizer': tokenizer,
        'sample_seed': seed,
        'corpus': corpus,
        'heldout_corpus': heldout_corpus,
        'components': [asdict(component) for component in components],
        **extra,
    }
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')


def write_dataset_info(path: str, corpus_names: Iterable[str]) -> None:
    """Write the project-local LLaMA Factory dataset registry.

    Args:
        path: Destination dataset_info.json path.
        corpus_names: Final corpus names to register.

    Returns:
        None.
    """
    info = {
        config_name: {
            'file_name': f'processed/{config_name}.jsonl',
            'columns': {'prompt': 'text'},
        }
        for config_name in corpus_names
    }
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write(json.dumps(info, ensure_ascii=False, indent=2) + '\n')


def preflight(specs: list[SourceSpec]) -> None:
    """Print source dataset metadata without writing corpora.

    Args:
        specs: Source descriptors to inspect.

    Returns:
        None.
    """
    for spec in specs:
        print(f'\n{spec.name}: {spec.dataset}')
        print(f'  loader: {spec.loader}')
        print(f'  revision: {spec.revision or "default"}')
        try:
            config_kwargs = {'revision': spec.revision} if spec.revision else {}
            configs = get_dataset_config_names(spec.dataset, **config_kwargs)  # type: ignore
            if len(configs) > 1:
                print(f'  configs: {configs[:10]}{" ..." if len(configs) > 10 else ""}')
        except Exception as exc:
            print(f'  config lookup failed: {type(exc).__name__}: {exc}')

        if spec.loader == 'hf_parquet':
            try:
                api = HfApi()
                files = [
                    item
                    for item in api.list_repo_tree(
                        spec.dataset,
                        path_in_repo=spec.config or '',
                        repo_type='dataset',
                        recursive=True,
                        revision=spec.revision,
                    )
                    if item.path.endswith('.parquet')
                ]
                print(f'  parquet files: {len(files)}')
                if files:
                    if spec.revision:
                        first_path = f'hf://datasets/{spec.dataset}@{spec.revision}/{files[0].path}'
                    else:
                        first_path = f'hf://datasets/{spec.dataset}/{files[0].path}'
                    print(f'  first file: {files[0].path}')
                    print(f'  first file size: {getattr(files[0], "size", None)}')
                    print(f'  schema: {pq.read_schema(first_path)}')
            except Exception as exc:
                print(f'  parquet metadata failed: {type(exc).__name__}: {exc}')
        else:
            try:
                builder = (
                    load_dataset_builder(
                        spec.dataset, spec.config, revision=spec.revision
                    )
                    if spec.config
                    else load_dataset_builder(spec.dataset, revision=spec.revision)
                )
                print(f'  config: {builder.config.name}')
                print(f'  splits: {builder.info.splits}')
                print(f'  features: {builder.info.features}')
            except Exception as exc:
                print(f'  metadata failed: {type(exc).__name__}: {exc}')


def component_stats_from_dict(data: dict[str, Any]) -> ComponentStats:
    """Build component statistics from manifest JSON data.

    Args:
        data: Component metadata loaded from a manifest.

    Returns:
        ComponentStats instance.
    """
    return ComponentStats(
        name=data['name'],
        source_name=data['source_name'],
        dataset=data['dataset'],
        config=optional_config(data.get('config')),
        split=data['split'],
        loader=data['loader'],
        revision=optional_config(data.get('revision')),
        selection=data['selection'],
        path=data['path'],
        selection_path=data['selection_path'],
        target_tokens=data.get('target_tokens'),
        actual_tokens=int(data['actual_tokens']),
        documents=int(data['documents']),
        seed=data.get('seed'),
    )


def component_stats_with_paths(stats: ComponentStats, paths: dict[str, str]) -> ComponentStats:
    """Return component statistics pointed at the active output directory.

    Args:
        stats: Component statistics loaded from a manifest.
        paths: Standard artifact paths for the active output directory.

    Returns:
        ComponentStats instance with active output and selection paths.
    """
    selection_name = os.path.basename(stats.selection_path)
    if not selection_name:
        selection_name = f'{stats.name}{SELECTION_SUFFIX}'

    return replace(
        stats,
        path=os.path.join(paths['components'], f'{stats.name}.jsonl'),
        selection_path=os.path.join(paths['selections'], selection_name),
    )


def load_manifests(manifest_dir: str, corpus_names: Iterable[str]) -> dict[str, Any]:
    """Load expected per-corpus manifests.

    Args:
        manifest_dir: Directory containing manifest JSON files.
        corpus_names: Corpus names whose manifests are required.

    Returns:
        Mapping of corpus names to manifest dictionaries.

    Raises:
        FileNotFoundError: If a required manifest is missing.
        ValueError: If a manifest has an unexpected corpus name.
    """
    manifests = {}
    for corpus_name in corpus_names:
        manifest_path = os.path.join(manifest_dir, f'{corpus_name}.manifest.json')
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f'Cannot reconstruct without manifest: {manifest_path}'
            )
        with open(manifest_path, 'r', encoding='utf-8') as handle:
            manifest = json.load(handle)
        if manifest.get('config_name') != corpus_name:
            raise ValueError(
                f'Manifest {manifest_path} has config_name={manifest.get("config_name")}, '
                f'expected {corpus_name}.'
            )
        manifests[corpus_name] = manifest

    return manifests


def collect_component_stats(manifests: dict[str, Any]) -> dict[str, ComponentStats]:
    """Collect unique component statistics from corpus manifests.

    Args:
        manifests: Per-corpus manifest dictionaries.

    Returns:
        Component statistics keyed by component name.

    Raises:
        ValueError: If the same component has conflicting manifest statistics.
    """
    components = {}
    for manifest in manifests.values():
        for data in manifest['components']:
            stats = component_stats_from_dict(data)
            existing = components.get(stats.name)
            if existing is not None and asdict(existing) != asdict(stats):
                raise ValueError(f'Conflicting manifest entries for {stats.name}.')
            components[stats.name] = stats

    return components


def verify_component_stats(expected: ComponentStats, actual: ComponentStats) -> None:
    """Verify reconstructed component counts against manifest metadata.

    Args:
        expected: Component statistics loaded from a manifest.
        actual: Component statistics computed during reconstruction.

    Returns:
        None.

    Raises:
        RuntimeError: If document or token counts differ.
    """
    if expected.documents != actual.documents or expected.actual_tokens != actual.actual_tokens:
        raise RuntimeError(
            f'Reconstructed {expected.name} does not match the manifest: '
            f'expected {expected.documents:,} docs / {expected.actual_tokens:,} tokens, '
            f'got {actual.documents:,} docs / {actual.actual_tokens:,} tokens.'
        )


def verify_split_manifest(
    corpus: dict[str, Any], heldout_corpus: dict[str, Any], manifest: dict[str, Any]
) -> None:
    """Verify reconstructed final train and held-out corpora against a manifest.

    Args:
        corpus: Reconstructed final train corpus summary.
        heldout_corpus: Reconstructed held-out corpus summary.
        manifest: Expected manifest dictionary.

    Returns:
        None.

    Raises:
        RuntimeError: If document or token counts differ.
    """
    expected = manifest['corpus']
    expected_heldout = manifest.get('heldout_corpus')
    if isinstance(expected_heldout, dict):
        train_matches = (
            corpus['documents'] == expected['documents']
            and corpus['tokens'] == expected['tokens']
        )
        heldout_matches = (
            heldout_corpus['documents'] == expected_heldout['documents']
            and heldout_corpus['tokens'] == expected_heldout['tokens']
        )
        if train_matches and heldout_matches:
            return

        raise RuntimeError(
            f'Reconstructed {corpus["name"]} does not match the manifest: train '
            f'expected {expected["documents"]:,} docs / {expected["tokens"]:,} tokens, '
            f'got {corpus["documents"]:,} docs / {corpus["tokens"]:,} tokens; held-out '
            f'expected {expected_heldout["documents"]:,} docs / '
            f'{expected_heldout["tokens"]:,} tokens, got {heldout_corpus["documents"]:,} '
            f'docs / {heldout_corpus["tokens"]:,} tokens.'
        )

    total_documents = corpus['documents'] + heldout_corpus['documents']
    total_tokens = corpus['tokens'] + heldout_corpus['tokens']
    if total_documents != expected['documents'] or total_tokens != expected['tokens']:
        raise RuntimeError(
            f'Reconstructed {corpus["name"]} does not match the legacy manifest total: '
            f'expected {expected["documents"]:,} docs / {expected["tokens"]:,} tokens, '
            f'got {total_documents:,} docs / {total_tokens:,} tokens.'
        )


def reconstruct_stream_component(
    component_name: str,
    expected_stats: ComponentStats,
    spec: SourceSpec,
    tokenizer: PreTrainedTokenizerBase,
    output_path: str,
    selection_path: str,
    *,
    shuffle: bool,
    seed: int,
    shuffle_buffer_size: int,
) -> ComponentStats:
    """Rebuild a component by replaying its selection ledger.

    Args:
        component_name: Component name to reconstruct.
        expected_stats: Manifest statistics for the component.
        spec: Locked source dataset descriptor.
        tokenizer: Tokenizer used to verify token counts.
        output_path: Destination component JSONL path.
        selection_path: Existing selection ledger JSONL path.
        shuffle: Whether to replay source shuffling.
        seed: Shuffle seed used by sampled components.
        shuffle_buffer_size: Shuffle buffer size for compatible streaming sources.

    Returns:
        Reconstructed component statistics.

    Raises:
        FileNotFoundError: If the selection ledger is missing.
        RuntimeError: If the source stream no longer matches the selection ledger.
    """
    if not os.path.exists(selection_path):
        raise FileNotFoundError(
            f'Cannot reconstruct {component_name} without selection ledger: {selection_path}'
        )

    selections = list(stream_jsonl(selection_path))
    selection_index = 0
    docs = 0
    tokens = 0

    with open(output_path, 'w', encoding='utf-8') as out:
        for index, row in enumerate(
            load_stream(
                spec,
                shuffle=shuffle,
                seed=seed,
                buffer_size=shuffle_buffer_size,
            )
        ):
            if selection_index >= len(selections):
                break

            text = clean_text(row.get(spec.text_column))
            if not text:
                continue

            expected = selections[selection_index]
            doc_id = row_id(row, spec.name, spec.id_column, index)
            if doc_id != expected['id']:
                raise RuntimeError(
                    f'Cannot reconstruct {component_name}: source stream mismatch at '
                    f'document {selection_index + 1:,}. Expected {expected["id"]}, '
                    f'got {doc_id}. The locked source snapshot may be unavailable or '
                    f'the dataset streaming order changed. No valid corpus was produced.'
                )

            doc_tokens = token_count(tokenizer, text)
            if doc_tokens != int(expected['tokens']):
                raise RuntimeError(
                    f'Cannot reconstruct {component_name}: token mismatch for {doc_id}. '
                    f'Expected {expected["tokens"]}, got {doc_tokens}.'
                )

            out.write(
                json.dumps(
                    {'id': doc_id, 'source': spec.name, 'text': text},
                    ensure_ascii=False,
                )
                + '\n'
            )
            docs += 1
            tokens += doc_tokens
            selection_index += 1

            if docs % 10000 == 0:
                print(
                    f'[{component_name}] reconstructed {docs:,} docs / {tokens:,} tokens',
                    flush=True,
                )

    if selection_index < len(selections):
        raise RuntimeError(
            f'Cannot reconstruct {component_name}: source ended after '
            f'{selection_index:,} of {len(selections):,} selected documents.'
        )

    return replace(
        expected_stats,
        path=output_path,
        selection_path=selection_path,
        actual_tokens=tokens,
        documents=docs,
    )


def reconstruct_fixed_subsample_component(
    expected_stats: ComponentStats,
    source_stats: ComponentStats,
    output_path: str,
    selection_path: str,
) -> ComponentStats:
    """Rebuild a fixed subsample component from its reconstructed source component.

    Args:
        expected_stats: Manifest statistics for the fixed subsample.
        source_stats: Reconstructed source component statistics.
        output_path: Destination component JSONL path.
        selection_path: Existing fixed subsample selection ledger path.

    Returns:
        Reconstructed component statistics.

    Raises:
        FileNotFoundError: If the selection ledger is missing.
        RuntimeError: If the source component does not match the fixed subsample ledger.
    """
    if not os.path.exists(selection_path):
        raise FileNotFoundError(
            f'Cannot reconstruct {expected_stats.name} without selection ledger: {selection_path}'
        )

    selections = list(stream_jsonl(selection_path))
    docs = 0
    tokens = 0

    with (
        open(output_path, 'w', encoding='utf-8') as out,
        open(source_stats.path, 'r', encoding='utf-8') as source_records,
    ):
        for expected, line in zip(selections, source_records, strict=False):
            record = json.loads(line)
            if record['id'] != expected['id']:
                raise RuntimeError(
                    f'Cannot reconstruct {expected_stats.name}: expected {expected["id"]}, '
                    f'got {record["id"]} from {source_stats.name}.'
                )

            doc_tokens = int(expected['tokens'])
            out.write(json.dumps(record, ensure_ascii=False) + '\n')
            docs += 1
            tokens += doc_tokens

    if docs < len(selections):
        raise RuntimeError(
            f'Cannot reconstruct {expected_stats.name}: source component '
            f'{source_stats.name} ended after {docs:,} of {len(selections):,} documents.'
        )

    return replace(
        expected_stats,
        path=output_path,
        selection_path=selection_path,
        actual_tokens=tokens,
        documents=docs,
    )


def reconstruct_from_auxiliary(
    config_path: str,
    output_dir: str | None,
    overwrite: bool,
) -> None:
    """Rebuild generated corpora from manifests, selections, and locked sources.

    Args:
        config_path: Dataset preparation config file.
        output_dir: Optional output directory override.
        overwrite: Whether to replace generated processed JSONL files.

    Returns:
        None.
    """
    resolved_config = resolve_config(
        load_config(config_path),
        output_dir=output_dir,
        target_scale=1.0,
        max_kuatia_docs=None,
        allow_incomplete_samples=False,
    )
    sources = source_specs_from_config(resolved_config)
    components_config = require_mapping(resolved_config, 'components')
    corpora_config = require_mapping(resolved_config, 'corpora')
    heldout_config = heldout_config_from_config(resolved_config)
    output_dir = resolved_config['output_dir']
    paths = output_paths(output_dir)  # type: ignore

    locked_sources = load_source_revisions_lock(paths['source_lock'])
    validate_locked_source_revisions(locked_sources)
    sources = pin_sources_to_revisions(sources, locked_sources)

    manifests = load_manifests(paths['manifests'], corpora_config.keys())
    expected_components = collect_component_stats(manifests)
    paths = ensure_writable_output(
        output_dir,  # type: ignore
        overwrite,
        corpus_names=corpora_config.keys(),
        component_names=expected_components.keys(),
        auxiliary=False,
        heldout_enabled=bool(heldout_config['enabled']),
    )

    tokenizer_instance = AutoTokenizer.from_pretrained(resolved_config['tokenizer'])
    seed = int(resolved_config['seed'])
    shuffle_buffer_size = int(resolved_config.get('shuffle_buffer_size', 10000))
    component_stats: dict[str, ComponentStats] = {}

    for component_name, component_config in components_config.items():
        if component_name not in expected_components:
            continue

        expected = component_stats_with_paths(expected_components[component_name], paths)
        output_path = expected.path
        selection_path = expected.selection_path
        kind = component_config.get('kind')

        if kind == 'all':
            source = sources[component_config['source']]
            actual = reconstruct_stream_component(
                component_name,
                expected,
                source,
                tokenizer_instance,
                output_path,
                selection_path,
                shuffle=False,
                seed=0,
                shuffle_buffer_size=1,
            )
        elif kind == 'sample':
            source = sources[component_config['source']]
            actual = reconstruct_stream_component(
                component_name,
                expected,
                source,
                tokenizer_instance,
                output_path,
                selection_path,
                shuffle=True,
                seed=expected.seed if expected.seed is not None else seed,
                shuffle_buffer_size=shuffle_buffer_size,
            )
        elif kind == 'fixed_subsample':
            source_component = component_config['source_component']
            if source_component not in component_stats:
                raise RuntimeError(
                    f'Cannot reconstruct {component_name} before source component '
                    f'{source_component}.'
                )
            actual = reconstruct_fixed_subsample_component(
                expected,
                component_stats[source_component],
                output_path,
                selection_path,
            )
        else:
            raise ValueError(f'Unsupported component kind: {kind}')

        verify_component_stats(expected, actual)
        component_stats[component_name] = actual
        print(
            f'[component] {component_name}: {actual.documents:,} docs / '
            f'{actual.actual_tokens:,} tokens',
            flush=True,
        )

    for config_name, manifest in manifests.items():
        component_names = manifest['corpus']['components']
        components = [component_stats[name] for name in component_names]
        corpus, heldout_corpus = write_corpus_splits(
            config_name,
            components,
            os.path.join(paths['processed'], f'{config_name}.jsonl'),
            os.path.join(paths['perplexity'], f'{config_name}.jsonl'),
            heldout_config,
        )
        verify_split_manifest(corpus, heldout_corpus, manifest)
        print(
            f'[corpus] {config_name}: {corpus["documents"]:,} docs / '
            f'{corpus["tokens"]:,} tokens',
            flush=True,
        )
        if heldout_corpus['documents']:
            print(
                f'[heldout] {config_name}: {heldout_corpus["documents"]:,} docs / '
                f'{heldout_corpus["tokens"]:,} tokens',
                flush=True,
            )

    write_dataset_info(os.path.join(output_dir, 'dataset_info.json'), corpora_config.keys())  # type: ignore
    print(f'[done] reconstructed corpora under {output_dir}', flush=True)


@click.command(
    context_settings={'show_default': True},
    help='Build frozen CPT corpora and metadata from configured sources.',
)
@click.option(
    '--config',
    'config_path',
    type=click.Path(exists=True, dir_okay=False),
    default='configs/data/gemma4_cpt.yaml',
    help='Dataset preparation config file.',
)
@click.option(
    '--output-dir',
    type=click.Path(file_okay=False, dir_okay=True),
    default=None,
    help='Override output directory from the config.',
)
@click.option('--overwrite', is_flag=True)
@click.option(
    '--reconstruct',
    is_flag=True,
    help='Rebuild processed corpora from manifests, selections, and locked source revisions.',
)
@click.option(
    '--allow-incomplete-samples',
    is_flag=True,
    help='Write smaller FineWeb samples if a source is exhausted before reaching the target.',
)
@click.option(
    '--preflight-only',
    is_flag=True,
    help='Print source metadata and exit without writing corpora.',
)
@click.option(
    '--max-kuatia-docs',
    type=int,
    default=None,
    help='Development-only cap for Kuatia documents. Omit for the real corpus.',
)
@click.option(
    '--target-scale',
    type=float,
    default=1.0,
    help='Development-only multiplier for FineWeb target tokens. Keep 1.0 for real corpora.',
)
def main(
    config_path: str,
    output_dir: str | None,
    overwrite: bool,
    reconstruct: bool,
    allow_incomplete_samples: bool,
    preflight_only: bool,
    max_kuatia_docs: int | None,
    target_scale: float,
) -> None:
    """Build frozen CPT corpora and metadata from configured sources.

    Args:
        config_path: Dataset preparation config file.
        output_dir: Optional output directory override.
        overwrite: Whether to replace existing generated files.
        reconstruct: Whether to rebuild generated corpora from auxiliary files.
        allow_incomplete_samples: Whether undersized FineWeb samples are allowed.
        preflight_only: Whether to inspect source metadata without writing outputs.
        max_kuatia_docs: Optional development cap for Kuatia documents.
        target_scale: Development multiplier for FineWeb token targets.

    Returns:
        None.
    """
    if reconstruct:
        reconstruct_from_auxiliary(config_path, output_dir, overwrite)
        return

    resolved_config = resolve_config(
        load_config(config_path),
        output_dir=output_dir,
        target_scale=target_scale,
        max_kuatia_docs=max_kuatia_docs,
        allow_incomplete_samples=allow_incomplete_samples,
    )
    sources = source_specs_from_config(resolved_config)
    components_config = require_mapping(resolved_config, 'components')
    corpora_config = require_mapping(resolved_config, 'corpora')
    heldout_config = heldout_config_from_config(resolved_config)

    if preflight_only:
        preflight(list(sources.values()))
        return

    output_dir = resolved_config['output_dir']
    tokenizer = resolved_config['tokenizer']
    seed = int(resolved_config['seed'])
    shuffle_buffer_size = int(resolved_config.get('shuffle_buffer_size', 10000))
    allow_incomplete = bool(resolved_config['allow_incomplete_samples'])
    max_docs = resolved_config['max_kuatia_docs']
    scale = float(resolved_config['target_scale'])

    paths = ensure_writable_output(
        output_dir,  # type: ignore
        overwrite,
        corpus_names=corpora_config.keys(),
        component_names=components_config.keys(),
        auxiliary=True,
        heldout_enabled=bool(heldout_config['enabled']),
    )
    locked_sources = resolve_source_revisions(sources)
    sources = pin_sources_to_revisions(sources, locked_sources)
    write_source_revisions_lock(paths['source_lock'], locked_sources)

    tokenizer_instance = AutoTokenizer.from_pretrained(tokenizer)

    component_stats: dict[str, ComponentStats] = {}
    for component_name, component_config in components_config.items():
        if component_config.get('kind') != 'all':
            continue

        source = sources[component_config['source']]
        component_stats[component_name] = write_kuatia_component(
            component_name,
            source,
            tokenizer_instance,
            os.path.join(paths['components'], f'{component_name}.jsonl'),
            selection_file_path(paths, component_name),
            max_docs=max_docs,
        )

    kuatia_tokens = component_stats['kuatia'].actual_tokens
    print(f'[targets] Kuatia={kuatia_tokens:,}', flush=True)

    for component_name, component_config in components_config.items():
        if component_config.get('kind') != 'sample':
            continue

        target_tokens = round(kuatia_tokens * float(component_config['target_ratio_of_kuatia']) * scale)
        print(f'[targets] {component_name}={target_tokens:,}', flush=True)
        source = sources[component_config['source']]
        component_stats[component_name] = write_sample_component(
            source,
            tokenizer_instance,
            os.path.join(paths['components'], f'{component_name}.jsonl'),
            selection_file_path(paths, component_name),
            target_tokens=target_tokens,
            seed=seed,
            shuffle_buffer_size=shuffle_buffer_size,
            allow_incomplete=allow_incomplete,
        )

    for component_name, component_config in components_config.items():
        if component_config.get('kind') != 'fixed_subsample':
            continue

        source_component = component_config['source_component']
        source_stats = component_stats[source_component]
        target_tokens = round(kuatia_tokens * float(component_config['target_ratio_of_kuatia']) * scale)
        print(f'[targets] {component_name}={target_tokens:,}', flush=True)
        component_stats[component_name] = write_fixed_subsample(
            source_stats,
            source_stats.path,
            source_stats.selection_path,
            os.path.join(paths['components'], f'{component_name}.jsonl'),
            selection_file_path(paths, component_name),
            target_tokens=target_tokens,
        )

    for config_name, corpus_config in corpora_config.items():
        components = [component_stats[component_name] for component_name in corpus_config['components']]
        corpus, heldout_corpus = write_corpus_splits(
            config_name,
            components,
            os.path.join(paths['processed'], f'{config_name}.jsonl'),
            os.path.join(paths['perplexity'], f'{config_name}.jsonl'),
            heldout_config,
        )
        write_manifest(
            os.path.join(paths['manifests'], f'{config_name}.manifest.json'),
            config_name=config_name,
            config_path=config_path,
            tokenizer=tokenizer,
            seed=seed,
            corpus=corpus,
            heldout_corpus=heldout_corpus,
            components=components,
            extra={
                'target_scale': scale,
                'max_kuatia_docs': max_docs,
            },
        )
        print(
            f'[corpus] {config_name}: {corpus["documents"]:,} docs / {corpus["tokens"]:,} tokens',
            flush=True,
        )
        if heldout_corpus['documents']:
            print(
                f'[heldout] {config_name}: {heldout_corpus["documents"]:,} docs / '
                f'{heldout_corpus["tokens"]:,} tokens',
                flush=True,
            )

    write_dataset_info(os.path.join(output_dir, 'dataset_info.json'), corpora_config.keys())  # type: ignore
    print(f'[done] wrote corpora under {output_dir}', flush=True)


if __name__ == '__main__':
    main()
