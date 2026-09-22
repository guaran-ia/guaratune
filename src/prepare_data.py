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
import shutil
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
    requested_tokens: int | None
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


def is_manifest_config(config: dict[str, Any]) -> bool:
    """Return whether a config uses the config-backed corpus scheme.

    Args:
        config: Resolved dataset preparation config.

    Returns:
        True when sources and corpus recipes are declared in the config.
    """
    corpora = config.get('corpora')
    if not isinstance(corpora, dict) or not corpora:
        return False

    return isinstance(config.get('sources'), dict) and all(
        isinstance(corpus, dict) and 'kuatia' in corpus
        for corpus in corpora.values()
    )


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
        'train': os.path.join(output_dir, 'train'),
        'components': os.path.join(output_dir, 'train', 'components'),
        'validation': os.path.join(output_dir, 'validation'),
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
        heldout_enabled: Whether held-out validation files are generated.

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

    generated = [os.path.join(paths['train'], f'{name}.jsonl') for name in corpus_names]
    if heldout_enabled:
        generated.extend(
            os.path.join(paths['validation'], f'{name}.jsonl') for name in corpus_names
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
    corpus_id_by_name: dict[str, str] | None = None,
) -> ComponentStats:
    """Write the Kuatia component and its document-selection ledger.

    Args:
        component_name: Component name used in outputs and manifests.
        spec: Kuatia source descriptor.
        tokenizer: Tokenizer used to count document tokens.
        output_path: Destination component JSONL path.
        selection_path: Destination selection ledger JSONL path.
        max_docs: Optional development cap on the number of documents written.
        corpus_id_by_name: Optional mapping from Kuatia corpus names to stable ids.

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
            source_corpus = row.get('source_corpus') or row.get('corpus')
            corpus_id = row.get('corpus_id')
            if corpus_id_by_name is not None and source_corpus:
                corpus_id = corpus_id_by_name.get(source_corpus, corpus_id)
            corpus_id = corpus_id or source_corpus
            if source_corpus:
                selection['corpus'] = source_corpus
                selection['source_corpus'] = source_corpus
            if corpus_id:
                selection['corpus_id'] = corpus_id

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
        requested_tokens=None,
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
    requested_tokens: int,
    seed: int,
    shuffle_buffer_size: int,
    allow_incomplete: bool,
) -> ComponentStats:
    """Sample full documents until an approximate token request is reached.

    Args:
        spec: FineWeb source descriptor.
        tokenizer: Tokenizer used to count document tokens.
        output_path: Destination component JSONL path.
        selection_path: Destination selection ledger JSONL path.
        requested_tokens: Approximate minimum token count to sample.
        seed: Random seed for deterministic sampling.
        shuffle_buffer_size: Shuffle buffer size for compatible streaming sources.
        allow_incomplete: Whether to accept samples smaller than requested_tokens.

    Returns:
        Component statistics for the sampled corpus.

    Raises:
        RuntimeError: If the source is exhausted before requested_tokens and
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
                    f'{docs:,} docs / {tokens:,} of {requested_tokens:,} requested tokens',
                    flush=True,
                )
            if tokens >= requested_tokens:
                break

    if tokens < requested_tokens and not allow_incomplete:
        raise RuntimeError(
            f'{spec.name} exhausted at {tokens:,} tokens before reaching requested minimum {requested_tokens:,}. '
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
        requested_tokens=requested_tokens,
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
    requested_tokens: int,
) -> ComponentStats:
    """Write a deterministic fixed subsample from an existing component.

    Args:
        source_stats: Statistics for the source component being subsampled.
        source_path: JSONL path for the source component.
        source_selection_path: Selection ledger path for the source component.
        output_path: Destination subsample JSONL path.
        selection_path: Destination subsample ledger path.
        requested_tokens: Approximate minimum token count to include in the subsample.

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
            if tokens >= requested_tokens:
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
        requested_tokens=requested_tokens,
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
    info = {}
    for config_name in corpus_names:
        info[config_name] = {
            'file_name': f'train/{config_name}.jsonl',
            'columns': {'prompt': 'text'},
        }
        info[f'{config_name}_validation'] = {
            'file_name': f'validation/{config_name}.jsonl',
            'columns': {'prompt': 'text'},
        }

    with open(path, 'w', encoding='utf-8') as handle:
        handle.write(json.dumps(info, ensure_ascii=False, indent=2) + '\n')


def component_path_from_selection(output_dir: str, selection_path: str) -> str:
    """Infer a train component path from a selection ledger path.

    Args:
        output_dir: Root directory for generated dataset artifacts.
        selection_path: Component selection ledger path.

    Returns:
        Train component JSONL path matching the selection ledger basename.
    """
    filename = os.path.basename(selection_path)
    if filename.endswith(SELECTION_SUFFIX):
        component_name = filename[: -len(SELECTION_SUFFIX)]
    elif filename.endswith('.selection.jsonl'):
        component_name = filename[: -len('.selection.jsonl')]
    else:
        component_name = os.path.splitext(filename)[0]

    selection_dir = os.path.dirname(selection_path)
    if os.path.basename(selection_dir) == 'selections':
        artifact_root = os.path.dirname(selection_dir)
    else:
        artifact_root = output_dir

    return os.path.join(
        artifact_root, 'train', 'components', f'{component_name}.jsonl'
    )


def manifest_output_paths(output_dir: str) -> dict[str, str]:
    """Create standard output directories for config-backed corpora.

    Args:
        output_dir: Root directory where generated dataset artifacts are written.

    Returns:
        Mapping of output directory names to paths.
    """
    paths = output_paths(output_dir)
    for key in ['train', 'validation', 'components', 'selections']:
        os.makedirs(paths[key], exist_ok=True)
    return paths


def ensure_manifest_outputs(
    output_dir: str,
    overwrite: bool,
    corpus_names: Iterable[str],
    component_paths: Iterable[str] = (),
    selection_paths: Iterable[str] = (),
) -> dict[str, str]:
    """Create manifest output directories and guard generated files.

    Args:
        output_dir: Root directory where generated dataset artifacts are written.
        overwrite: Whether existing generated outputs may be replaced.
        corpus_names: Corpus names expected to be generated.
        component_paths: Source component files expected to be generated.
        selection_paths: Selection ledger files expected to be generated.

    Returns:
        Standard output paths.

    Raises:
        FileExistsError: If generated files already exist and overwrite is false.
    """
    paths = manifest_output_paths(output_dir)
    generated = [os.path.join(paths['train'], f'{name}.jsonl') for name in corpus_names]
    generated.extend(os.path.join(paths['validation'], f'{name}.jsonl') for name in corpus_names)
    generated.extend(
        os.path.join(paths['manifests'], f'{name}.manifest.json')
        for name in corpus_names
    )
    generated.extend(component_paths)
    generated.extend(selection_paths)
    generated.append(os.path.join(output_dir, 'dataset_info.json'))
    generated.append(paths['source_lock'])

    existing = [path for path in generated if os.path.exists(path)]
    if existing and not overwrite:
        formatted = '\n'.join(f'  {path}' for path in existing)
        raise FileExistsError(
            f'Refusing to overwrite existing generated files:\n{formatted}'
        )

    return paths


def source_lock_from_manifest_config(config: dict[str, Any]) -> dict[str, Any]:
    """Build a source revision lock from config-backed source entries.

    Args:
        config: Resolved config-backed data config.

    Returns:
        Source revision lock entries keyed by source name.
    """
    locked_sources = {}
    for key, source in require_mapping(config, 'sources').items():
        name = source.get('name', key)
        locked_sources[name] = {
            'dataset': source['dataset'],
            'requested_revision': source.get('requested_revision'),
            'resolved_revision': source['commit_id'],
        }

    return locked_sources


def manifest_source_entry(
    source_name: str,
    source_config: dict[str, Any],
    *,
    include_selection_path: bool,
) -> dict[str, Any]:
    """Build source metadata for a generated data-preparation manifest.

    Args:
        source_name: Source key from the data config.
        source_config: Source metadata from the data config.
        include_selection_path: Whether to include the source-level selection path.

    Returns:
        Source metadata with empty run accounting.
    """
    entry = {
        'name': source_config.get('name', source_name),
        'dataset': source_config['dataset'],
        'version': source_config.get('version'),
        'loader': source_config.get('loader'),
        'requested_revision': source_config.get('requested_revision'),
        'commit_id': source_config.get('commit_id'),
        'documents': 0,
        'tokens': 0,
    }
    if include_selection_path:
        entry['selection_path'] = source_config['selection_path']

    return entry


def split_dataset_metadata(split_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return Kuatia dataset metadata keyed by configured dataset name.

    Args:
        split_config: Parsed Kuatia split configuration.

    Returns:
        Mapping from dataset name to metadata.
    """
    return {
        item['name']: item
        for item in split_config.get('datasets', [])
        if isinstance(item, dict) and 'name' in item
    }


def kuatia_corpus_id_map(config: dict[str, Any]) -> dict[str, str]:
    """Return Kuatia corpus-name to corpus-id mappings from the split config.

    Args:
        config: Resolved config-backed data recipe.

    Returns:
        Mapping from configured Kuatia corpus names and ids to stable corpus ids.
    """
    split_config = load_config(config['split_config'])
    mapping = {}
    for item in split_config.get('datasets', []):
        if not isinstance(item, dict):
            continue
        name = item.get('name')
        corpus_id = item.get('corpus_id')
        if name and corpus_id:
            mapping[name] = corpus_id
            mapping[corpus_id] = corpus_id

    return mapping


def kuatia_split_components(
    split_config: dict[str, Any],
    split_name: str,
    *,
    include_synthetic: bool,
) -> list[dict[str, Any]]:
    """Build Kuatia manifest components for one configured split.

    Args:
        split_config: Parsed Kuatia split configuration.
        split_name: Split name, usually train or validation.
        include_synthetic: Whether synthetic corpora should be included.

    Returns:
        Manifest component entries with empty run accounting.
    """
    datasets = split_dataset_metadata(split_config)
    synthetic_names = set(split_config.get('synthetic', []))
    components = []
    for dataset_name in split_config.get(split_name, []):
        metadata = datasets.get(dataset_name, {})
        corpus_id = metadata.get('corpus_id', dataset_name)
        synthetic = dataset_name in synthetic_names or corpus_id in synthetic_names
        if synthetic and not include_synthetic:
            continue

        components.append(
            {
                'name': dataset_name,
                'corpus_id': corpus_id,
                'synthetic': synthetic,
                'documents': 0,
                'tokens': 0,
            }
        )

    return components


def augmentation_display_name(source_name: str, ratio: float) -> str:
    """Build a readable augmentation component name.

    Args:
        source_name: Source key from the data config.
        ratio: Augmentation ratio.

    Returns:
        Human-readable augmentation name.
    """
    source_labels = {
        'fineweb_edu_en': 'FineWeb-Edu English',
        'fineweb_edu_es': 'FineWeb-Edu Spanish',
    }
    source_label = source_labels.get(source_name, source_name)
    percent = ratio * 100
    if percent.is_integer():
        percent_label = f'{int(percent)}%'
    else:
        percent_label = f'{percent:g}%'

    return f'{source_label} {percent_label}'


def build_config_manifests(
    config_path: str, config: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Build in-memory manifests from tracked data configuration files.

    Args:
        config_path: Dataset preparation config file.
        config: Resolved config-backed data config.

    Returns:
        Generated manifests keyed by corpus configuration name.
    """
    split_config_path = config.get('split_config', config_path)
    split_config = load_config(split_config_path)
    sources = require_mapping(config, 'sources')
    corpora_config = require_mapping(config, 'corpora')
    manifests = {}

    for config_name, corpus_config in corpora_config.items():
        include_synthetic = bool(
            corpus_config.get('kuatia', {}).get('include_synthetic', True)
        )
        manifest_sources = {
            'kuatia': manifest_source_entry(
                'kuatia', sources['kuatia'], include_selection_path=False
            )
        }
        splits = {
            split_name: {
                'documents': 0,
                'tokens': 0,
                'corpora': kuatia_split_components(
                    split_config,
                    split_name,
                    include_synthetic=include_synthetic,
                ),
            }
            for split_name in ('train', 'validation')
        }

        for augmentation in corpus_config.get('augmentations', []):
            source_name = augmentation['source']
            ratio = float(augmentation['ratio'])
            manifest_sources[source_name] = manifest_source_entry(
                source_name, sources[source_name], include_selection_path=True
            )
            component_name = augmentation_display_name(source_name, ratio)
            for split_name, selection_key in (
                ('train', 'train_selection_path'),
                ('validation', 'validation_selection_path'),
            ):
                splits[split_name]['corpora'].append(
                    {
                        'name': component_name,
                        'source': source_name,
                        'selection': 'sampled_document_level',
                        'documents': 0,
                        'tokens': 0,
                        'seed': int(config['seed']),
                        'selection_path': augmentation[selection_key],
                    }
                )

        manifests[config_name] = {
            'config_name': config_name,
            'created_at': datetime.now(UTC).isoformat(),
            'config_path': split_config_path,
            'selection_path': sources['kuatia']['selection_path'],
            'tokenizer': config['tokenizer'],
            'sources': manifest_sources,
            'sample_seed': int(config['seed']),
            'splits': splits,
        }

    return manifests


def write_config_manifests(
    manifests: dict[str, dict[str, Any]], manifest_dir: str
) -> None:
    """Write generated data-preparation manifests.

    Args:
        manifests: Generated manifest data keyed by corpus name.
        manifest_dir: Destination manifest directory.

    Returns:
        None.
    """
    os.makedirs(manifest_dir, exist_ok=True)
    for config_name, manifest in manifests.items():
        manifest['created_at'] = datetime.now(UTC).isoformat()
        path = os.path.join(manifest_dir, f'{config_name}.manifest.json')
        with open(path, 'w', encoding='utf-8') as handle:
            handle.write(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')


def load_json(path: str) -> dict[str, Any]:
    """Load a JSON object from disk.

    Args:
        path: JSON file path.

    Returns:
        Parsed JSON object.
    """
    with open(path, 'r', encoding='utf-8') as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f'JSON file must contain an object: {path}')

    return data


def load_selection_tokens(selection_path: str) -> tuple[dict[str, int], int, int]:
    """Load selected document ids, token counts, and totals from a ledger.

    Args:
        selection_path: Selection ledger JSONL path.

    Returns:
        Tuple of selected token counts by id, document count, and token count.

    Raises:
        RuntimeError: If a selection ledger contains duplicate ids.
    """
    tokens_by_id = {}
    documents = 0
    tokens = 0
    for row in stream_jsonl(selection_path):
        doc_id = row['id']
        if doc_id in tokens_by_id:
            raise RuntimeError(f'Duplicate document id in {selection_path}: {doc_id}')
        doc_tokens = int(row['tokens'])
        tokens_by_id[doc_id] = doc_tokens
        documents += 1
        tokens += doc_tokens

    return tokens_by_id, documents, tokens


def load_selection_corpus_totals(selection_path: str) -> dict[str, tuple[int, int]]:
    """Load document and token totals grouped by corpus id from a ledger.

    Args:
        selection_path: Selection ledger JSONL path.

    Returns:
        Mapping from corpus id to document and token totals.
    """
    totals = {}
    for row in stream_jsonl(selection_path):
        corpus_id = row.get('corpus_id')
        if not corpus_id:
            continue

        documents, tokens = totals.get(corpus_id, (0, 0))
        totals[corpus_id] = (documents + 1, tokens + int(row['tokens']))

    return totals


def stream_component_records(
    component_path: str, selection_path: str
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Yield aligned train component records and selection metadata.

    Args:
        component_path: Train component JSONL path containing text records.
        selection_path: Component selection ledger path containing token metadata.

    Returns:
        Iterator over record and selection pairs.

    Raises:
        RuntimeError: If record ids and selection ids are not aligned.
    """
    for record, selection in zip(
        stream_jsonl(component_path), stream_jsonl(selection_path), strict=True
    ):
        if record['id'] != selection['id']:
            raise RuntimeError(
                f'Component and selection ids differ for {component_path}: '
                f'{record["id"]} != {selection["id"]}.'
            )
        yield record, selection


def write_kuatia_manifest_records(
    output_handle: Any,
    *,
    output_dir: str,
    source_config: dict[str, Any],
    allowed_corpus_ids: set[str],
) -> tuple[int, int]:
    """Write Kuatia records whose corpus ids belong to a manifest split.

    Args:
        output_handle: Open JSONL output handle.
        output_dir: Root directory containing train components.
        source_config: Kuatia source configuration.
        allowed_corpus_ids: Corpus ids included in the target split.

    Returns:
        Pair of written document and token counts.
    """
    selection_path = source_config['selection_path']
    component_path = component_path_from_selection(output_dir, selection_path)
    documents = 0
    tokens = 0

    for record, selection in stream_component_records(component_path, selection_path):
        if selection.get('corpus_id') not in allowed_corpus_ids:
            continue
        output_handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        documents += 1
        tokens += int(selection['tokens'])

    return documents, tokens


def kuatia_cache_path(output_dir: str, allowed_corpus_ids: set[str]) -> str:
    """Build a cache path for a reusable Kuatia split component.

    Args:
        output_dir: Root directory for generated dataset artifacts.
        allowed_corpus_ids: Corpus ids included in the cached split component.

    Returns:
        Cache JSONL path.
    """
    joined_ids = '\n'.join(sorted(allowed_corpus_ids))
    digest = hashlib.sha256(joined_ids.encode('utf-8')).hexdigest()[:16]
    return os.path.join(
        output_dir,
        'train',
        'components',
        f'kuatia_split_{digest}.jsonl',
    )


def write_kuatia_manifest_records_cached(
    output_handle: Any,
    *,
    output_dir: str,
    source_config: dict[str, Any],
    allowed_corpus_ids: set[str],
    cache: dict[tuple[str, ...], dict[str, Any]],
) -> tuple[int, int]:
    """Write Kuatia records using a reusable cached split component.

    Args:
        output_handle: Open JSONL output handle.
        output_dir: Root directory containing generated dataset artifacts.
        source_config: Kuatia source configuration.
        allowed_corpus_ids: Corpus ids included in the target split.
        cache: Mutable cache metadata keyed by selected corpus ids.

    Returns:
        Pair of written document and token counts.
    """
    key = tuple(sorted(allowed_corpus_ids))
    if key not in cache:
        cache_path = kuatia_cache_path(output_dir, allowed_corpus_ids)
        print(
            f'[component] kuatia split cache: writing {cache_path}',
            flush=True,
        )
        with open(cache_path, 'w', encoding='utf-8') as cache_handle:
            documents, tokens = write_kuatia_manifest_records(
                cache_handle,
                output_dir=output_dir,
                source_config=source_config,
                allowed_corpus_ids=allowed_corpus_ids,
            )
        cache[key] = {
            'path': cache_path,
            'documents': documents,
            'tokens': tokens,
        }
        print(
            f'[component] kuatia split cache: {documents:,} docs / '
            f'{tokens:,} tokens',
            flush=True,
        )

    cached = cache[key]
    with open(cached['path'], 'r', encoding='utf-8') as cache_handle:
        shutil.copyfileobj(cache_handle, output_handle)

    return int(cached['documents']), int(cached['tokens'])


def write_selected_source_records(
    output_handle: Any,
    *,
    output_dir: str,
    source_config: dict[str, Any],
    selected_tokens: dict[str, int],
) -> tuple[int, int]:
    """Write source records selected by a split-specific ledger.

    Args:
        output_handle: Open JSONL output handle.
        output_dir: Root directory containing train components.
        source_config: Source configuration for the component pool.
        selected_tokens: Selected document token counts keyed by document id.

    Returns:
        Pair of written document and token counts.

    Raises:
        RuntimeError: If the component pool does not contain every selected id.
    """
    selection_path = source_config['selection_path']
    component_path = component_path_from_selection(output_dir, selection_path)
    documents = 0
    tokens = 0
    found_ids = set()

    for record in stream_jsonl(component_path):
        doc_id = record['id']
        if doc_id not in selected_tokens:
            continue
        output_handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        documents += 1
        tokens += selected_tokens[doc_id]
        found_ids.add(doc_id)

        if len(found_ids) == len(selected_tokens):
            break

    missing_ids = set(selected_tokens) - found_ids
    if missing_ids:
        examples = ', '.join(sorted(missing_ids)[:5])
        raise RuntimeError(
            f'Source component {component_path} is missing {len(missing_ids):,} '
            f'selected documents. Examples: {examples}'
        )

    return documents, tokens


def validate_manifest_source_config(
    source_name: str, source_config: dict[str, Any], manifest_source: dict[str, Any]
) -> None:
    """Validate that a config source matches manifest source metadata.

    Args:
        source_name: Source key used in the config-backed recipe.
        source_config: Source metadata from the config.
        manifest_source: Source metadata from a manifest.

    Returns:
        None.

    Raises:
        RuntimeError: If source metadata conflicts.
    """
    checks = {
        'dataset': source_config.get('dataset'),
        'version': source_config.get('version'),
        'loader': source_config.get('loader'),
        'commit_id': source_config.get('commit_id'),
    }
    for key, expected in checks.items():
        if manifest_source.get(key) != expected:
            raise RuntimeError(
                f'Source mismatch for {source_name}.{key}: config has {expected}, '
                f'manifest has {manifest_source.get(key)}.'
            )


def validate_manifest_config_sources(
    config: dict[str, Any], manifest: dict[str, Any]
) -> None:
    """Validate manifest source metadata against the active config.

    Args:
        config: Resolved config-backed data recipe.
        manifest: Corpus manifest data.

    Returns:
        None.
    """
    sources_config = require_mapping(config, 'sources')
    if 'source' in manifest:
        validate_manifest_source_config('kuatia', sources_config['kuatia'], manifest['source'])
        return

    for source_name, manifest_source in manifest['sources'].items():
        validate_manifest_source_config(
            source_name, sources_config[source_name], manifest_source
        )


def source_spec_from_manifest_source(
    source_name: str, source_config: dict[str, Any]
) -> SourceSpec:
    """Build a source descriptor from a config-backed source config.

    Args:
        source_name: Source key in the data config.
        source_config: Manifest-backed source configuration.

    Returns:
        SourceSpec pinned to the configured source commit.
    """
    return SourceSpec(
        name=source_config.get('name', source_name),
        dataset=source_config['dataset'],
        config=optional_config(source_config.get('config') or source_config.get('version')),
        split=source_config.get('split', 'train'),
        text_column=source_config.get('text_column', 'text'),
        id_column=optional_config(source_config.get('id_column', 'id')),
        loader=source_config.get('loader', 'datasets'),
        revision=optional_config(
            source_config.get('revision') or source_config.get('commit_id')
        ),
    )


def manifest_source_metadata(
    manifests: dict[str, Any], source_name: str
) -> dict[str, Any] | None:
    """Return consistent source metadata from corpus manifests.

    Args:
        manifests: Loaded manifest dictionaries.
        source_name: Source key whose metadata is requested.

    Returns:
        Source metadata when present, otherwise None.

    Raises:
        RuntimeError: If manifests disagree about source metadata.
    """
    metadata = None
    for manifest in manifests.values():
        manifest_source = manifest.get('source') if source_name == 'kuatia' else None
        if manifest_source is None:
            sources = manifest.get('sources', {})
            if isinstance(sources, dict):
                manifest_source = sources.get(source_name)
        if manifest_source is None:
            continue
        if metadata is not None and manifest_source != metadata:
            raise RuntimeError(f'Manifests disagree about source metadata: {source_name}')
        metadata = manifest_source

    return metadata


def required_manifest_selection_paths(
    config: dict[str, Any], manifests: dict[str, Any]
) -> list[str]:
    """Collect selection ledger paths that will be generated.

    Args:
        config: Resolved config-backed data config.
        manifests: Loaded manifest dictionaries.

    Returns:
        Sorted selection ledger paths.
    """
    paths = set()
    sources = require_mapping(config, 'sources')
    if manifest_source_metadata(manifests, 'kuatia') is not None:
        paths.add(sources['kuatia']['selection_path'])

    for manifest in manifests.values():
        for split in manifest['splits'].values():
            for component in split['corpora']:
                if 'source' in component:
                    paths.add(component['selection_path'])

    return sorted(paths)


def required_manifest_component_paths(
    config: dict[str, Any], manifests: dict[str, Any]
) -> list[str]:
    """Collect source component paths that will be generated.

    Args:
        config: Resolved config-backed data config.
        manifests: Loaded manifest dictionaries.

    Returns:
        Sorted source component JSONL paths.
    """
    source_names = set()
    if manifest_source_metadata(manifests, 'kuatia') is not None:
        source_names.add('kuatia')

    for manifest in manifests.values():
        for split in manifest['splits'].values():
            for component in split['corpora']:
                source_name = component.get('source')
                if source_name:
                    source_names.add(source_name)

    sources = require_mapping(config, 'sources')
    return sorted(
        component_path_from_selection(
            config['output_dir'],  # type: ignore
            sources[source_name]['selection_path'],
        )
        for source_name in source_names
    )


def find_manifest_selection_component(
    manifest: dict[str, Any],
    split_name: str,
    selection_path: str,
    source_name: str,
) -> dict[str, Any]:
    """Find a sampled-source component in a manifest split.

    Args:
        manifest: Corpus manifest data.
        split_name: Split containing the sampled-source component.
        selection_path: Selection ledger path from the data config.
        source_name: Source key from the data config.

    Returns:
        Manifest component matching the selection path and source.

    Raises:
        RuntimeError: If no matching component exists.
    """
    for component in manifest['splits'][split_name]['corpora']:
        if (
            component.get('source') == source_name
            and component.get('selection_path') == selection_path
        ):
            return component

    raise RuntimeError(
        f'Manifest {manifest["config_name"]} {split_name} does not contain '
        f'{source_name} selection {selection_path}.'
    )


def manifest_selection_specs(
    config: dict[str, Any], manifests: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Collect unique sampled-source selection specs from config ratios.

    Args:
        config: Resolved config-backed data config.
        manifests: Loaded manifest dictionaries.

    Returns:
        Selection metadata keyed by selection path.

    Raises:
        RuntimeError: If one selection path has conflicting metadata.
    """
    specs = {}
    corpora_config = require_mapping(config, 'corpora')
    default_seed = int(config['seed'])
    for config_name, corpus_config in corpora_config.items():
        manifest = manifests[config_name]
        for augmentation in corpus_config.get('augmentations', []):
            source_name = augmentation['source']
            ratio = float(augmentation['ratio'])
            reference_name = augmentation.get('ratio_reference', config_name)
            if reference_name not in manifests:
                raise RuntimeError(
                    f'{config_name} references unknown ratio_reference {reference_name}.'
                )

            for split_name, selection_key in (
                ('train', 'train_selection_path'),
                ('validation', 'validation_selection_path'),
            ):
                selection_path = augmentation[selection_key]
                component = find_manifest_selection_component(
                    manifest, split_name, selection_path, source_name
                )
                requested_tokens = round(
                    int(manifests[reference_name]['splits'][split_name]['tokens']) * ratio
                )
                spec = {
                    'source': source_name,
                    'split': split_name,
                    'selection_path': selection_path,
                    'requested_tokens': requested_tokens,
                    'ratio': ratio,
                    'ratio_reference': reference_name,
                    'documents': int(component['documents']),
                    'tokens': int(component['tokens']),
                    'seed': int(component.get('seed', manifest.get('sample_seed', default_seed))),
                }
                existing = specs.get(selection_path)
                comparable = {
                    key: spec[key]
                    for key in (
                        'source',
                        'split',
                        'selection_path',
                        'requested_tokens',
                        'ratio',
                        'ratio_reference',
                        'seed',
                    )
                }
                existing_comparable = (
                    {
                        key: existing[key]
                        for key in (
                            'source',
                            'split',
                            'selection_path',
                            'requested_tokens',
                            'ratio',
                            'ratio_reference',
                            'seed',
                        )
                    }
                    if existing is not None
                    else None
                )
                if existing is not None and existing_comparable != comparable:
                    raise RuntimeError(
                        f'Conflicting manifest selection metadata: {selection_path}'
                    )
                if existing is None:
                    specs[selection_path] = spec

    return specs


def selection_entry_from_row(
    row: dict[str, Any],
    spec: SourceSpec,
    tokenizer: PreTrainedTokenizerBase,
    index: int,
) -> dict[str, Any] | None:
    """Build one selection ledger entry from a source row.

    Args:
        row: Raw dataset row.
        spec: Source descriptor.
        tokenizer: Tokenizer used for token accounting.
        index: Stream row index used for fallback ids.

    Returns:
        Selection entry, or None for empty text rows.
    """
    text = clean_text(row.get(spec.text_column))
    if not text:
        return None

    selection = {
        'id': row_id(row, spec.name, spec.id_column, index),
        'source': spec.name,
        'text': text,
        'tokens': token_count(tokenizer, text),
    }
    raw_metadata = row.get('metadata')
    metadata: dict[str, Any] = (
        raw_metadata if isinstance(raw_metadata, dict) else {}
    )
    url = row.get('url') or metadata.get('url')
    if url:
        selection['url'] = url

    return selection


def write_selection_entries(path: str, entries: list[dict[str, Any]]) -> tuple[int, int]:
    """Write selection ledger entries to disk.

    Args:
        path: Destination selection ledger path.
        entries: Selection entries to write.

    Returns:
        Pair of written document and token counts.
    """
    documents = 0
    tokens = 0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open_text(path, 'w') as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + '\n')
            documents += 1
            tokens += int(entry['tokens'])

    return documents, tokens


def selection_prefix_for_request(
    entries: list[dict[str, Any]], requested_tokens: int
) -> list[dict[str, Any]]:
    """Return the shortest entry prefix reaching a token request.

    Args:
        entries: Source selection entries.
        requested_tokens: Approximate minimum token count to reach.

    Returns:
        Prefix of entries reaching the requested token count.

    Raises:
        RuntimeError: If the entries do not reach requested_tokens.
    """
    selected = []
    tokens = 0
    for entry in entries:
        selected.append(entry)
        tokens += int(entry['tokens'])
        if tokens >= requested_tokens:
            return selected

    raise RuntimeError(
        f'Selection prefix reached {tokens:,} tokens before request {requested_tokens:,}.'
    )


def verify_selection_totals(
    path: str, documents: int, tokens: int, expected: dict[str, Any]
) -> None:
    """Validate generated selection totals against the approximate token request.

    Args:
        path: Selection path being checked.
        documents: Generated document count.
        tokens: Generated token count.
        expected: Selection metadata derived from config ratios.

    Returns:
        None.

    Raises:
        RuntimeError: If generated totals do not reach the requested token count.
    """
    requested_tokens = int(expected['requested_tokens'])
    if tokens < requested_tokens:
        raise RuntimeError(
            f'Generated selection {path} has {tokens:,} tokens, '
            f'below requested minimum {requested_tokens:,}.'
        )


def take_selection_entries(
    entry_stream: Iterator[dict[str, Any]],
    requested_tokens: int,
) -> list[dict[str, Any]]:
    """Consume selection entries until a token request is reached.

    Args:
        entry_stream: Selection entry iterator.
        requested_tokens: Approximate minimum token count to reach.

    Returns:
        Selected entries.

    Raises:
        RuntimeError: If the stream ends before the request is reached.
    """
    entries = []
    tokens = 0
    for entry in entry_stream:
        entries.append(entry)
        tokens += int(entry['tokens'])
        if tokens >= requested_tokens:
            return entries

    raise RuntimeError(
        f'Source stream ended at {tokens:,} tokens before request {requested_tokens:,}.'
    )


def iter_source_selection_entries(
    spec: SourceSpec,
    tokenizer: PreTrainedTokenizerBase,
    *,
    seed: int,
    shuffle_buffer_size: int,
) -> Iterator[dict[str, Any]]:
    """Yield deterministic shuffled selection entries for a source.

    Args:
        spec: Source descriptor.
        tokenizer: Tokenizer used for token accounting.
        seed: Shuffle seed.
        shuffle_buffer_size: Shuffle buffer size for compatible streaming sources.

    Returns:
        Iterator over non-empty document selection entries.
    """
    for index, row in enumerate(
        load_stream(
            spec,
            shuffle=True,
            seed=seed,
            buffer_size=shuffle_buffer_size,
        )
    ):
        entry = selection_entry_from_row(row, spec, tokenizer, index)
        if entry is not None:
            yield entry


def write_source_selection_specs(
    source_name: str,
    specs: list[dict[str, Any]],
    source_config: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    shuffle_buffer_size: int,
) -> None:
    """Generate selection ledgers for one non-Kuatia source.

    The largest train selection is sampled first. Validation is sampled from
    the continuation of the same shuffled stream, and smaller selections are
    deterministic prefixes of the largest selection for the same split.

    Args:
        source_name: Source key.
        specs: Selection specs for the source.
        source_config: Source configuration.
        tokenizer: Tokenizer used for token accounting.
        shuffle_buffer_size: Shuffle buffer size for compatible streaming sources.

    Returns:
        None.
    """
    seeds = {spec['seed'] for spec in specs}
    if len(seeds) != 1:
        raise RuntimeError(f'Multiple seeds for {source_name} selections: {sorted(seeds)}')

    spec = source_spec_from_manifest_source(source_name, source_config)
    entry_stream = iter_source_selection_entries(
        spec,
        tokenizer,
        seed=seeds.pop(),
        shuffle_buffer_size=shuffle_buffer_size,
    )

    for split_name in ('train', 'validation'):
        split_specs = [
            item for item in specs if item['split'] == split_name
        ]
        if not split_specs:
            continue

        max_requested_tokens = max(item['requested_tokens'] for item in split_specs)
        base_entries = take_selection_entries(entry_stream, max_requested_tokens)
        for selection_spec in sorted(
            split_specs,
            key=lambda item: (item['requested_tokens'], item['selection_path']),
            reverse=True,
        ):
            entries = selection_prefix_for_request(
                base_entries, selection_spec['requested_tokens']
            )
            documents, tokens = write_selection_entries(
                selection_spec['selection_path'], entries
            )
            verify_selection_totals(
                selection_spec['selection_path'], documents, tokens, selection_spec
            )
            print(
                f'[selection] {selection_spec["selection_path"]}: '
                f'{documents:,} docs / {tokens:,} tokens',
                flush=True,
            )


def write_manifest_selection_ledgers(
    config: dict[str, Any],
    manifests: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Generate all selection ledgers required by a config-backed run.

    Args:
        config: Resolved config-backed data config.
        manifests: Loaded manifest dictionaries.
        tokenizer: Tokenizer used for token accounting.

    Returns:
        None.
    """
    sources = require_mapping(config, 'sources')
    kuatia_metadata = manifest_source_metadata(manifests, 'kuatia')
    if kuatia_metadata is not None:
        kuatia_config = sources['kuatia']
        component_path = component_path_from_selection(
            config['output_dir'],  # type: ignore
            kuatia_config['selection_path'],
        )
        stats = write_kuatia_component(
            'kuatia',
            source_spec_from_manifest_source('kuatia', kuatia_config),
            tokenizer,
            component_path,
            kuatia_config['selection_path'],
            max_docs=None,
            corpus_id_by_name=kuatia_corpus_id_map(config),
        )
        print(
            f'[selection] {kuatia_config["selection_path"]}: '
            f'{stats.documents:,} docs / {stats.actual_tokens:,} tokens',
            flush=True,
        )

    refresh_manifest_kuatia_totals(manifests)
    specs_by_path = manifest_selection_specs(config, manifests)
    specs_by_source: dict[str, list[dict[str, Any]]] = {}
    for spec in specs_by_path.values():
        specs_by_source.setdefault(spec['source'], []).append(spec)

    shuffle_buffer_size = int(config.get('shuffle_buffer_size', 10000))
    for source_name, specs in sorted(specs_by_source.items()):
        write_source_selection_specs(
            source_name,
            specs,
            sources[source_name],
            tokenizer,
            shuffle_buffer_size,
        )


def refresh_manifest_kuatia_totals(manifests: dict[str, Any]) -> None:
    """Refresh in-memory Kuatia component totals from generated ledgers.

    Args:
        manifests: Loaded manifest dictionaries to update in place.

    Returns:
        None.
    """
    corpus_totals_by_path = {}
    selection_totals_by_path = {}
    for manifest in manifests.values():
        selection_path = manifest.get('selection_path')
        if not selection_path:
            continue

        if selection_path not in selection_totals_by_path:
            _, documents, tokens = load_selection_tokens(selection_path)
            selection_totals_by_path[selection_path] = (documents, tokens)
        if selection_path not in corpus_totals_by_path:
            corpus_totals_by_path[selection_path] = (
                load_selection_corpus_totals(selection_path)
            )

        documents, tokens = selection_totals_by_path[selection_path]
        manifest['sources']['kuatia']['documents'] = documents
        manifest['sources']['kuatia']['tokens'] = tokens

        for split in manifest['splits'].values():
            split_documents = 0
            split_tokens = 0
            for component in split['corpora']:
                if 'corpus_id' in component:
                    documents, tokens = corpus_totals_by_path[selection_path].get(
                        component['corpus_id'], (0, 0)
                    )
                    component['documents'] = documents
                    component['tokens'] = tokens

                split_documents += int(component['documents'])
                split_tokens += int(component['tokens'])

            split['documents'] = split_documents
            split['tokens'] = split_tokens


def refresh_manifest_selection_totals(manifests: dict[str, Any]) -> None:
    """Refresh in-memory manifest totals from generated selection ledgers.

    Args:
        manifests: Loaded manifest dictionaries to update in place.

    Returns:
        None.
    """
    totals_by_path = {}
    corpus_totals_by_path = {}
    for manifest in manifests.values():
        source_tokens_by_id = {}
        selection_path = manifest.get('selection_path')
        if selection_path:
            _, documents, tokens = load_selection_tokens(selection_path)
            manifest['sources']['kuatia']['documents'] = documents
            manifest['sources']['kuatia']['tokens'] = tokens

        for split in manifest['splits'].values():
            split_documents = 0
            split_tokens = 0
            for component in split['corpora']:
                if 'source' in component:
                    selection_path = component['selection_path']
                    if selection_path not in totals_by_path:
                        tokens_by_id, documents, tokens = load_selection_tokens(
                            selection_path
                        )
                        totals_by_path[selection_path] = (
                            tokens_by_id,
                            documents,
                            tokens,
                        )

                    tokens_by_id, documents, tokens = totals_by_path[selection_path]
                    component['documents'] = documents
                    component['tokens'] = tokens
                    source_name = component['source']
                    source_tokens_by_id.setdefault(source_name, {}).update(tokens_by_id)
                elif 'corpus_id' in component:
                    selection_path = manifest['selection_path']
                    if selection_path not in corpus_totals_by_path:
                        corpus_totals_by_path[selection_path] = (
                            load_selection_corpus_totals(selection_path)
                        )

                    documents, tokens = corpus_totals_by_path[selection_path].get(
                        component['corpus_id'], (0, 0)
                    )
                    component['documents'] = documents
                    component['tokens'] = tokens

                split_documents += int(component['documents'])
                split_tokens += int(component['tokens'])

            split['documents'] = split_documents
            split['tokens'] = split_tokens

        for source_name, tokens_by_id in source_tokens_by_id.items():
            manifest['sources'][source_name]['documents'] = len(tokens_by_id)
            manifest['sources'][source_name]['tokens'] = sum(tokens_by_id.values())


def add_selection_tokens(
    selected_tokens: dict[str, int], selection_path: str
) -> tuple[int, int]:
    """Merge selection ledger ids into a source-level selection map.

    Args:
        selected_tokens: Mutable token map keyed by document id.
        selection_path: Selection ledger to merge.

    Returns:
        Pair of document and token counts from the selection ledger.

    Raises:
        RuntimeError: If a repeated id has a different token count.
    """
    tokens_by_id, documents, tokens = load_selection_tokens(selection_path)
    for doc_id, doc_tokens in tokens_by_id.items():
        existing_tokens = selected_tokens.get(doc_id)
        if existing_tokens is not None and existing_tokens != doc_tokens:
            raise RuntimeError(
                f'Conflicting token counts for {doc_id}: '
                f'{existing_tokens} != {doc_tokens}.'
            )
        selected_tokens[doc_id] = doc_tokens

    return documents, tokens


def required_manifest_source_selections(
    config: dict[str, Any], manifests: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Collect source selections needed by active config-backed corpora.

    Args:
        config: Resolved config-backed data config.
        manifests: Loaded corpus manifests.

    Returns:
        Mapping from source name to output component metadata and selected ids.
    """
    required = {}
    sources = require_mapping(config, 'sources')
    for source_name, source_config in sources.items():
        required[source_name] = {
            'source_config': source_config,
            'component_path': component_path_from_selection(
                config['output_dir'],  # type: ignore
                source_config['selection_path'],
            ),
            'selected_tokens': {},
            'selection_paths': [],
        }

    for manifest in manifests.values():
        for split in manifest['splits'].values():
            for component in split['corpora']:
                source_name = component.get('source')
                if not source_name:
                    continue
                if source_name not in required:
                    raise ValueError(
                        f'Manifest component references unknown source: {source_name}'
                    )
                add_selection_tokens(
                    required[source_name]['selected_tokens'],
                    component['selection_path'],
                )
                if component['selection_path'] not in required[source_name]['selection_paths']:
                    required[source_name]['selection_paths'].append(component['selection_path'])

    return {
        source_name: metadata
        for source_name, metadata in required.items()
        if metadata['selected_tokens']
    }


def selection_ledgers_include_text(selection_paths: list[str]) -> bool:
    """Return whether every row in selection ledgers includes text.

    Args:
        selection_paths: Selection ledger paths to inspect.

    Returns:
        True when every non-empty selected row has text.
    """
    for selection_path in selection_paths:
        for row in stream_jsonl(selection_path):
            if not clean_text(row.get('text')):
                return False

    return True


def write_manifest_source_component_from_selection_text(
    source_name: str,
    metadata: dict[str, Any],
) -> bool:
    """Write one source component pool from text-bearing selection ledgers.

    Args:
        source_name: Source key being rebuilt.
        metadata: Source config, selected token map, selection paths, and component path.

    Returns:
        True if the component was written from selection text; otherwise False.

    Raises:
        RuntimeError: If selected ids or token counts are inconsistent.
    """
    selection_paths = metadata['selection_paths']
    if not selection_paths or not selection_ledgers_include_text(selection_paths):
        return False

    component_path = metadata['component_path']
    selected_tokens = metadata['selected_tokens']
    found_ids = set()
    documents = 0
    tokens = 0

    print(
        f'[component] {source_name}: writing {len(selected_tokens):,} selected docs '
        f'from local selection ledgers',
        flush=True,
    )
    os.makedirs(os.path.dirname(component_path), exist_ok=True)
    with open(component_path, 'w', encoding='utf-8') as output_handle:
        for selection_path in selection_paths:
            for row in stream_jsonl(selection_path):
                doc_id = row['id']
                if doc_id in found_ids:
                    continue
                if doc_id not in selected_tokens:
                    raise RuntimeError(
                        f'{selection_path} contains unrequested document id {doc_id}.'
                    )

                doc_tokens = int(row['tokens'])
                expected_tokens = selected_tokens[doc_id]
                if doc_tokens != expected_tokens:
                    raise RuntimeError(
                        f'Cannot rebuild {source_name}: token mismatch for {doc_id}. '
                        f'Expected {expected_tokens}, got {doc_tokens}.'
                    )

                output_handle.write(
                    json.dumps(
                        {
                            'id': doc_id,
                            'source': row.get('source', source_name),
                            'text': clean_text(row['text']),
                        },
                        ensure_ascii=False,
                    )
                    + '\n'
                )
                found_ids.add(doc_id)
                documents += 1
                tokens += doc_tokens

                if documents % 1000 == 0:
                    print(
                        f'[component] {source_name}: {documents:,} docs / '
                        f'{tokens:,} tokens',
                        flush=True,
                    )

    missing_ids = set(selected_tokens) - found_ids
    if missing_ids:
        examples = ', '.join(sorted(missing_ids)[:5])
        raise RuntimeError(
            f'Cannot rebuild {source_name}: selection ledgers are missing '
            f'{len(missing_ids):,} selected documents. Examples: {examples}'
        )

    print(
        f'[component] {source_name}: {documents:,} docs / {tokens:,} tokens',
        flush=True,
    )
    return True


def write_manifest_source_component(
    source_name: str,
    metadata: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Write one source component pool from current selection ledgers.

    Args:
        source_name: Source key being rebuilt.
        metadata: Source config, selected token map, and component path.
        tokenizer: Tokenizer used to verify source token counts.

    Returns:
        None.

    Raises:
        RuntimeError: If selected ids cannot be found or token counts changed.
    """
    if write_manifest_source_component_from_selection_text(source_name, metadata):
        return

    source_config = metadata['source_config']
    component_path = metadata['component_path']
    selected_tokens = metadata['selected_tokens']
    spec = source_spec_from_manifest_source(source_name, source_config)
    found_ids = set()
    documents = 0
    tokens = 0

    print(
        f'[component] {source_name}: scanning source for '
        f'{len(selected_tokens):,} selected docs',
        flush=True,
    )
    os.makedirs(os.path.dirname(component_path), exist_ok=True)
    with open(component_path, 'w', encoding='utf-8') as output_handle:
        for index, row in enumerate(
            load_stream(spec, shuffle=False, seed=0, buffer_size=1)
        ):
            if index > 0 and index % 500000 == 0:
                print(
                    f'[component] {source_name}: scanned {index:,} rows, '
                    f'found {documents:,}/{len(selected_tokens):,} docs',
                    flush=True,
                )

            doc_id = row_id(row, spec.name, spec.id_column, index)
            if doc_id not in selected_tokens:
                continue

            text = clean_text(row.get(spec.text_column))
            if not text:
                continue

            doc_tokens = token_count(tokenizer, text)
            expected_tokens = selected_tokens[doc_id]
            if doc_tokens != expected_tokens:
                raise RuntimeError(
                    f'Cannot rebuild {source_name}: token mismatch for {doc_id}. '
                    f'Expected {expected_tokens}, got {doc_tokens}.'
                )

            output_handle.write(
                json.dumps(
                    {'id': doc_id, 'source': spec.name, 'text': text},
                    ensure_ascii=False,
                )
                + '\n'
            )
            found_ids.add(doc_id)
            documents += 1
            tokens += doc_tokens

            if documents % 1000 == 0:
                print(
                    f'[component] {source_name}: {documents:,} docs / {tokens:,} tokens',
                    flush=True,
                )

            if len(found_ids) == len(selected_tokens):
                break

    missing_ids = set(selected_tokens) - found_ids
    if missing_ids:
        examples = ', '.join(sorted(missing_ids)[:5])
        raise RuntimeError(
            f'Cannot rebuild {source_name}: source ended before '
            f'{len(missing_ids):,} selected documents were found. Examples: {examples}'
        )

    print(
        f'[component] {source_name}: {documents:,} docs / {tokens:,} tokens',
        flush=True,
    )


def write_manifest_source_components(
    required_sources: dict[str, dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Write all source component pools required by active manifests.

    Args:
        required_sources: Source selection metadata keyed by source name.
        tokenizer: Tokenizer used to verify source token counts.

    Returns:
        None.
    """
    for source_name, metadata in required_sources.items():
        write_manifest_source_component(source_name, metadata, tokenizer)


def source_config_for_manifest_component(
    config: dict[str, Any], component: dict[str, Any]
) -> dict[str, Any]:
    """Return the source config for a non-Kuatia manifest component.

    Args:
        config: Resolved config-backed data config.
        component: Manifest split component.

    Returns:
        Source config referenced by the component.
    """
    source_name = component.get('source')
    if not source_name:
        raise ValueError(f'Manifest component has no source: {component}')

    sources = require_mapping(config, 'sources')
    if source_name not in sources:
        raise ValueError(f'Manifest component references unknown source: {source_name}')

    return sources[source_name]


def write_manifest_split_file(
    *,
    config: dict[str, Any],
    manifest: dict[str, Any],
    split_name: str,
    output_path: str,
    kuatia_cache: dict[tuple[str, ...], dict[str, Any]],
) -> dict[str, Any]:
    """Write one manifest split to a JSONL file.

    Args:
        config: Resolved config-backed data config.
        manifest: Corpus manifest data.
        split_name: Split name to write, usually train or validation.
        output_path: Destination JSONL path.
        kuatia_cache: Reusable Kuatia split component cache.

    Returns:
        Written split summary.
    """
    output_dir = config['output_dir']
    split = manifest['splits'][split_name]
    kuatia_source = require_mapping(config, 'sources')['kuatia']
    kuatia_corpus_ids = {
        corpus['corpus_id']
        for corpus in split['corpora']
        if 'corpus_id' in corpus and int(corpus['documents']) > 0
    }
    documents = 0
    tokens = 0

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as output_handle:
        if kuatia_corpus_ids:
            docs, toks = write_kuatia_manifest_records_cached(
                output_handle,
                output_dir=output_dir,  # type: ignore
                source_config=kuatia_source,
                allowed_corpus_ids=kuatia_corpus_ids,
                cache=kuatia_cache,
            )
            documents += docs
            tokens += toks

        for component in split['corpora']:
            if 'source' not in component:
                continue
            selected_tokens, expected_docs, expected_tokens = load_selection_tokens(
                component['selection_path']
            )
            if expected_docs != int(component['documents']) or expected_tokens != int(component['tokens']):
                raise RuntimeError(
                    f'{manifest["config_name"]} {split_name} selection totals do not '
                    f'match component {component["name"]}.'
                )
            source_config = source_config_for_manifest_component(config, component)
            docs, toks = write_selected_source_records(
                output_handle,
                output_dir=output_dir,  # type: ignore
                source_config=source_config,
                selected_tokens=selected_tokens,
            )
            if docs != expected_docs or toks != expected_tokens:
                raise RuntimeError(
                    f'{manifest["config_name"]} {split_name} wrote {docs:,} docs / '
                    f'{toks:,} tokens for {component["name"]}, expected '
                    f'{expected_docs:,} docs / {expected_tokens:,} tokens.'
                )
            documents += docs
            tokens += toks

    expected_documents = int(split['documents'])
    expected_tokens = int(split['tokens'])
    if documents != expected_documents or tokens != expected_tokens:
        raise RuntimeError(
            f'{manifest["config_name"]} {split_name} wrote {documents:,} docs / '
            f'{tokens:,} tokens, expected {expected_documents:,} docs / '
            f'{expected_tokens:,} tokens.'
        )

    return {
        'name': manifest['config_name'],
        'path': output_path,
        'documents': documents,
        'tokens': tokens,
    }


def write_manifest_config_outputs(
    config_path: str,
    config: dict[str, Any],
    output_dir: str | None,
    overwrite: bool,
) -> None:
    """Write train, validation, and manifest files from a config-backed recipe.

    Args:
        config_path: Dataset preparation config file.
        config: Loaded config-backed data recipe.
        output_dir: Optional output directory override.
        overwrite: Whether to replace generated outputs.

    Returns:
        None.
    """
    resolved_config = resolve_config(
        config,
        output_dir=output_dir,
        target_scale=1.0,
        max_kuatia_docs=None,
        allow_incomplete_samples=False,
    )
    corpora_config = require_mapping(resolved_config, 'corpora')
    output_root = resolved_config['output_dir']
    manifests = build_config_manifests(config_path, resolved_config)

    paths = ensure_manifest_outputs(
        output_root,  # type: ignore
        overwrite,
        corpora_config.keys(),
        component_paths=required_manifest_component_paths(resolved_config, manifests),
        selection_paths=required_manifest_selection_paths(resolved_config, manifests),
    )

    locked_sources = source_lock_from_manifest_config(resolved_config)
    write_source_revisions_lock(paths['source_lock'], locked_sources)
    tokenizer_instance = AutoTokenizer.from_pretrained(resolved_config['tokenizer'])
    write_manifest_selection_ledgers(resolved_config, manifests, tokenizer_instance)
    refresh_manifest_selection_totals(manifests)
    required_sources = required_manifest_source_selections(resolved_config, manifests)
    write_manifest_source_components(required_sources, tokenizer_instance)

    kuatia_cache: dict[tuple[str, ...], dict[str, Any]] = {}
    for config_name, manifest in manifests.items():
        train_summary = write_manifest_split_file(
            config=resolved_config,
            manifest=manifest,
            split_name='train',
            output_path=os.path.join(paths['train'], f'{config_name}.jsonl'),
            kuatia_cache=kuatia_cache,
        )
        validation_summary = write_manifest_split_file(
            config=resolved_config,
            manifest=manifest,
            split_name='validation',
            output_path=os.path.join(paths['validation'], f'{config_name}.jsonl'),
            kuatia_cache=kuatia_cache,
        )
        print(
            f'[corpus] {config_name}: {train_summary["documents"]:,} docs / '
            f'{train_summary["tokens"]:,} tokens',
            flush=True,
        )
        print(
            f'[validation] {config_name}: {validation_summary["documents"]:,} docs / '
            f'{validation_summary["tokens"]:,} tokens',
            flush=True,
        )

    write_config_manifests(manifests, paths['manifests'])
    write_dataset_info(os.path.join(output_root, 'dataset_info.json'), corpora_config.keys())  # type: ignore
    print(f'[done] wrote config-backed corpora under {output_root}', flush=True)


def preflight_manifest_config(config: dict[str, Any]) -> None:
    """Print config-backed data config metadata without writing outputs.

    Args:
        config: Resolved config-backed data config.

    Returns:
        None.
    """
    print('sources:')
    for source_name, source in require_mapping(config, 'sources').items():
        print(f'  {source_name}: {source["dataset"]}@{source["commit_id"]}')
        print(f'    selection: {source["selection_path"]}')
    print('corpora:')
    for corpus_name, corpus in require_mapping(config, 'corpora').items():
        include_synthetic = corpus.get('kuatia', {}).get('include_synthetic', True)
        print(f'  {corpus_name}: include_synthetic={include_synthetic}')
        for augmentation in corpus.get('augmentations', []):
            print(
                f'    {augmentation["source"]}: ratio={augmentation["ratio"]} '
                f'reference={augmentation.get("ratio_reference", corpus_name)}'
            )


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
        requested_tokens=data.get('requested_tokens'),
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
    """Rebuild generated corpora from tracked auxiliary metadata.

    Manifest-backed configs regenerate selection ledgers from pinned source
    revisions. Legacy component-style configs replay existing selection ledgers.

    Args:
        config_path: Dataset preparation config file.
        output_dir: Optional output directory override.
        overwrite: Whether to replace generated train JSONL files.

    Returns:
        None.
    """
    loaded_config = load_config(config_path)
    if is_manifest_config(loaded_config):
        write_manifest_config_outputs(config_path, loaded_config, output_dir, overwrite)
        return

    resolved_config = resolve_config(
        loaded_config,
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
            os.path.join(paths['train'], f'{config_name}.jsonl'),
            os.path.join(paths['validation'], f'{config_name}.jsonl'),
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
    help='Development-only multiplier for FineWeb requested tokens. Keep 1.0 for real corpora.',
)
def main(
    config_path: str,
    output_dir: str | None,
    overwrite: bool,
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
        allow_incomplete_samples: Whether undersized FineWeb samples are allowed.
        preflight_only: Whether to inspect source metadata without writing outputs.
        max_kuatia_docs: Optional development cap for Kuatia documents.
        target_scale: Development multiplier for FineWeb token targets.

    Returns:
        None.
    """
    loaded_config = load_config(config_path)

    resolved_config = resolve_config(
        loaded_config,
        output_dir=output_dir,
        target_scale=target_scale,
        max_kuatia_docs=max_kuatia_docs,
        allow_incomplete_samples=allow_incomplete_samples,
    )

    if is_manifest_config(resolved_config):
        if preflight_only:
            preflight_manifest_config(resolved_config)
            return
        if target_scale != 1.0:
            raise click.ClickException(
                '--target-scale is not supported for config-backed data recipes.'
            )
        if max_kuatia_docs is not None:
            raise click.ClickException(
                '--max-kuatia-docs is not supported for config-backed data recipes.'
            )
        if allow_incomplete_samples:
            raise click.ClickException(
                '--allow-incomplete-samples is not supported for config-backed data recipes.'
            )
        write_manifest_config_outputs(config_path, resolved_config, output_dir, overwrite)
        return

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

        requested_tokens = round(kuatia_tokens * float(component_config['target_ratio_of_kuatia']) * scale)
        print(f'[requests] {component_name}={requested_tokens:,}', flush=True)
        source = sources[component_config['source']]
        component_stats[component_name] = write_sample_component(
            source,
            tokenizer_instance,
            os.path.join(paths['components'], f'{component_name}.jsonl'),
            selection_file_path(paths, component_name),
            requested_tokens=requested_tokens,
            seed=seed,
            shuffle_buffer_size=shuffle_buffer_size,
            allow_incomplete=allow_incomplete,
        )

    for component_name, component_config in components_config.items():
        if component_config.get('kind') != 'fixed_subsample':
            continue

        source_component = component_config['source_component']
        source_stats = component_stats[source_component]
        requested_tokens = round(kuatia_tokens * float(component_config['target_ratio_of_kuatia']) * scale)
        print(f'[requests] {component_name}={requested_tokens:,}', flush=True)
        component_stats[component_name] = write_fixed_subsample(
            source_stats,
            source_stats.path,
            source_stats.selection_path,
            os.path.join(paths['components'], f'{component_name}.jsonl'),
            selection_file_path(paths, component_name),
            requested_tokens=requested_tokens,
        )

    for config_name, corpus_config in corpora_config.items():
        components = [component_stats[component_name] for component_name in corpus_config['components']]
        corpus, heldout_corpus = write_corpus_splits(
            config_name,
            components,
            os.path.join(paths['train'], f'{config_name}.jsonl'),
            os.path.join(paths['validation'], f'{config_name}.jsonl'),
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
