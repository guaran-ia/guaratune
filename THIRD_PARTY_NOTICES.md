# Third-Party Notices

This repository's source code is licensed under GPL-3.0-only; see
[`LICENSE`](LICENSE). That license does not apply to third-party models, data,
software, or generated artifacts. This repository does not include or
redistribute downloaded model weights or remote datasets. Obtain each artifact
directly from its provider and comply with its then-current terms, licenses,
attribution requirements, and access restrictions.

The entries below cover artifacts directly named by the tracked training,
evaluation, and dependency configuration as of 2026-09-29. A "not pinned"
revision means the configuration asks the upstream service for its default
revision; the resulting artifact can change after this notice is published.

## Models

| Artifact | Configuration | Revision | Terms and attribution | Redistribution |
| --- | --- | --- | --- | --- |
| Google Gemma 4 E2B | `google/gemma-4-E2B` | `main` (not pinned) | [Gemma terms](https://ai.google.dev/gemma/terms). Google / Gemma attribution and any access requirements apply. | Not included; do not redistribute from this repository. |
| Google Gemma 4 E4B | `google/gemma-4-E4B` | `main` (not pinned) | [Gemma terms](https://ai.google.dev/gemma/terms). Google / Gemma attribution and any access requirements apply. | Not included; do not redistribute from this repository. |
| Google Gemma 4 12B | `google/gemma-4-12B` | `main` (not pinned) | [Gemma terms](https://ai.google.dev/gemma/terms). Google / Gemma attribution and any access requirements apply. | Not included; do not redistribute from this repository. |

## Training Data

| Artifact | Pinned revision | License or terms | Attribution | Redistribution |
| --- | --- | --- | --- | --- |
| [Kuatia](https://huggingface.co/datasets/guaran-ia/kuatia) (`guaran-ia/kuatia`, `compiled_full`) | [`5045d1bd67d2ea6d40d9c2a30d109ddc0a508a94`](https://huggingface.co/datasets/guaran-ia/kuatia/tree/5045d1bd67d2ea6d40d9c2a30d109ddc0a508a94) | CC-BY-4.0 | Guaran-IA / Kuatia. | Not included or redistributed. |
| [FineWeb-Edu Translated](https://huggingface.co/datasets/Helsinki-NLP/fineweb-edu-translated) (`spa`) | [`d82b075a4a8152727b232876aade2ade003e12b9`](https://huggingface.co/datasets/Helsinki-NLP/fineweb-edu-translated/tree/d82b075a4a8152727b232876aade2ade003e12b9) | [ODC-BY 1.0](https://opendatacommons.org/licenses/by/1-0/) | Helsinki-NLP, FineWeb-Edu Translated. | Not included or redistributed. |
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | [`87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/tree/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9) | [ODC-BY 1.0](https://opendatacommons.org/licenses/by/1-0/) | Hugging Face FineWeb team, FineWeb-Edu. | Not included or redistributed. |

## Evaluation Data

Remote evaluation data is fetched by `lm_eval` only when a suite is run. The
configurations do not pin dataset revisions; preserve the downloaded snapshot
and upstream card for a reproducible result.

| Artifact | Configured use | License or terms | Attribution | Redistribution |
| --- | --- | --- | --- | --- |
| [Global-MMLU-Lite](https://huggingface.co/datasets/CohereLabs/Global-MMLU-Lite) | `CohereLabs/Global-MMLU-Lite`, English and Spanish | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Cohere Labs, Global-MMLU-Lite. | Not included or redistributed. |
| Local Global-MMLU-Lite Guarani translation | `data/evaluation/gmlgnt.jsonl`, Guarani | CC-BY-4.0 | Cohere Labs, Global-MMLU-Lite; Guaran-IA translation contributors. | Tracked only for evaluation; do not distribute it separately without verifying provenance. |
| [2M-Belebele](https://huggingface.co/datasets/facebook/2M-Belebele) | `facebook/2M-Belebele`, English, Guarani, Spanish | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | Meta / Belebele authors. | Not included or redistributed. |
| [FLORES-200](https://huggingface.co/datasets/facebook/flores) | `facebook/flores`, English, Guarani, Spanish translation directions | Upstream dataset card is access-restricted at the time of this notice. Consult its card and the original [FLORES paper](https://aclanthology.org/2022.tacl-1.30/) for applicable terms and attribution. | Meta AI / FLORES-200 authors. | Not included or redistributed. |
| [Multi-Wiki-QA](https://huggingface.co/datasets/alexandrainst/multi-wiki-qa) | `alexandrainst/multi-wiki-qa`, English, Guarani, Spanish | Consult the upstream dataset card for the applicable license and source-data terms. | Alexandra Institute, Multi-Wiki-QA authors. | Not included or redistributed. |
| [ARC_es](https://huggingface.co/datasets/BSC-LT/arc_es) | `BSC-LT/arc_es`, Easy and Challenge Spanish validation splits | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | Barcelona Supercomputing Center; Peter Clark et al., ARC. | Not included or redistributed. |
| [GPQA Multilingual](https://huggingface.co/datasets/ellamind/gpqa-multilingual) | `ellamind/gpqa-multilingual`, Spanish | Upstream dataset card is access-restricted at the time of this notice. Consult its card and the original [GPQA paper](https://arxiv.org/abs/2311.12022) for applicable terms and attribution. | EllaMind, GPQA authors. | Not included or redistributed. |
| Coreguapa | `data/evaluation/coreguapa_identified_all.jsonl`, Guarani perplexity | Proprietary local artifact; no license is granted by this repository. | Coreguapa rights holders. | Not tracked or redistributed; evaluation is skipped when unavailable. |

The configured `lm_eval` English, Spanish, and instruction suites also use
built-in tasks from the pinned harness revision: `arc_easy`, `arc_challenge`,
`piqa`, `hellaswag`, `winogrande`, `xnli_en`, `xstorycloze_en`,
`mgsm_direct_en`, `bbh`, `gpqa_diamond_zeroshot`, `truthfulqa_mc1`,
`humaneval`, `global_piqa_nonparallel_cloze_spa_latn_spai`, `hellaswag_es`,
`copa_es`, `xstorycloze_es`, `truthfulqa_es_mc1`, `xnli_es`,
`mgsm_direct_es_spanish_bench`, `ifeval`, and `ifeval_es`. Their data sources
and licenses are specified by the task implementations and upstream dataset
cards in the [pinned lm-evaluation-harness source](https://github.com/EleutherAI/lm-evaluation-harness/tree/f4d4b3de3ee6741a7151a9fe74945ee515262f4c). They are not included or redistributed here.

## Software

| Component | Version or revision | License | Attribution | Redistribution |
| --- | --- | --- | --- | --- |
| [LlamaFactory](https://github.com/hiyouga/LlamaFactory) | [`ef2d8f9da66ef13378dc8068dabc99cfe89a9736`](https://github.com/hiyouga/LlamaFactory/commit/ef2d8f9da66ef13378dc8068dabc99cfe89a9736) | [Apache-2.0](https://github.com/hiyouga/LlamaFactory/blob/ef2d8f9da66ef13378dc8068dabc99cfe89a9736/LICENSE) | LlamaFactory contributors. | Installed from upstream; retain upstream notices if redistributing it. |
| [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | [`f4d4b3de3ee6741a7151a9fe74945ee515262f4c`](https://github.com/EleutherAI/lm-evaluation-harness/commit/f4d4b3de3ee6741a7151a9fe74945ee515262f4c) | See [upstream license and notices](https://github.com/EleutherAI/lm-evaluation-harness/tree/f4d4b3de3ee6741a7151a9fe74945ee515262f4c). | EleutherAI and lm-evaluation-harness contributors. | Installed from upstream; retain upstream notices if redistributing it. |
| spaCy models `ca_core_news_sm`, `es_core_news_sm`, `xx_sent_ud_sm` | `3.8.0` | [MIT](https://github.com/explosion/spacy-models/blob/master/LICENSE) | Explosion and each model's listed data sources. | Installed from upstream; retain model metadata and source-data notices. |

Other Python packages are resolved in the generated `requirements*.txt` lock
files. Their own distribution metadata and licenses govern them; this notice is
not a substitute for a complete dependency license inventory in a binary or
container distribution.
