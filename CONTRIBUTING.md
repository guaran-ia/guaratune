# Contributing to GuaraTune

Thank you for improving GuaraTune. Contributions to code, configurations,
documentation, tests, and reproducibility are welcome.

## Before You Start

- Read the [README](README.md) and the configuration documentation relevant to
  your change.
- Discuss substantial changes in an issue before implementation.
- Do not commit credentials, Hugging Face tokens, Weights & Biases keys, model
  weights, downloaded corpora, generated checkpoints, or proprietary data.
- Respect the licenses and access conditions documented in
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Development Setup

Use Python 3.12. Install the project and development dependencies as described
in the [installation instructions](README.md#installation):

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

Run the required checks before opening a pull request:

```bash
pytest
ruff check .
```

Use `python -m src.check_environment --require-cuda` before GPU-backed pipeline
runs. Unit tests and configuration validation do not require a GPU.

## Pull Requests

- Keep each pull request focused on one change.
- Explain the motivation, behavior change, and validation performed.
- Update tests and documentation with behavior or configuration changes.
- Regenerate dependency locks with `python scripts/compile_requirements.py` when
  changing a `.in` dependency file; do not edit generated lock files directly.
- Preserve immutable commit pins for upstream frameworks and training datasets.
- Report evaluation results with the model, data configuration, method, seed,
  and evaluation profile needed to reproduce them.

## Style

Follow the surrounding code and configuration style. Run Ruff rather than
formatting unrelated files, keep line lengths within the configured limit, and
write configuration changes so they can be validated by the test suite.
