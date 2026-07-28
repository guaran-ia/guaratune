"""Utilities for local lm-evaluation-harness tasks."""

from __future__ import annotations

import datasets

from typing import Any


ANSWER_TO_INDEX = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
}


def clean_option(value: Any) -> str:
    """Normalize a multiple-choice option.

    Args:
        value: Raw option value from a JSONL record.

    Returns:
        Stripped option text.
    """
    return str(value).strip() if value is not None else ''


def process_global_mmlu_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Convert Guarani Global-MMLU-Lite JSONL rows to lm-eval MCQA rows.

    Args:
        dataset: Hugging Face dataset split loaded from local JSONL.

    Returns:
        Dataset split with `choices` and numeric `gold` columns.

    Raises:
        ValueError: If a row has an unsupported answer label.
    """

    def process_doc(doc: dict[str, Any]) -> dict[str, Any]:
        """Convert one Guarani Global-MMLU-Lite row.

        Args:
            doc: Raw dataset row.

        Returns:
            Processed multiple-choice row.

        Raises:
            ValueError: If the answer label is not A, B, C, or D.
        """
        answer = str(doc['answer']).strip().upper()
        if answer not in ANSWER_TO_INDEX:
            raise ValueError(f'Unsupported answer label: {answer}')

        return {
            'sample_id': doc.get('sample_id'),
            'subject': doc.get('subject'),
            'subject_category': doc.get('subject_category'),
            'question': str(doc['question']).strip(),
            'choices': [
                clean_option(doc.get('option_a')),
                clean_option(doc.get('option_b')),
                clean_option(doc.get('option_c')),
                clean_option(doc.get('option_d')),
            ],
            'gold': ANSWER_TO_INDEX[answer],
        }

    return dataset.map(process_doc)
