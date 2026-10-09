"""Utilities for local lm-evaluation-harness tasks."""

from __future__ import annotations

import datasets
import hashlib
import re
import sacrebleu
import string

from typing import Any


MULTIWIKIQA_PUNCTUATION = string.punctuation + '¿¡“”‘’«»…'


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


def answer_index(value: Any) -> int:
    """Convert a multiple-choice answer label to a numeric index.

    Args:
        value: Raw answer label.

    Returns:
        Numeric answer index.

    Raises:
        ValueError: If the answer label is not A, B, C, or D.
    """
    answer = str(value).strip().upper()
    if answer not in ANSWER_TO_INDEX:
        raise ValueError(f'Unsupported answer label: {answer}')

    return ANSWER_TO_INDEX[answer]


def choice_texts(doc: dict[str, Any]) -> list[str]:
    """Read answer choices from common ARC-like schemas.

    Args:
        doc: Raw multiple-choice row.

    Returns:
        Ordered answer choice texts.
    """
    choices = doc.get('choices')
    if isinstance(choices, dict):
        labels = choices.get('label', [])
        texts = choices.get('text', [])
        if labels and texts:
            labeled = sorted(
                zip(labels, texts),
                key=lambda item: str(item[0]).strip().upper(),
            )
            return [clean_option(text) for _, text in labeled]

    return [
        clean_option(doc.get('answer_a')),
        clean_option(doc.get('answer_b')),
        clean_option(doc.get('answer_c')),
        clean_option(doc.get('answer_d')),
    ]


def process_spanish_arc_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Convert Spanish ARC rows to lm-eval MCQA rows.

    Args:
        dataset: Hugging Face Spanish ARC split.

    Returns:
        Dataset split with question, choices, and numeric gold columns.

    Raises:
        ValueError: If a row has an unsupported answer label.
    """

    def process_doc(doc: dict[str, Any]) -> dict[str, Any]:
        """Convert one Spanish ARC row.

        Args:
            doc: Raw dataset row.

        Returns:
            Processed multiple-choice row.

        Raises:
            ValueError: If the answer label is not A, B, C, or D.
        """
        answer = doc.get('answerkey', doc.get('answerKey', doc.get('answer')))

        return {
            'id': doc.get('id'),
            'question': str(doc['question']).strip(),
            'choices': choice_texts(doc),
            'gold': answer_index(answer),
        }

    return dataset.map(process_doc, remove_columns=dataset.column_names)


def deterministic_answer_order(
    key: str, answers: list[tuple[str, bool]]
) -> list[tuple[str, bool]]:
    """Order answer choices deterministically using a row-specific key.

    Args:
        key: Stable question identifier.
        answers: Answer text and correctness pairs.

    Returns:
        Deterministically shuffled answer pairs.
    """
    return [
        item
        for _, item in sorted(
            enumerate(answers),
            key=lambda indexed: hashlib.sha256(
                f'{key}:{indexed[0]}'.encode('utf-8')
            ).hexdigest(),
        )
    ]


def process_spanish_gpqa_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Convert Spanish GPQA Diamond rows to lm-eval MCQA rows.

    Args:
        dataset: Hugging Face GPQA multilingual Spanish split.

    Returns:
        Diamond-only dataset split with question, choices, and numeric gold columns.
    """

    def is_diamond_doc(doc: dict[str, Any]) -> bool:
        """Return whether a GPQA row belongs to the Diamond subset.

        Args:
            doc: Raw GPQA row.

        Returns:
            True when the row is marked as Diamond.
        """
        value = doc.get('is_diamond')
        if isinstance(value, str):
            return value.strip().lower() == 'true'

        return bool(value)

    def process_doc(doc: dict[str, Any]) -> dict[str, Any]:
        """Convert one Spanish GPQA row.

        Args:
            doc: Raw dataset row.

        Returns:
            Processed multiple-choice row.
        """
        incorrect_answers = doc.get('incorrect_answers', [])
        if not isinstance(incorrect_answers, list):
            incorrect_answers = [incorrect_answers]

        answers = [(clean_option(doc.get('correct_answer')), True)] + [
            (clean_option(answer), False)
            for answer in incorrect_answers
            if clean_option(answer)
        ]
        key = str(doc.get('original_id') or doc.get('question') or '')
        ordered_answers = deterministic_answer_order(key, answers)
        choices = [answer for answer, _ in ordered_answers]
        gold = next(
            index for index, (_, is_correct) in enumerate(ordered_answers) if is_correct
        )

        return {
            'original_id': doc.get('original_id'),
            'domain': doc.get('domain'),
            'subdomain': doc.get('subdomain'),
            'question': str(doc['question']).strip(),
            'choices': choices,
            'gold': gold,
        }

    return dataset.filter(is_diamond_doc).map(
        process_doc, remove_columns=dataset.column_names
    )


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


def _process_flores_plus_direction(
    dataset: Any, source_config: str, target_config: str
) -> Any:
    """Join aligned FLORES+ language configs by row id for one translation direction."""
    if len(dataset) == 0:
        return dataset

    split = str(dataset[0]["split"])
    source_dataset = datasets.load_dataset(
        "openlanguagedata/flores_plus", source_config, split=split
    )
    source_by_id = {str(row["id"]): str(row["text"]) for row in source_dataset}
    source_field = f"sentence_{source_config}"
    target_field = f"sentence_{target_config}"

    def add_source(row: dict[str, Any]) -> dict[str, str]:
        row[source_field] = source_by_id[str(row["id"])]
        row[target_field] = str(row["text"])
        return row

    return dataset.map(add_source)


def process_flores_plus_eng_to_grn(dataset: Any) -> Any:
    return _process_flores_plus_direction(dataset, "eng_Latn", "gug_Latn")


def process_flores_plus_grn_to_eng(dataset: Any) -> Any:
    return _process_flores_plus_direction(dataset, "gug_Latn", "eng_Latn")


def process_flores_plus_grn_to_spa(dataset: Any) -> Any:
    return _process_flores_plus_direction(dataset, "gug_Latn", "spa_Latn")


def process_flores_plus_spa_to_grn(dataset: Any) -> Any:
    return _process_flores_plus_direction(dataset, "spa_Latn", "gug_Latn")



def process_2m_belebele_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Convert 2M-Belebele rows to text-only lm-eval MCQA rows.

    Args:
        dataset: Hugging Face 2M-Belebele split.

    Returns:
        Dataset split with passage, question, answer choices, and numeric gold columns.

    Raises:
        ValueError: If a row has an unsupported answer number.
    """

    def process_doc(doc: dict[str, Any]) -> dict[str, Any]:
        """Convert one 2M-Belebele row.

        Args:
            doc: Raw dataset row.

        Returns:
            Processed multiple-choice row.

        Raises:
            ValueError: If the answer number is not 1, 2, 3, or 4.
        """
        answer = str(doc['correct_answer_num']).strip()
        if answer.endswith('.0'):
            answer = answer[:-2]

        if answer not in {'1', '2', '3', '4'}:
            raise ValueError(f'Unsupported answer number: {answer}')

        return {
            'link': doc.get('link'),
            'question_number': doc.get('question_number'),
            'flores_passage': str(doc['flores_passage']).strip(),
            'question': str(doc['question']).strip(),
            'choices': [
                clean_option(doc.get('mc_answer1')),
                clean_option(doc.get('mc_answer2')),
                clean_option(doc.get('mc_answer3')),
                clean_option(doc.get('mc_answer4')),
            ],
            'gold': int(answer) - 1,
        }

    return dataset.map(process_doc, remove_columns=dataset.column_names)


def answer_texts(doc: dict[str, Any]) -> list[str]:
    """Extract answer strings from a SQuAD-style document.

    Args:
        doc: Dataset row with an `answers` field.

    Returns:
        List of non-empty reference answer strings.
    """
    answers = doc.get('answers', {})
    if not isinstance(answers, dict):
        return []

    texts = answers.get('text', [])
    if isinstance(texts, str):
        texts = [texts]

    return [str(text).strip() for text in texts if str(text).strip()]


def doc_to_multiwikiqa_text(doc: dict[str, Any]) -> str:
    """Build a Guarani reading-comprehension prompt for MultiWikiQA.

    Args:
        doc: MultiWikiQA row.

    Returns:
        Prompt asking the model to answer from the provided context.
    """
    title = str(doc.get('title', '')).strip()
    context = str(doc.get('context', '')).strip()
    question = str(doc.get('question', '')).strip()

    return (
        'Eipuru ko jehaipyre embohovái hag̃ua porandu. '
        'Eme’ẽ mbohovái mbyky añónte.\n\n'
        f'Mba’e réra: {title}\n\n'
        f'Jehaipyre: {context}\n\n'
        f'Porandu: {question}\n\n'
        'Mbohovái:'
    )


def doc_to_multiwikiqa_target(doc: dict[str, Any]) -> str:
    """Return the first reference answer for display and sample logging.

    Args:
        doc: MultiWikiQA row.

    Returns:
        First reference answer, or an empty string if none exists.
    """
    answers = answer_texts(doc)
    return answers[0] if answers else ''


def normalize_qa_answer(value: str) -> str:
    """Normalize a QA answer before exact-match and F1 scoring.

    Args:
        value: Raw predicted or reference answer.

    Returns:
        Lowercased answer with punctuation removed and whitespace normalized.
    """
    lowered = value.lower()
    no_punctuation = ''.join(
        character for character in lowered if character not in MULTIWIKIQA_PUNCTUATION
    )
    return re.sub(r'\s+', ' ', no_punctuation).strip()


def clean_multiwikiqa_prediction(value: str) -> str:
    """Remove common answer prefixes from a generated QA response.

    Args:
        value: Raw model generation.

    Returns:
        Cleaned prediction string.
    """
    prediction = str(value).strip().split('\n', maxsplit=1)[0].strip()
    prefixes = ('mbohovái:', 'answer:', 'respuesta:')
    lowered = prediction.lower()

    for prefix in prefixes:
        if lowered.startswith(prefix):
            return prediction[len(prefix):].strip()

    return prediction


def exact_match_score(prediction: str, reference: str) -> float:
    """Compute normalized exact-match score for one reference.

    Args:
        prediction: Model answer.
        reference: Gold answer.

    Returns:
        1.0 when normalized strings match, otherwise 0.0.
    """
    return float(normalize_qa_answer(prediction) == normalize_qa_answer(reference))


def f1_score(prediction: str, reference: str) -> float:
    """Compute token-level F1 for one reference.

    Args:
        prediction: Model answer.
        reference: Gold answer.

    Returns:
        Token-level F1 score.
    """
    prediction_tokens = normalize_qa_answer(prediction).split()
    reference_tokens = normalize_qa_answer(reference).split()

    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)

    common = set(prediction_tokens) & set(reference_tokens)
    overlap = sum(
        min(prediction_tokens.count(token), reference_tokens.count(token))
        for token in common
    )
    if overlap == 0:
        return 0.0

    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def best_score(
    prediction: str, references: list[str], scorer: Any
) -> float:
    """Score a prediction against multiple references and keep the best score.

    Args:
        prediction: Model answer.
        references: Gold answers.
        scorer: Function that scores one prediction/reference pair.

    Returns:
        Best score across references, or 0.0 if there are no references.
    """
    if not references:
        return 0.0

    return max(scorer(prediction, reference) for reference in references)


def process_multiwikiqa_results(
    doc: dict[str, Any], results: list[str]
) -> dict[str, float]:
    """Score one generated MultiWikiQA answer.

    Args:
        doc: MultiWikiQA row.
        results: lm-eval generated responses.

    Returns:
        Exact-match and token-F1 scores for the row.
    """
    prediction = clean_multiwikiqa_prediction(results[0] if results else '')
    references = answer_texts(doc)

    return {
        'exact_match': best_score(prediction, references, exact_match_score),
        'f1': best_score(prediction, references, f1_score),
    }


def chrf_plus_plus(predictions: list[str], references: list[str]) -> tuple[str, str]:
    """Return one translation pair for chrF++ aggregation.

    Args:
        predictions: Generated translation strings from lm-eval.
        references: Reference translation strings from lm-eval.

    Returns:
        Reference and prediction pair.
    """
    prediction = predictions[0].strip() if predictions else ''
    reference = references[0].strip() if references else ''
    return reference, prediction


def aggregate_chrf_plus_plus(items: list[tuple[str, str]]) -> float:
    """Compute corpus chrF++ with sacrebleu.

    Args:
        items: Reference and prediction pairs.

    Returns:
        sacrebleu corpus chrF++ score.
    """
    references = [reference for reference, _ in items]
    predictions = [prediction for _, prediction in items]
    return sacrebleu.corpus_chrf(predictions, [references], word_order=2).score
