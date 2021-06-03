from collections import deque
import json
import re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from typing import List, Deque, Iterable, Callable
from scml import nlp as snlp

__all__ = ["jaccard", "clean_text", "fbeta", "qa_predict"]


def jaccard(str1: str, str2: str) -> float:
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def clean_text(txt: str) -> str:
    return re.sub("[^A-Za-z0-9]+", " ", str(txt).lower())


def fbeta(
    y_true: List[str], y_pred: List[str], beta: float = 0.5, sep: str = "|"
) -> float:
    """Compute the Jaccard-based micro FBeta score.

    References
    - https://www.kaggle.com/c/coleridgeinitiative-show-us-the-data/overview/evaluation
    - https://www.kaggle.com/c/coleridgeinitiative-show-us-the-data/discussion/230091
    """
    tp: float = 0  # true positive
    fp: int = 0  # false positive
    fn: float = 0  # false negative
    for truth_str, pred_str in zip(y_true, y_pred):
        preds = pred_str.split(sep)
        preds.sort()
        truths = truth_str.split(sep)
        truths.sort()
        matched = set()
        for t in truths:
            scores: List[float] = []
            for p in preds:
                scores.append(jaccard(t, p))
            i = int(np.argmax(scores))
            if scores[i] >= 0.5:
                matched.add(i)
                tp += 1
            else:
                fn += 1
        fp += len(preds) - len(matched)
    tp *= 1 + beta ** 2
    fn *= beta ** 2
    return tp / (tp + fp + fn)


def _pack_sentences(
    tokenizer: AutoTokenizer, sentences: Deque[str], question: str, max_tokens: int
) -> torch.Tensor:
    inputs = None
    prev = None
    _len = 0
    tmp = []
    while len(sentences) != 0 and _len < max_tokens:
        prev = inputs
        tmp.append(sentences[0])
        passage = " ".join(tmp)
        inputs = tokenizer.encode_plus(
            question,
            passage,
            truncation="only_second",
            max_length=max_tokens,
            add_special_tokens=True,
            return_tensors="pt",
        )
        _len = len(inputs["input_ids"][0])
        if _len < max_tokens:
            sentences.popleft()
        # print(f"inputs={inputs}")
        # print(f"_len={_len}")
    if _len >= max_tokens and prev is not None:
        inputs = prev
    return inputs


def qa_predict(
    data_dir: str,
    model: AutoModelForQuestionAnswering,
    tokenizer: AutoTokenizer,
    questions: Iterable[str],
    n_window: int,
    max_length: int = 1_000_000,
    max_tokens: int = 512,
    verbose: bool = False,
) -> Callable:
    def fn(row) -> str:
        rid = row["Id"]
        tmp = []
        with open(f"{data_dir}/{rid}.json") as in_file:
            sections = json.load(in_file)
            for section in sections:
                tmp.append(section["text"])
        text = " ".join(tmp).strip()
        if len(text) == 0:
            print(f"len(text)=0, Id={rid}")
            return ""
        if len(text) > max_length:
            text = text[:max_length]
        sentences: Deque[str] = deque(snlp.sentences(text))
        if len(sentences) == 0:
            print(f"len(sentences)=0, Id={rid}")
            return ""
        res = set()
        for question in questions:
            for _ in range(n_window):
                if len(sentences) == 0:
                    break
                inputs = _pack_sentences(
                    tokenizer=tokenizer,
                    sentences=sentences,
                    question=question,
                    max_tokens=max_tokens,
                )
                input_ids = inputs["input_ids"].tolist()[0]
                sep_index = input_ids.index(tokenizer.sep_token_id)
                answer_start_scores, answer_end_scores = model(**inputs).values()
                if verbose:
                    print(
                        f"answer_start_scores.shape={answer_start_scores.shape}, "
                        "answer_end_scores.shape={answer_end_scores.shape}"
                    )
                ai = torch.argmax(answer_start_scores)
                aj = torch.argmax(answer_end_scores) + 1
                if verbose:
                    print(f"ai={ai}, aj={aj}")
                if ai <= sep_index:
                    continue
                a = tokenizer.convert_tokens_to_string(
                    tokenizer.convert_ids_to_tokens(input_ids[ai:aj])
                )
                a = clean_text(a)
                if len(a) < 4 or len(a) > 150:
                    continue
                n_digits = snlp.count_digit(a)
                if n_digits > 4 or n_digits / len(a) > 0.2:
                    continue
                res.add(a)
        return "|".join(res)

    return fn
