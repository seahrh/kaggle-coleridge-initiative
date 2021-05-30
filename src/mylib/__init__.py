import re
import numpy as np
from typing import List

__all__ = ["jaccard", "clean_text", "fbeta"]


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
