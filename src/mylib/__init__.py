import re

__all__ = ["jaccard", "clean_text"]


def jaccard(str1: str, str2: str) -> float:
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def clean_text(txt: str) -> str:
    return re.sub("[^A-Za-z0-9]+", " ", str(txt).lower())
