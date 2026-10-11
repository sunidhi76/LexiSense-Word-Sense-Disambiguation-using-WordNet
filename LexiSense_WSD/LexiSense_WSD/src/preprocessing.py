import re
from typing import List
from nltk.corpus import stopwords


def normalize_tokens(text: str, remove_stopwords: bool = True) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9_\-]+", text.lower())
    tokens = [re.sub(r"[^a-z0-9_-]", "", t) for t in tokens]
    tokens = [t for t in tokens if t]
    if remove_stopwords:
        try:
            sw = set(stopwords.words("english"))
            tokens = [t for t in tokens if t not in sw]
        except LookupError:
            pass
    return tokens


def context_tokens(sentence: str, target: str) -> List[str]:
    target = target.lower()
    return [t for t in normalize_tokens(sentence) if t != target]
