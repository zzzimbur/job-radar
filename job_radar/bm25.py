"""BM25 Okapi, реализованный руками – без ElasticSearch и сторонних библиотек.

Формула: score(q, d) = Σ IDF(t) · (tf·(k1+1)) / (tf + k1·(1 − b + b·|d|/avgdl))
"""

import math
import re
from collections import Counter

_WORD_RE = re.compile(r"[a-zа-яё0-9+#]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.corpus = [tokenize(d) for d in docs]
        self.doc_lens = [len(d) for d in self.corpus]
        self.avgdl = sum(self.doc_lens) / len(self.doc_lens) if self.corpus else 0.0
        self.doc_freqs = [Counter(d) for d in self.corpus]

        # document frequency каждого терма
        df: Counter = Counter()
        for doc in self.corpus:
            df.update(set(doc))
        n = len(self.corpus)
        # IDF с поправкой +1, чтобы не уходить в минус на частых термах
        self.idf = {t: math.log((n - f + 0.5) / (f + 0.5) + 1) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        q_tokens = tokenize(query)
        out = []
        for freqs, dl in zip(self.doc_freqs, self.doc_lens):
            s = 0.0
            for t in q_tokens:
                if t not in freqs:
                    continue
                tf = freqs[t]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                s += self.idf.get(t, 0.0) * tf * (self.k1 + 1) / denom
            out.append(s)
        return out

    def top(self, query: str, k: int = 20) -> list[tuple[int, float]]:
        scores = self.scores(query)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(i, s) for i, s in ranked[:k] if s > 0]
