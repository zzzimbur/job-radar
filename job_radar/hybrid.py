"""Гибридный поиск: BM25 + dense с Reciprocal Rank Fusion.

RRF(d) = Σ 1 / (k + rank_i(d)) по каждому источнику рангов.
Устойчивее, чем взвешивание сырых скоров: BM25 и косинус
живут в разных шкалах, а ранги – в одной.
"""

from .bm25 import BM25
from .store import Store, doc_text

RRF_K = 60


def hybrid_search(store: Store, query: str, k: int = 10) -> list[tuple[dict, float]]:
    if not store.vacancies:
        return []

    bm25 = BM25([doc_text(v) for v in store.vacancies])
    sparse = bm25.top(query, k=30)
    dense = store.dense_top(query, k=30)

    rrf: dict[int, float] = {}
    for rank, (idx, _) in enumerate(sparse):
        rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
    for rank, (idx, _) in enumerate(dense):
        rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)

    ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:k]
    return [(store.vacancies[i], score) for i, score in ranked]
