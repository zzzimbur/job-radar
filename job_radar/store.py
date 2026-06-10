"""Хранилище: вакансии в JSON, эмбеддинги в .npy рядом.

Простой векторный стор на numpy – для тысяч документов
косинус по матрице быстрее и проще, чем поднимать векторную БД.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

DATA_DIR = Path("data")
VACANCIES_FILE = DATA_DIR / "vacancies.json"
EMBEDDINGS_FILE = DATA_DIR / "embeddings.npy"

_MODEL = None
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def _embedder():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        log.info("Загружаю %s ...", MODEL_NAME)
        _MODEL = SentenceTransformer(MODEL_NAME)
    return _MODEL


def doc_text(v: dict) -> str:
    return f"{v['name']}. {v['employer']}. {v.get('experience', '')}. {v.get('description', '')[:1500]}"


class Store:
    def __init__(self):
        DATA_DIR.mkdir(exist_ok=True)
        self.vacancies: list[dict] = []
        self.embeddings: np.ndarray | None = None
        self._load()

    def _load(self):
        if VACANCIES_FILE.exists():
            self.vacancies = json.loads(VACANCIES_FILE.read_text(encoding="utf-8"))
        if EMBEDDINGS_FILE.exists():
            self.embeddings = np.load(EMBEDDINGS_FILE)

    def add(self, new: list[dict]) -> int:
        """Добавляет вакансии, пропуская дубли по id. Возвращает число добавленных."""
        seen = {v["id"] for v in self.vacancies}
        fresh = [v for v in new if v["id"] not in seen]
        if not fresh:
            return 0

        texts = [doc_text(v) for v in fresh]
        embs = _embedder().encode(texts, show_progress_bar=False, normalize_embeddings=True)

        self.vacancies.extend(fresh)
        self.embeddings = (
            np.vstack([self.embeddings, embs]) if self.embeddings is not None else np.asarray(embs)
        )

        VACANCIES_FILE.write_text(
            json.dumps(self.vacancies, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        np.save(EMBEDDINGS_FILE, self.embeddings)
        return len(fresh)

    def dense_top(self, query: str, k: int = 20) -> list[tuple[int, float]]:
        """Косинусная близость запроса ко всем вакансиям."""
        if self.embeddings is None or not len(self.vacancies):
            return []
        q = _embedder().encode([query], normalize_embeddings=True)[0]
        sims = self.embeddings @ q  # эмбеддинги нормализованы – dot = cosine
        ranked = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in ranked]
