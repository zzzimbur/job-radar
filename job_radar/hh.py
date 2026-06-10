"""Парсер вакансий hh.ru без OAuth.

Данные вшиты в HTML как JSON (Next.js). Читаем страницу потоком
и останавливаемся, как только массив vacancies полностью пришёл –
это первые ~150–300 KB вместо целого мегабайта.
"""

from __future__ import annotations

import json
import logging
import re

import httpx

log = logging.getLogger(__name__)

SEARCH_URL = "https://hh.ru/search/vacancy"
VACANCY_URL = "https://hh.ru/vacancy/{}"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept-Language": "ru-RU,ru;q=0.9",
}

EXP_MAP = {
    "noExperience": "без опыта",
    "between1And3": "1–3 года",
    "between3And6": "3–6 лет",
    "moreThan6": "6+ лет",
}


def _extract_array(buf: str, key: str = '"vacancies":[') -> str | None:
    """Возвращает полный JSON-массив после key или None, если он ещё не докачан."""
    idx = buf.find(key)
    if idx == -1:
        return None
    start = idx + len(key) - 1  # позиция '['
    depth = 0
    for i, ch in enumerate(buf[start:], start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return buf[start:i + 1]
    return None


def search(query: str, area: int = 113, per_page: int = 50, pages: int = 2) -> list[dict]:
    """Поиск вакансий. area=113 – вся Россия."""
    out: list[dict] = []
    timeout = httpx.Timeout(connect=15, read=90, write=15, pool=15)
    with httpx.Client(headers=HEADERS, timeout=timeout, follow_redirects=True) as client:
        for page in range(pages):
            params = {"text": query, "area": area, "per_page": per_page,
                      "page": page, "search_period": 30}
            raw_array = None
            with client.stream("GET", SEARCH_URL, params=params) as r:
                r.raise_for_status()
                buf = ""
                for chunk in r.iter_text(chunk_size=32768):
                    buf += chunk
                    raw_array = _extract_array(buf)
                    if raw_array is not None:
                        break  # массив целый – дальше не качаем
            if not raw_array:
                log.warning("Страница %d: массив vacancies не найден", page)
                continue
            try:
                items = json.loads(raw_array)
            except json.JSONDecodeError as e:
                log.warning("Страница %d: битый JSON (%s)", page, e)
                continue
            out.extend(_normalize(v) for v in items)
            if len(items) < per_page:
                break  # вакансий меньше, чем влезает на страницу – дальше пусто
    log.info("hh.ru: «%s» – %d вакансий", query, len(out))
    return out


def _normalize(v: dict) -> dict:
    comp = v.get("compensation") or {}
    salary = ""
    lo, hi = comp.get("from"), comp.get("to")
    if lo and hi:
        salary = f"{lo}–{hi}"
    elif lo:
        salary = f"от {lo}"
    elif hi:
        salary = f"до {hi}"

    return {
        "id": str(v.get("vacancyId", "")),
        "name": v.get("name", ""),
        "employer": (v.get("company") or {}).get("visibleName", ""),
        "area": (v.get("area") or {}).get("name", ""),
        "salary": salary,
        "remote": v.get("@workSchedule") == "remote",
        "experience": EXP_MAP.get(v.get("workExperience", ""), ""),
        "url": VACANCY_URL.format(v.get("vacancyId", "")),
        "description": "",
    }


def fetch_description(vacancy_id: str) -> str:
    """Текст описания со страницы вакансии."""
    with httpx.Client(headers=HEADERS, timeout=30, follow_redirects=True) as client:
        r = client.get(VACANCY_URL.format(vacancy_id))
        if r.status_code != 200:
            return ""
    m = re.search(r'data-qa="vacancy-description"[^>]*>(.*?)</div>', r.text, re.DOTALL)
    if not m:
        return ""
    text = re.sub(r"<[^>]+>", " ", m.group(1))
    return re.sub(r"\s+", " ", text).strip()
