"""LLM-ранжирование топа гибридного поиска под профиль кандидата."""

import json
import logging

import httpx

log = logging.getLogger(__name__)

PROMPT = (
    "Ты – карьерный ассистент. Вот профиль кандидата:\n\n{profile}\n\n"
    "Вот вакансии (id, название, компания, опыт, зарплата, описание):\n\n{vacancies}\n\n"
    "Оцени каждую вакансию от 0 до 100 – насколько кандидату реально стоит откликаться.\n"
    "Учитывай: совпадение стека, требуемый опыт против реального, удалёнка.\n"
    "Жёстко занижай оценку, если требуется опыт сильно больше, чем у кандидата.\n"
    "Ответь СТРОГО JSON-массивом:\n"
    '[{{"id": "...", "score": 85, "reason": "одно предложение почему"}}]'
)


def _ask_groq(api_key: str, prompt: str) -> str:
    r = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 1500,
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _ask_gemini(api_key: str, prompt: str) -> str:
    r = httpx.post(
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.0-flash:generateContent?key={api_key}",
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1500},
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


def _parse_json(raw: str) -> list[dict]:
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"Нет JSON в ответе LLM: {raw[:200]}")
    return json.loads(raw[start:end + 1])


def rerank(candidates: list[dict], profile: str,
           groq_key: str = "", gemini_key: str = "") -> list[dict]:
    """Возвращает вакансии с полями llm_score и llm_reason, отсортированные по score."""
    lines = []
    for v in candidates:
        lines.append(
            f"id={v['id']} | {v['name']} | {v['employer']} | "
            f"опыт: {v.get('experience') or '?'} | зп: {v.get('salary') or '?'} | "
            f"{'удалёнка' if v.get('remote') else 'офис/гибрид'}\n"
            f"  {v.get('description', '')[:400]}"
        )
    prompt = PROMPT.format(profile=profile, vacancies="\n".join(lines))

    raw = None
    if groq_key:
        try:
            raw = _ask_groq(groq_key, prompt)
        except Exception as exc:
            log.warning("Groq не ответил (%s), пробую Gemini", exc)
    if raw is None and gemini_key:
        raw = _ask_gemini(gemini_key, prompt)
    if raw is None:
        raise RuntimeError("Ни один LLM-провайдер не доступен")

    scores = {item["id"]: item for item in _parse_json(raw)}
    for v in candidates:
        item = scores.get(v["id"], {})
        v["llm_score"] = item.get("score", 0)
        v["llm_reason"] = item.get("reason", "")
    return sorted(candidates, key=lambda v: v["llm_score"], reverse=True)
