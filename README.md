# 🎯 Job Radar

> Гибридный RAG-поиск вакансий hh.ru: BM25 + векторный поиск + LLM-ранжирование под твой профиль

Парсит hh.ru без OAuth, складывает вакансии в локальную векторную базу и ищет гибридом из двух ретриверов с Reciprocal Rank Fusion. Финальный топ прогоняется через LLM, который оценивает каждую вакансию против твоего профиля – с причиной, почему стоит (или не стоит) откликаться.

## Пример

```
$ python main.py search "junior LLM RAG telegram боты" --top 5

1. [80] LLM Разработчик – True Engineering
   💬 вакансия соответствует желаемой позиции и стеку
   https://hh.ru/vacancy/...

2. [70] Разработчик систем интеграции с LLM – Кредит Европа Банк
   💬 соответствует стеку, но опыт на верхней границе
   https://hh.ru/vacancy/...

3. [0] Junior GO разработчик – ...
   💬 требует GO, не соответствует стеку кандидата
```

LLM честно ставит ноль вакансиям не по стеку, тимлидским позициям и требованиям 3+ лет – даже если лексически они в топе.

## Как работает

```
hh.ru (стриминг HTML) –> локальная база (JSON + npy)
                              │
        запрос –> BM25 (руками) ──┐
                                  ├–> RRF fusion –> LLM-ранжирование –> топ
        запрос –> dense (MiniLM) ─┘        под profile.md
```

**Три уровня поиска:**
1. **BM25 Okapi** – собственная реализация, ~50 строк, без ElasticSearch и библиотек
2. **Dense** – sentence-transformers (multilingual MiniLM), косинус по нормализованной матрице numpy
3. **RRF** – `score = Σ 1/(60 + rank)`: ранги вместо сырых скоров, потому что BM25 и косинус живут в разных шкалах

**LLM-ранжирование** – Groq LLaMA 3.3 70B (fallback Gemini 2.0 Flash) получает профиль кандидата и топ-10 гибрида, возвращает structured JSON со score и причиной.

**Парсер hh.ru** – данные вшиты в HTML как JSON (Next.js). Читаем потоком и обрываем скачивание, как только массив `vacancies` закрылся – первые ~200 KB вместо мегабайта.

## Запуск

```bash
pip install -r requirements.txt
cp profile.example.md profile.md   # свой профиль
cp .env.example .env               # GROQ_API_KEY или GEMINI_API_KEY

python main.py fetch "python разработчик"          # спарсить в базу
python main.py fetch "llm engineer" --descriptions # с полными описаниями
python main.py search "LLM RAG удалёнка"           # гибрид + LLM
python main.py search "..." --no-llm               # только ретриверы
python main.py stats
```

## Стек

Python 3.9+, httpx, numpy, sentence-transformers, Groq/Gemini API

## Лицензия

MIT

---

<p align="center"><a href="https://github.com/zzzimbur">@zzzimbur</a></p>
