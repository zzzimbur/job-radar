"""Job Radar – гибридный RAG-поиск вакансий hh.ru под твой профиль.

    python main.py fetch "python разработчик"   # спарсить hh.ru в локальную базу
    python main.py search "LLM RAG боты"        # гибридный поиск + LLM-ранжирование
    python main.py search "..." --no-llm        # только BM25 + dense, без LLM
    python main.py stats                        # размер базы
"""

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("main")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PROFILE_FILE = Path("profile.md")


def cmd_fetch(args) -> int:
    from job_radar import hh
    from job_radar.store import Store

    store = Store()
    vacancies = hh.search(args.query, pages=args.pages)
    if args.descriptions:
        for i, v in enumerate(vacancies):
            v["description"] = hh.fetch_description(v["id"])
            if (i + 1) % 10 == 0:
                log.info("Описания: %d/%d", i + 1, len(vacancies))
    added = store.add(vacancies)
    print(f"Добавлено {added} новых вакансий (всего в базе: {len(store.vacancies)})")
    return 0


def cmd_search(args) -> int:
    from job_radar.store import Store
    from job_radar.hybrid import hybrid_search

    store = Store()
    if not store.vacancies:
        print("База пуста – сначала: python main.py fetch \"запрос\"")
        return 1

    results = hybrid_search(store, args.query, k=args.top)
    candidates = [v for v, _ in results]

    if not args.no_llm:
        profile = PROFILE_FILE.read_text(encoding="utf-8") if PROFILE_FILE.exists() else ""
        if not profile:
            print("⚠️  Нет profile.md – LLM ранжирует без профиля (скопируй profile.example.md)")
        from job_radar.rerank import rerank
        candidates = rerank(
            candidates, profile or "(профиль не задан)",
            groq_key=os.getenv("GROQ_API_KEY", ""),
            gemini_key=os.getenv("GEMINI_API_KEY", ""),
        )

    for i, v in enumerate(candidates, 1):
        score = f" [{v['llm_score']}]" if "llm_score" in v else ""
        remote = " · удалёнка" if v.get("remote") else ""
        salary = f" · {v['salary']}" if v.get("salary") else ""
        print(f"\n{i}.{score} {v['name']} – {v['employer']}{salary}{remote}")
        if v.get("llm_reason"):
            print(f"   💬 {v['llm_reason']}")
        print(f"   {v['url']}")
    print()
    return 0


def cmd_stats(_args) -> int:
    from job_radar.store import Store
    store = Store()
    with_desc = sum(1 for v in store.vacancies if v.get("description"))
    print(f"Вакансий в базе: {len(store.vacancies)} (с описаниями: {with_desc})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Job Radar")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_fetch = sub.add_parser("fetch", help="спарсить hh.ru в базу")
    p_fetch.add_argument("query")
    p_fetch.add_argument("--pages", type=int, default=2)
    p_fetch.add_argument("--descriptions", action="store_true",
                         help="дотянуть полные описания (медленно)")
    p_fetch.set_defaults(func=cmd_fetch)

    p_search = sub.add_parser("search", help="гибридный поиск по базе")
    p_search.add_argument("query")
    p_search.add_argument("--top", type=int, default=10)
    p_search.add_argument("--no-llm", action="store_true")
    p_search.set_defaults(func=cmd_search)

    p_stats = sub.add_parser("stats", help="размер базы")
    p_stats.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
