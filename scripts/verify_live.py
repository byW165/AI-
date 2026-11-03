"""Real API smoke check; summaries are real calls only when --ai is supplied.

Run from the project root: python -m scripts.verify_live --days 180 --limit 3
"""
import argparse
import json

from src.ai_summary import summarize
from src.config import ConfigurationError, load_config
from src.fetch_papers import PROFILE_NAMES, fetch_papers
from src.http_client import ServiceError
from src.search_terms import SearchTermError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--profile", choices=PROFILE_NAMES, default="综合追踪")
    parser.add_argument("--keyword", default="")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--ai", action="store_true", help="对第一篇论文进行真实 AI 提取，仅打印结果，可能消耗 API 额度")
    args = parser.parse_args()
    try:
        result = fetch_papers(args.days, args.profile, args.limit, keyword=args.keyword, offset=args.offset)
    except (ServiceError, SearchTermError) as exc:
        print(json.dumps({"source": "failed", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps({"source": "passed", "count": len(result.papers), "total": result.total,
                      "truncated": result.truncated, "start": result.start.isoformat(), "end": result.end.isoformat(),
                      "papers": [{"id": p.id, "title": p.title, "published": p.published, "url": p.url} for p in result.papers]}, ensure_ascii=False))
    if args.ai and result.papers:
        paper = result.papers[0]
        try:
            summary = summarize(paper, load_config())
            print(json.dumps({"ai": "passed", "paper": paper.id, "summary": summary.__dict__, "original_abstract": paper.abstract}, ensure_ascii=False))
        except (ConfigurationError, ServiceError) as exc:
            print(json.dumps({"ai": "failed", "error": str(exc)}, ensure_ascii=False))
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
