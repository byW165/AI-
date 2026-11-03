"""arXiv Atom retrieval with UTC date boundaries and canonical-ID deduplication."""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import re
import threading
import time
from urllib.parse import urlsplit
import xml.etree.ElementTree as ET

import requests

from .http_client import RetryableError, request_once, with_retries
from .models import Paper, iso_time, parse_time, utc_now
from .search_terms import keyword_query, SearchTermError

API_URL = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom", "o": "http://a9.com/-/spec/opensearch/1.1/"}
QUERIES = {
    "加速度预测（核心）": '(abs:"vehicle" AND (abs:"acceleration prediction" OR abs:"acceleration forecasting" OR (abs:"acceleration" AND abs:"car-following" AND abs:"prediction")))',
    "车辆交互图与时序模型": '((abs:"vehicle" OR abs:"car-following") AND (abs:"graph neural" OR abs:"graph attention" OR abs:"graph convolutional") AND (abs:"trajectory prediction" OR abs:"trajectory forecasting" OR abs:"motion prediction" OR abs:"speed prediction" OR abs:"acceleration prediction" OR abs:"car-following"))',
    "车速、跟驰与轨迹预测": '(abs:"vehicle speed prediction" OR abs:"vehicle speed forecasting" OR abs:"car-following prediction" OR abs:"vehicle trajectory prediction")',
    "混动车预测性能量管理": '((abs:"hybrid electric vehicle" OR abs:"hybrid vehicle") AND abs:"energy management" AND (abs:"prediction" OR abs:"predictive"))',
}
ALL_ARXIV = "不限领域（全部 arXiv）"
PROFILE_NAMES = ["综合追踪", *QUERIES, ALL_ARXIV]
_rate_lock = threading.Lock()
_last_request = float("-inf")


def wait_for_arxiv() -> None:
    # Shared across Streamlit sessions in this process; includes retry requests.
    global _last_request
    with _rate_lock:
        remaining = 3.1 - (time.monotonic() - _last_request)
        if remaining > 0:
            time.sleep(remaining)
        _last_request = time.monotonic()


@dataclass(frozen=True)
class FetchResult:
    papers: list[Paper]
    total: int
    truncated: bool
    start: datetime
    end: datetime
    query: str
    offset: int = 0
    next_start: int = 0


def build_query(profile: str, start: datetime, end: datetime, keyword: str = "") -> str:
    if profile not in PROFILE_NAMES:
        raise ValueError("未知研究范围")
    keywords = keyword_query(keyword)
    if profile == ALL_ARXIV:
        if not keywords:
            raise SearchTermError("不限领域检索时请输入关键词，例如“图神经网络”。")
        terms = keywords
    elif profile == "综合追踪" and keywords:
        # Custom keywords use a broad road-vehicle prediction scope rather than
        # intersecting every narrow preset phrase and silently losing recall.
        terms = f'((abs:"vehicle" OR abs:"car-following" OR abs:"autonomous driving" OR abs:"traffic") AND (abs:"prediction" OR abs:"forecasting")) AND ({keywords})'
    else:
        terms = " OR ".join(f"({q})" for q in QUERIES.values()) if profile == "综合追踪" else QUERIES[profile]
        if keywords:
            terms = f"({terms}) AND ({keywords})"
    # Exclude title topics that real-query review found to be background-only
    # matches. This favors precision and may miss joint tasks.
    exclusions = '(ti:"vessel" OR ti:"pedestrian" OR ti:"UAV" OR ti:"EEG")'
    scope = f"({terms})" if profile == ALL_ARXIV else f"(({terms}) ANDNOT {exclusions})"
    return f"{scope} AND submittedDate:[{start:%Y%m%d%H%M} TO {end:%Y%m%d%H%M}]"


def canonical_id(value: str) -> str:
    path = urlsplit(value).path
    identifier = path.split("/abs/", 1)[-1] if "/abs/" in path else value
    identifier = re.sub(r"v\d+$", "", identifier.strip())
    if not re.fullmatch(r"(?:\d{4}\.\d{4,5}|[a-zA-Z][\w.-]*(?:\.[A-Z]{2})?/\d{7})", identifier):
        raise ValueError("论文 ID 格式无效")
    return identifier


def parse_feed(content: bytes, start: datetime, end: datetime, limit: int) -> tuple[list[Paper], int, bool]:
    try:
        root = ET.fromstring(content)
        if root.tag != f"{{{NS['a']}}}feed":
            raise ValueError("非 Atom feed")
        total_text = root.findtext("o:totalResults", namespaces=NS)
        if total_text is None:
            raise ValueError("缺少总数")
        total = int(total_text)
        if total < 0:
            raise ValueError("总数无效")
        papers: dict[str, Paper] = {}
        for entry in root.findall("a:entry", NS):
            def read(tag):
                return " ".join((entry.findtext(f"a:{tag}", namespaces=NS) or "").split())
            identifier = canonical_id(read("id"))
            published = parse_time(read("published"))
            updated = parse_time(read("updated"))
            if not start <= published <= end:
                continue
            title, abstract = read("title"), read("summary")
            authors = [" ".join((a.findtext("a:name", namespaces=NS) or "").split()) for a in entry.findall("a:author", NS)]
            if not title or not abstract or not any(authors):
                raise ValueError("论文元数据缺失")
            paper = Paper(identifier, title, [a for a in authors if a], iso_time(published), iso_time(updated), abstract, f"https://arxiv.org/abs/{identifier}")
            if identifier not in papers or updated > parse_time(papers[identifier].updated):
                papers[identifier] = paper
        ordered = sorted(papers.values(), key=lambda p: p.published, reverse=True)
        return ordered[:limit], total, total > limit
    except (ET.ParseError, ValueError, TypeError, OverflowError):
        raise RetryableError("arXiv 响应格式或论文元数据无效，无法当作正常空结果") from None


def fetch_papers(days: int = 30, profile: str = "综合追踪", limit: int = 20, *, end: datetime | None = None,
                 keyword: str = "", offset: int = 0, session=None, sleep=time.sleep, rate_limit=wait_for_arxiv) -> FetchResult:
    if not 1 <= days <= 3650 or not 1 <= limit <= 50:
        raise ValueError("天数应为 1–3650，篇数应为 1–50")
    if offset < 0:
        raise ValueError("分页位置不能为负数")
    end = end or utc_now()
    if end.tzinfo is None:
        raise ValueError("检索时间必须包含时区")
    end = end.astimezone(timezone.utc)
    start = end - timedelta(days=days)
    query = build_query(profile, start, end, keyword)
    def operation(attempt):
        rate_limit()
        response = request_once(client, "GET", API_URL, service="arXiv", params={
            "search_query": query, "start": offset, "max_results": limit,
            "sortBy": "submittedDate", "sortOrder": "descending",
        }, headers={"User-Agent": "VehicleLiteratureTracker/1.0 (academic coursework)"}, timeout=(10, 35))
        return parse_feed(response.content, start, end, limit)
    client = session or requests.Session()
    try:
        papers, total, truncated = with_retries(operation, sleep=sleep)
        return FetchResult(papers, total, total > offset + limit, start, end, query, offset, offset + limit)
    finally:
        if session is None:
            client.close()
