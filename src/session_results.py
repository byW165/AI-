"""Temporary results for one page session. No filesystem or database access."""
from dataclasses import asdict

from .models import Paper, Summary

MAX_SESSION_PAPERS = 200


class SessionResults:
    def __init__(self):
        self._records: dict[str, dict] = {}

    def clear(self) -> None:
        self._records.clear()

    def upsert(self, paper: Paper) -> bool:
        existing = self._records.get(paper.id)
        if existing and existing["fingerprint"] == paper.fingerprint:
            return False
        if not existing and len(self._records) >= MAX_SESSION_PAPERS:
            raise ValueError("当前会话最多展示 200 篇，请清空结果后重新检索。")
        self._records[paper.id] = {
            **asdict(paper), "fingerprint": paper.fingerprint,
            "core_contribution": None, "methods_and_datasets": None,
            "code_status": None, "status": "pending", "error": None,
        }
        return True

    def set_summary(self, paper: Paper, summary: Summary) -> None:
        row = self._records.get(paper.id)
        if row and row["fingerprint"] == paper.fingerprint:
            row.update(asdict(summary), status="success", error=None)

    def set_error(self, paper: Paper, error: str) -> None:
        row = self._records.get(paper.id)
        if row and row["fingerprint"] == paper.fingerprint:
            row.update(status="failed", error=error)

    def all(self) -> list[dict]:
        return sorted(self._records.values(), key=lambda row: (row["published"], row["id"]), reverse=True)


def paper_from_record(record: dict) -> Paper:
    return Paper(**{name: record[name] for name in Paper.__dataclass_fields__})


def filter_records(records: list[dict], code_status: str = "全部") -> list[dict]:
    return [row for row in records if code_status == "全部" or row["code_status"] == code_status]
