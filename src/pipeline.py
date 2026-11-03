"""Explicit actions only. Searching/filtering never invokes this module."""
from dataclasses import dataclass, field
from typing import Callable

from .ai_summary import summarize
from .config import AIConfig, ConfigurationError
from .http_client import PermanentServiceError, TemporaryServiceError, ServiceError
from .session_results import SessionResults, paper_from_record


@dataclass
class ProcessingReport:
    success: int = 0
    failed: int = 0
    skipped: int = 0
    remaining: int = 0
    message: str = ""
    errors: list[str] = field(default_factory=list)


def process_records(records: list[dict], store: SessionResults, config: AIConfig,
                    *, extractor=None, on_progress: Callable | None = None) -> ProcessingReport:
    extractor = extractor or summarize
    pending = [r for r in records if r["status"] != "success"]
    report = ProcessingReport(skipped=len(records) - len(pending), remaining=len(pending))
    if not pending:
        report.message = "现有成功摘要已复用，没有重复调用 AI。"
        return report
    try:
        config.validate()
    except ConfigurationError as exc:
        report.message = str(exc) + " 当前页面仍可查看原摘要，配置后可生成摘要。"
        return report
    for index, row in enumerate(pending):
        paper = paper_from_record(row)
        try:
            summary = extractor(paper, config)
            store.set_summary(paper, summary)
            report.success += 1
            report.remaining -= 1
        except (PermanentServiceError, TemporaryServiceError) as exc:
            store.set_error(paper, str(exc))
            report.failed += 1
            report.remaining -= 1
            report.message = str(exc) + " 已停止后续 AI 调用，其他论文仍保留。"
            if on_progress:
                on_progress(index + 1, len(pending))
            break
        except ServiceError as exc:
            store.set_error(paper, str(exc))
            report.failed += 1
            report.remaining -= 1
            report.errors.append(f"{paper.id}：{exc}")
        if on_progress:
            on_progress(index + 1, len(pending))
    return report
