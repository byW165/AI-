"""One in-memory summary request at a time; only tick() updates page results.

The daemon worker does not own the job, the store, or Streamlit state. Dropping
the job cannot launch further requests. Cancelling stops subsequent requests;
an HTTP request already in flight may still finish, and its result is discarded.
"""
from collections import deque
from dataclasses import dataclass, replace
from queue import Empty, Queue
from threading import Thread
import time
from typing import Callable
import weakref

from .ai_summary import summarize
from .config import AIConfig, ConfigurationError
from .http_client import PermanentServiceError, ServiceError, TemporaryServiceError
from .models import Paper, Summary
from .pipeline import ProcessingReport
from .session_results import SessionResults, paper_from_record

_UNEXPECTED_ERROR = "AI 摘要处理发生异常，本轮已停止；可稍后重试，原摘要仍可阅读。"


@dataclass(frozen=True)
class _Outcome:
    summary: Summary | None = None
    error: str = ""
    stop_batch: bool = False


def _extract_one(paper: Paper, config: AIConfig, outcomes: Queue,
                 extractor: Callable) -> None:
    """Module-level target: intentionally holds neither job nor result store."""
    try:
        summary = extractor(paper, config)
        if not isinstance(summary, Summary):
            raise TypeError("invalid extractor result")
        outcome = _Outcome(summary=summary)
    except (PermanentServiceError, TemporaryServiceError) as exc:
        outcome = _Outcome(error=str(exc), stop_batch=True)
    except ServiceError as exc:
        outcome = _Outcome(error=str(exc))
    except Exception:
        # Arbitrary exception text can include request headers or credentials.
        outcome = _Outcome(error=_UNEXPECTED_ERROR, stop_batch=True)
    outcomes.put(outcome)


class SummaryJob:
    """Session-owned coordinator. Call tick(store) on the UI/main thread only."""

    def __init__(self, records: list[dict], config: AIConfig, *, extractor=None,
                 clock=time.monotonic):
        # Paper has a mutable authors list, so copy it as well as the record data.
        papers = [paper_from_record(row) for row in records if row["status"] != "success"]
        self._pending = deque(replace(paper, authors=list(paper.authors)) for paper in papers)
        self.report = ProcessingReport(skipped=len(records) - len(papers), remaining=len(papers))
        self._config: AIConfig | None = config
        self._extractor = extractor or summarize
        self._clock = clock
        self._thread: Thread | None = None
        self._outcomes: Queue | None = None
        self._current: Paper | None = None
        self._started_at: float | None = None
        self._store_ref: weakref.ReferenceType | None = None
        self._stopped = False
        self._cancelled = False
        try:
            config.validate()
        except ConfigurationError as exc:
            self.report.message = str(exc) + " 当前页面仍可阅读原摘要。"
            self._finish()
        if not papers:
            self.report.message = "现有成功摘要已复用，没有重复调用 AI。"
            self._finish()

    @property
    def active(self) -> bool:
        # Keep polling until a completed worker's outcome has been consumed too.
        return self._thread is not None or (not self._stopped and bool(self._pending))

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def current_id(self) -> str | None:
        return self._current.id if self._current else None

    @property
    def current_title(self) -> str | None:
        return self._current.title if self._current else None

    @property
    def elapsed(self) -> float:
        return max(0.0, self._clock() - self._started_at) if self._started_at is not None else 0.0

    def _finish(self) -> None:
        self._stopped = True
        self._pending.clear()
        self._config = None

    def cancel(self) -> None:
        self._cancelled = True
        self._finish()
        self.report.message = (
            "已停止后续摘要处理；当前请求仍在结束中，其结果将丢弃。"
            if self.busy else "已停止后续摘要处理，未处理的论文仍可阅读或重试。"
        )

    @staticmethod
    def _matching_record(store: SessionResults, paper: Paper) -> dict | None:
        return next((row for row in store.all()
                     if row["id"] == paper.id and row["fingerprint"] == paper.fingerprint), None)

    def tick(self, store: SessionResults) -> bool:
        """Never waits for HTTP. Consume a result, then start at most one request.

        A job is bound to its first store. New searches must cancel the previous
        job before clearing/reusing that same store; a different store is rejected.
        """
        if self._store_ref is None:
            self._store_ref = weakref.ref(store)
        elif self._store_ref() is not store:
            self.cancel()

        if self.busy:
            return False

        changed = False
        if self._thread is not None:
            try:
                outcome = self._outcomes.get_nowait()
            except Empty:
                # A worker terminated outside the normal Exception path.
                outcome = _Outcome(error=_UNEXPECTED_ERROR, stop_batch=True)
            paper = self._current
            self._thread = None
            self._outcomes = None
            self._current = None
            self._started_at = None
            changed = True
            if not self._cancelled:
                row = self._matching_record(store, paper)
                if row is None:
                    self.cancel()
                    self.report.message = "论文结果已变化，本轮摘要已停止；请对当前结果重新生成。"
                elif row["status"] == "success":
                    self.report.skipped += 1
                    self.report.remaining -= 1
                elif outcome.summary is not None:
                    store.set_summary(paper, outcome.summary)
                    self.report.success += 1
                    self.report.remaining -= 1
                else:
                    store.set_error(paper, outcome.error)
                    self.report.failed += 1
                    self.report.remaining -= 1
                    self.report.errors.append(f"{paper.id}：{outcome.error}")
                    if outcome.stop_batch:
                        self.report.message = outcome.error + " 已暂停后续 AI 调用，其他论文仍可阅读；稍后可重试。"
                        self._finish()

        if self._stopped:
            return changed

        # Skip invalidated/already-completed rows without issuing a request.
        while self._pending:
            paper = self._pending.popleft()
            row = self._matching_record(store, paper)
            if row is None:
                self.cancel()
                self.report.message = "论文结果已变化，本轮摘要已停止；请对当前结果重新生成。"
                return True
            if row["status"] == "success":
                self.report.skipped += 1
                self.report.remaining -= 1
                changed = True
                continue
            self._current = paper
            self._started_at = self._clock()
            self._outcomes = Queue(maxsize=1)
            self._thread = Thread(target=_extract_one,
                                  args=(paper, self._config, self._outcomes, self._extractor),
                                  name="paper-summary", daemon=True)
            self._thread.start()
            return True

        self._finish()
        return True
