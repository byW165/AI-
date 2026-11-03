"""Deterministic worker tests: no network, files, sleeps, or Streamlit calls."""
from dataclasses import replace
import gc
from threading import Event
import time
import unittest
from unittest.mock import Mock
import weakref

from src.ai_jobs import SummaryJob
from src.config import AIConfig
from src.http_client import PermanentServiceError, ServiceError, TemporaryServiceError
from src.session_results import SessionResults
from tests.test_core import CONFIG, PAPER, SUMMARY


class SummaryJobTests(unittest.TestCase):
    def store(self, count=1):
        store = SessionResults()
        for index in range(count):
            store.upsert(replace(PAPER, id=f"2611.{index + 1:05}"))
        return store

    def finish_worker(self, job):
        # Waiting is test-only. Production tick never joins its worker.
        self.assertIsNotNone(job._thread)
        job._thread.join(timeout=2)
        self.assertFalse(job.busy, "test worker did not finish")

    def test_blocked_request_does_not_block_tick_or_duplicate_calls(self):
        entered, release = Event(), Event()
        def extract(paper, config):
            entered.set()
            release.wait(2)
            return SUMMARY
        extractor = Mock(side_effect=extract)
        store = self.store(2)
        job = SummaryJob(store.all(), CONFIG, extractor=extractor)
        try:
            started = time.monotonic()
            self.assertTrue(job.tick(store))
            self.assertLess(time.monotonic() - started, .5)
            self.assertTrue(entered.wait(1))
            for _ in range(5):
                self.assertFalse(job.tick(store))
            self.assertTrue(job.active)
            self.assertTrue(job.busy)
            self.assertEqual(extractor.call_count, 1)
            self.assertTrue(all(row["status"] == "pending" for row in store.all()))
        finally:
            job.cancel()
            release.set()
            self.finish_worker(job)
            job.tick(store)

    def test_success_is_applied_only_by_tick_and_then_next_request_starts(self):
        extractor = Mock(return_value=SUMMARY)
        store = self.store(2)
        job = SummaryJob(store.all(), CONFIG, extractor=extractor)
        job.tick(store)
        self.finish_worker(job)
        self.assertTrue(all(row["status"] == "pending" for row in store.all()))
        self.assertTrue(job.tick(store))
        self.assertEqual(job.report.success, 1)
        self.finish_worker(job)
        job.tick(store)
        self.assertFalse(job.active)
        self.assertEqual((job.report.success, job.report.remaining), (2, 0))
        self.assertTrue(all(row["status"] == "success" for row in store.all()))

    def test_temporary_service_failure_pauses_remaining_batch(self):
        store = self.store(3)
        extractor = Mock(side_effect=TemporaryServiceError("AI 暂时不可用（HTTP 503）"))
        job = SummaryJob(store.all(), CONFIG, extractor=extractor)
        job.tick(store)
        self.finish_worker(job)
        job.tick(store)
        self.assertEqual(extractor.call_count, 1)
        self.assertEqual((job.report.failed, job.report.remaining), (1, 2))
        self.assertFalse(job.active)
        self.assertIn("503", job.report.message)
        self.assertEqual(sum(row["status"] == "pending" for row in store.all()), 2)

    def test_permanent_failure_pauses_remaining_batch(self):
        store = self.store(2)
        extractor = Mock(side_effect=PermanentServiceError("AI：密钥无效（HTTP 401）"))
        job = SummaryJob(store.all(), CONFIG, extractor=extractor)
        job.tick(store)
        self.finish_worker(job)
        job.tick(store)
        self.assertFalse(job.active)
        self.assertEqual(extractor.call_count, 1)
        self.assertEqual((job.report.failed, job.report.remaining), (1, 1))

    def test_format_failure_is_isolated_and_next_paper_succeeds(self):
        store = self.store(2)
        extractor = Mock(side_effect=[ServiceError("AI JSON 解析或字段校验失败"), SUMMARY])
        job = SummaryJob(store.all(), CONFIG, extractor=extractor)
        job.tick(store)
        self.finish_worker(job)
        job.tick(store)
        self.finish_worker(job)
        job.tick(store)
        self.assertFalse(job.active)
        self.assertEqual((job.report.failed, job.report.success, job.report.remaining), (1, 1, 0))

    def test_cancel_keeps_busy_until_request_ends_and_discards_late_result(self):
        entered, release = Event(), Event()
        def extract(paper, config):
            entered.set()
            release.wait(2)
            return SUMMARY
        store = self.store(2)
        extractor = Mock(side_effect=extract)
        job = SummaryJob(store.all(), CONFIG, extractor=extractor)
        try:
            job.tick(store)
            self.assertTrue(entered.wait(1))
            job.cancel()
            self.assertTrue(job.active)
            self.assertTrue(job.busy)
            self.assertIn("当前请求仍在结束", job.report.message)
        finally:
            release.set()
            self.finish_worker(job)
        job.tick(store)
        self.assertFalse(job.active)
        self.assertEqual(extractor.call_count, 1)
        self.assertEqual(job.report.success, 0)
        self.assertTrue(all(row["status"] == "pending" for row in store.all()))

    def test_worker_does_not_own_job_or_mutate_replaced_store(self):
        entered, release, finished = Event(), Event(), Event()
        def extract(paper, config):
            entered.set()
            release.wait(2)
            finished.set()
            return SUMMARY
        old_store = self.store(2)
        extractor = Mock(side_effect=extract)
        job = SummaryJob(old_store.all(), CONFIG, extractor=extractor)
        job.tick(old_store)
        thread = job._thread
        reference = weakref.ref(job)
        try:
            self.assertTrue(entered.wait(1))
            del job
            gc.collect()
            self.assertIsNone(reference(), "worker must not keep the coordinator alive")
        finally:
            release.set()
            thread.join(timeout=2)
        self.assertTrue(finished.is_set())
        self.assertEqual(extractor.call_count, 1)
        self.assertTrue(all(row["status"] == "pending" for row in old_store.all()))
        self.assertTrue(all(row["status"] == "pending" for row in self.store().all()))

    def test_different_store_rejects_late_result(self):
        store = self.store()
        job = SummaryJob(store.all(), CONFIG, extractor=Mock(return_value=SUMMARY))
        job.tick(store)
        self.finish_worker(job)
        replacement = self.store()
        job.tick(replacement)
        self.assertFalse(job.active)
        self.assertEqual(store.all()[0]["status"], "pending")
        self.assertEqual(replacement.all()[0]["status"], "pending")

    def test_changed_abstract_cannot_receive_old_summary(self):
        store = self.store()
        job = SummaryJob(store.all(), CONFIG, extractor=Mock(return_value=SUMMARY))
        job.tick(store)
        self.finish_worker(job)
        store.upsert(replace(PAPER, abstract="Changed synthetic abstract"))
        job.tick(store)
        self.assertFalse(job.active)
        self.assertEqual(store.all()[0]["status"], "pending")
        self.assertIn("结果已变化", job.report.message)

    def test_unexpected_error_is_fixed_text_without_exception_or_key(self):
        store = self.store(2)
        extractor = Mock(side_effect=RuntimeError("upstream-sensitive-" + CONFIG.api_key))
        job = SummaryJob(store.all(), CONFIG, extractor=extractor)
        job.tick(store)
        self.finish_worker(job)
        job.tick(store)
        self.assertFalse(job.active)
        self.assertEqual(extractor.call_count, 1)
        self.assertNotIn(CONFIG.api_key, job.report.message)
        self.assertNotIn("upstream-sensitive", repr(job.report))
        self.assertNotIn("upstream-sensitive", store.all()[0]["error"] or "")

    def test_missing_configuration_never_starts_request(self):
        store = self.store()
        extractor = Mock()
        job = SummaryJob(store.all(), AIConfig(), extractor=extractor)
        self.assertFalse(job.active)
        job.tick(store)
        extractor.assert_not_called()
        self.assertIn("配置", job.report.message)
        self.assertEqual(job.report.remaining, 1)

    def test_successful_rows_are_skipped_and_copied_authors_are_independent(self):
        store = self.store(2)
        first = store.all()[0]
        from src.session_results import paper_from_record
        store.set_summary(paper_from_record(first), SUMMARY)
        extractor = Mock(return_value=SUMMARY)
        rows = store.all()
        job = SummaryJob(rows, CONFIG, extractor=extractor)
        rows[1]["authors"].append("Changed outside job")
        job.tick(store)
        self.finish_worker(job)
        job.tick(store)
        self.assertEqual(job.report.skipped, 1)
        self.assertEqual(extractor.call_count, 1)
        self.assertNotIn("Changed outside job", extractor.call_args.args[0].authors)

    def test_elapsed_uses_monotonic_clock_and_cancel_before_start_is_safe(self):
        entered, release = Event(), Event()
        def extract(paper, config):
            entered.set()
            release.wait(2)
            return SUMMARY
        now = [100.0]
        store = self.store()
        job = SummaryJob(store.all(), CONFIG, extractor=extract, clock=lambda: now[0])
        try:
            job.tick(store)
            self.assertTrue(entered.wait(1))
            now[0] += 7.5
            self.assertEqual(job.elapsed, 7.5)
            self.assertEqual(job.current_id, PAPER.id)
            self.assertEqual(job.current_title, PAPER.title)
        finally:
            job.cancel()
            release.set()
            self.finish_worker(job)
            job.tick(store)
        extractor = Mock()
        never_started = SummaryJob(store.all(), CONFIG, extractor=extractor)
        never_started.cancel()
        never_started.tick(store)
        self.assertFalse(never_started.active)
        extractor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
