from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import requests

from src.ai_summary import parse_summary, summarize
from src.config import AIConfig, ConfigurationError
from src.fetch_papers import build_query, canonical_id, fetch_papers, parse_feed
from src.http_client import PermanentServiceError, TemporaryServiceError, RetryableError, ServiceError, request_once, with_retries
from src.models import Paper, Summary
from src.pipeline import process_records
from src.session_results import SessionResults, filter_records, MAX_SESSION_PAPERS

END = datetime(2026, 11, 3, 6, 0, tzinfo=timezone.utc)
START = END - timedelta(days=30)
PAPER = Paper("2611.00001", "Synthetic test: acceleration prediction", ["Test Author"],
              "2026-11-02T00:00:00+00:00", "2026-11-02T00:00:00+00:00",
              "Synthetic abstract for tests. A GAT predicts acceleration. Dataset unspecified.",
              "https://arxiv.org/abs/2611.00001")
SUMMARY = Summary("使用图模型预测车辆加速度。", "GAT；数据集未说明。", "Unknown")
CONFIG = AIConfig("https://example.invalid", "test-model", "synthetic-key-not-a-secret")


def response(status=200, data=None, content=b"", headers=None):
    obj = Mock(status_code=status, headers=headers or {}, content=content)
    obj.json.return_value = data
    obj.iter_content.side_effect = lambda chunk_size=1: iter([json.dumps(data).encode("utf-8")])
    return obj


def envelope(content):
    return {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}


def feed(entries=None, total=1):
    entries = entries if entries is not None else [PAPER]
    from xml.sax.saxutils import escape
    rows = "".join(f"""<entry><id>https://arxiv.org/abs/{p.id}v1</id><title>{escape(p.title)}</title>
        <published>{p.published}</published><updated>{p.updated}</updated><summary>{escape(p.abstract)}</summary>
        <author><name>Test Author</name></author></entry>""" for p in entries)
    return f'<feed xmlns="http://www.w3.org/2005/Atom" xmlns:o="http://a9.com/-/spec/opensearch/1.1/"><o:totalResults>{total}</o:totalResults>{rows}</feed>'.encode()


class FetchTests(unittest.TestCase):
    def test_date_range_dedupe_and_original_abstract(self):
        older = replace(PAPER, id="2601.00001", published="2026-01-01T00:00:00+00:00")
        papers, total, truncated = parse_feed(feed([PAPER, PAPER, older], 3), START, END, 2)
        self.assertEqual([p.id for p in papers], [PAPER.id])
        self.assertEqual(papers[0].abstract, PAPER.abstract)
        self.assertEqual(total, 3)
        self.assertTrue(truncated)

    def test_first_submitted_not_updated(self):
        old = replace(PAPER, published="2026-01-01T00:00:00+00:00")
        self.assertEqual(parse_feed(feed([old]), START, END, 20)[0], [])

    def test_boundary_dates_inclusive(self):
        p = replace(PAPER, published=START.isoformat())
        self.assertEqual(len(parse_feed(feed([p]), START, END, 20)[0]), 1)

    def test_invalid_atom_is_not_empty_result(self):
        for content in [b"not XML", b"<html/>", b'<feed xmlns="http://www.w3.org/2005/Atom"/>']:
            with self.subTest(content=content), self.assertRaises(RetryableError):
                parse_feed(content, START, END, 20)

    def test_valid_empty_result(self):
        self.assertEqual(parse_feed(feed([], 0), START, END, 20), ([], 0, False))

    def test_query_has_date_and_no_nine_node_restriction(self):
        query = build_query("综合追踪", START, END)
        self.assertIn("submittedDate:[202610040600 TO 202611030600]", query)
        self.assertNotIn("nine", query)
        self.assertEqual(canonical_id("http://arxiv.org/abs/hep-th/9901001v2"), "hep-th/9901001")

    def test_fetch_retries_429_then_succeeds(self):
        session = Mock()
        session.request.side_effect = [response(429, headers={"Retry-After": "2"}), response(content=feed())]
        wait, limiter = Mock(), Mock()
        result = fetch_papers(30, limit=1, end=END, session=session, sleep=wait, rate_limit=limiter)
        self.assertEqual(len(result.papers), 1)
        self.assertEqual(limiter.call_count, 2)
        wait.assert_called_once_with(2.0)

    def test_fetch_timeout_has_finite_budget(self):
        session = Mock()
        session.request.side_effect = requests.Timeout("sensitive upstream text")
        with self.assertRaises(ServiceError) as ctx:
            fetch_papers(end=END, session=session, sleep=Mock(), rate_limit=Mock())
        self.assertEqual(session.request.call_count, 3)
        self.assertNotIn("sensitive", str(ctx.exception))


class AITests(unittest.TestCase):
    def test_valid_json_and_unknown(self):
        self.assertEqual(parse_summary(json.dumps(SUMMARY.__dict__)), SUMMARY)

    def test_invalid_schema_is_retryable(self):
        invalid = ["bad json", "null", "[]", "{}", '{"core_contribution":1}',
                   json.dumps({**SUMMARY.__dict__, "code_status": "Maybe"}),
                   json.dumps({**SUMMARY.__dict__, "core_contribution": ""}),
                   json.dumps({**SUMMARY.__dict__, "extra": "unexpected"}),
                   '{"core_contribution":"x","core_contribution":"y","methods_and_datasets":"z","code_status":"No"}']
        for item in invalid:
            with self.subTest(item=item), self.assertRaises(RetryableError):
                parse_summary(item)

    def test_format_error_retries_with_correction(self):
        session = Mock()
        session.request.side_effect = [response(data=envelope("bad json")), response(data=envelope(json.dumps(SUMMARY.__dict__)))]
        self.assertEqual(summarize(PAPER, CONFIG, session=session, sleep=Mock()), SUMMARY)
        self.assertIn("上次输出格式", session.request.call_args.kwargs["json"]["messages"][-1]["content"])

    def test_transport_and_json_share_three_attempt_budget(self):
        session = Mock()
        session.request.side_effect = [requests.Timeout(), response(data=envelope("oops")), response(503)]
        with self.assertRaises(ServiceError):
            summarize(PAPER, CONFIG, session=session, sleep=Mock())
        self.assertEqual(session.request.call_count, 3)

    def test_auth_error_never_retries_or_echoes_body(self):
        session = Mock()
        session.request.return_value = response(401, data={"error": CONFIG.api_key})
        with self.assertRaises(PermanentServiceError) as ctx:
            summarize(PAPER, CONFIG, session=session, sleep=Mock())
        self.assertEqual(session.request.call_count, 1)
        self.assertNotIn(CONFIG.api_key, str(ctx.exception))
        self.assertNotIn(CONFIG.api_key, repr(CONFIG))

    def test_long_retry_after_stops_without_sleep(self):
        wait = Mock()
        def fail(attempt):
            raise RetryableError("limited", retry_after=120)
        with self.assertRaises(ServiceError):
            with_retries(fail, sleep=wait)
        wait.assert_not_called()

    def test_deepseek_disables_thinking_and_uses_json_mode(self):
        session = Mock()
        session.request.return_value = response(data=envelope(json.dumps(SUMMARY.__dict__)))
        summarize(PAPER, replace(CONFIG, base_url="https://api.deepseek.com"), session=session, sleep=Mock())
        payload = session.request.call_args.kwargs["json"]
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(payload["response_format"], {"type": "json_object"})

    def test_503_exhaustion_is_service_pause_not_key_error(self):
        session = Mock()
        session.request.return_value = response(503)
        with self.assertRaises(TemporaryServiceError) as ctx:
            summarize(PAPER, CONFIG, session=session, sleep=Mock())
        self.assertEqual(session.request.call_count, 3)
        self.assertIn("不代表密钥错误", str(ctx.exception))
        self.assertNotIn("密钥无效", str(ctx.exception))

    def test_heartbeat_cannot_extend_ai_budget_indefinitely(self):
        session = Mock()
        streamed = response()
        streamed.iter_content.side_effect = lambda chunk_size=1: iter([b" "])
        session.request.return_value = streamed
        clock = Mock(side_effect=[0, 0, 46])
        with self.assertRaises(TemporaryServiceError):
            summarize(PAPER, CONFIG, session=session, sleep=Mock(), clock=clock)
        self.assertEqual(session.request.call_count, 1)
        streamed.close.assert_called_once()

    def test_stream_disconnect_is_sanitized_and_pauses_batch(self):
        session = Mock()
        streamed = response()
        streamed.iter_content.side_effect = requests.ConnectionError("secret upstream body")
        session.request.return_value = streamed
        with self.assertRaises(TemporaryServiceError) as ctx:
            summarize(PAPER, CONFIG, session=session, sleep=Mock())
        self.assertEqual(session.request.call_count, 3)
        self.assertNotIn("secret", str(ctx.exception))

    def test_config_validation(self):
        for config in [AIConfig(), replace(CONFIG, base_url="http://remote.example"),
                       replace(CONFIG, base_url="https://example.invalid/chat/completions"),
                       replace(CONFIG, base_url="https://secret@example.invalid")]:
            with self.subTest(config=config), self.assertRaises(ConfigurationError):
                config.validate()


class SessionAndPipelineTests(unittest.TestCase):
    def setUp(self):
        self.store = SessionResults()
        self.store.upsert(PAPER)

    def test_dedup_and_cache_success(self):
        extractor = Mock(return_value=SUMMARY)
        first = process_records(self.store.all(), self.store, CONFIG, extractor=extractor)
        self.assertEqual(first.success, 1)
        self.assertFalse(self.store.upsert(PAPER))
        again = process_records(self.store.all(), self.store, CONFIG, extractor=extractor)
        self.assertEqual(again.skipped, 1)
        self.assertEqual(extractor.call_count, 1)
        self.assertEqual(len(self.store.all()), 1)

    def test_changed_abstract_invalidates_summary(self):
        self.store.set_summary(PAPER, SUMMARY)
        self.store.upsert(replace(PAPER, abstract="Changed synthetic abstract"))
        self.assertEqual(self.store.all()[0]["status"], "pending")
        self.assertIsNone(self.store.all()[0]["code_status"])

    def test_old_concurrent_summary_cannot_overwrite_new_abstract(self):
        self.store.upsert(replace(PAPER, abstract="Changed"))
        self.store.set_summary(PAPER, SUMMARY)
        self.assertEqual(self.store.all()[0]["status"], "pending")

    def test_missing_configuration_keeps_original_paper(self):
        extractor = Mock()
        report = process_records(self.store.all(), self.store, AIConfig(), extractor=extractor)
        extractor.assert_not_called()
        self.assertEqual(report.remaining, 1)
        self.assertEqual(self.store.all()[0]["abstract"], PAPER.abstract)

    def test_single_failure_does_not_block_others_and_can_retry(self):
        self.store.upsert(replace(PAPER, id="2611.00002"))
        report = process_records(self.store.all(), self.store, CONFIG, extractor=Mock(side_effect=[ServiceError("模拟格式错误"), SUMMARY]))
        self.assertEqual((report.failed, report.success), (1, 1))
        again = process_records(self.store.all(), self.store, CONFIG, extractor=Mock(return_value=SUMMARY))
        self.assertEqual((again.success, again.skipped), (1, 1))

    def test_permanent_auth_failure_stops_batch(self):
        self.store.upsert(replace(PAPER, id="2611.00002"))
        extractor = Mock(side_effect=PermanentServiceError("模拟密钥无效"))
        report = process_records(self.store.all(), self.store, CONFIG, extractor=extractor)
        self.assertEqual(extractor.call_count, 1)
        self.assertEqual(report.remaining, 1)

    def test_code_status_filter_preserves_unknown_distinction(self):
        self.store.set_summary(PAPER, SUMMARY)
        self.assertEqual(len(filter_records(self.store.all(), "Unknown")), 1)
        self.assertEqual(filter_records(self.store.all(), "No"), [])

    def test_new_session_is_empty_and_independent(self):
        self.store.set_summary(PAPER, SUMMARY)
        fresh = SessionResults()
        self.assertEqual(fresh.all(), [])
        self.assertEqual(len(self.store.all()), 1)

    def test_clear_releases_all_records(self):
        self.store.set_summary(PAPER, SUMMARY)
        self.store.clear()
        self.assertEqual(self.store.all(), [])

    def test_session_results_never_connect_to_database(self):
        from unittest.mock import patch
        with patch("sqlite3.connect", side_effect=AssertionError("不得创建论文数据库")):
            result = SessionResults()
            result.upsert(PAPER)
            result.set_summary(PAPER, SUMMARY)
            self.assertEqual(result.all()[0]["status"], "success")
            result.clear()

    def test_session_memory_has_explicit_limit(self):
        for index in range(1, MAX_SESSION_PAPERS):
            self.store.upsert(replace(PAPER, id=f"test-{index}"))
        with self.assertRaises(ValueError):
            self.store.upsert(replace(PAPER, id="one-too-many"))


if __name__ == "__main__":
    unittest.main()
