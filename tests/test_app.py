"""Session-only dashboard tests. No real network or paper persistence."""
from dataclasses import replace
from datetime import timedelta
import os
from pathlib import Path
import unittest
from unittest.mock import patch
from threading import Event

from streamlit.testing.v1 import AppTest

from src.fetch_papers import FetchResult
from src.http_client import ServiceError
from src.models import Summary, iso_time, utc_now
from src.session_results import SessionResults
from tests.test_core import PAPER

ROOT = Path(__file__).resolve().parents[1]


def result_for(papers, total=84, offset=0, next_start=1):
    now = utc_now()
    return FetchResult(papers, total, total > next_start, now - timedelta(days=180), now,
                       "synthetic test query", offset, next_start)


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"AI_API_KEY": "", "AI_BASE_URL": "", "AI_MODEL": ""})
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def app(self, papers=None):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=15)
        if papers is not None:
            results = SessionResults()
            for paper, summary in papers:
                results.upsert(paper)
                if summary:
                    results.set_summary(paper, summary)
            app.session_state["paper_results"] = results
            app.session_state["online_search"] = {
                "ids": [p.id for p, _ in papers], "total": len(papers), "truncated": False,
                "start": iso_time(utc_now() - timedelta(days=180)), "end": iso_time(utc_now()),
                "query": "synthetic test query", "keyword": "", "profile": "综合追踪",
                "days": 180, "limit": 20, "next_start": len(papers),
            }
        return app.run()

    def test_empty_startup_has_no_local_mode_or_paper_history(self):
        with patch("src.fetch_papers.fetch_papers") as fetch:
            app = self.app()
            self.assertEqual(len(app.radio), 0)
            self.assertEqual(len(app.metric), 0)
            self.assertTrue(any("当前页面会话" in item.value for item in app.markdown))
            self.assertEqual(len(app.exception), 0)
            fetch.assert_not_called()

    def test_chinese_online_submit_without_key_uses_memory_not_database(self):
        with patch("src.fetch_papers.fetch_papers", return_value=result_for([PAPER])) as fetch, \
             patch("sqlite3.connect", side_effect=AssertionError("不得保存论文数据库")), \
             patch("src.pipeline.summarize") as ai:
            app = self.app()
            app.text_input(key="keyword").set_value("图神经网络")
            app.button(key="search_button").click().run()
            self.assertEqual(fetch.call_args.kwargs["keyword"], "图神经网络")
            self.assertEqual(app.metric[0].value, "1")
            self.assertEqual(app.session_state["paper_results"].all()[0]["abstract"], PAPER.abstract)
            self.assertTrue(any("arXiv API 命中 84 篇" in item.value for item in app.markdown))
            ai.assert_not_called()
            self.assertEqual(len(app.exception), 0)

    def test_next_batch_appends_using_fixed_query_end(self):
        p2 = replace(PAPER, id="2611.00002")
        first = result_for([PAPER], total=3)
        second = replace(first, papers=[p2], offset=1, next_start=2)
        with patch("src.fetch_papers.fetch_papers", side_effect=[first, second]) as fetch:
            app = self.app()
            app.number_input(key="limit").set_value(1).run()
            app.button(key="search_button").click().run()
            app.button(key="load_more").click().run()
            self.assertEqual(fetch.call_args.kwargs["offset"], 1)
            self.assertEqual(fetch.call_args.kwargs["end"], first.end)
            self.assertEqual(app.metric[0].value, "2")
            self.assertEqual(len(app.exception), 0)

    def test_new_search_replaces_previous_query_results(self):
        p2 = replace(PAPER, id="2611.00002")
        with patch("src.fetch_papers.fetch_papers", side_effect=[result_for([PAPER]), result_for([p2])]):
            app = self.app()
            app.button(key="search_button").click().run()
            app.selectbox(key="code_filter").select("No").run()
            self.assertEqual(app.metric[0].value, "0")
            app.text_input(key="keyword").set_value("NGSIM")
            app.button(key="search_button").click().run()
            self.assertEqual([r["id"] for r in app.session_state["paper_results"].all()], [p2.id])
            self.assertEqual(app.metric[0].value, "1")
            self.assertEqual(app.selectbox(key="code_filter").value, "全部")

    def test_new_browser_session_has_no_previous_results(self):
        app = self.app([(PAPER, Summary("人工测试摘要", "GAT", "Unknown"))])
        self.assertEqual(app.metric[0].value, "1")
        fresh = self.app()
        self.assertEqual(len(fresh.metric), 0)
        self.assertEqual(fresh.session_state["paper_results"].all(), [])

    def test_clear_results_releases_records_and_query(self):
        app = self.app([(PAPER, Summary("人工测试摘要", "GAT", "Unknown"))])
        app.button(key="reset_filters").click().run()
        self.assertEqual(len(app.metric), 0)
        self.assertEqual(app.session_state["paper_results"].all(), [])
        self.assertNotIn("online_search", app.session_state)

    def test_code_filter_is_session_only_unknown_is_not_no(self):
        p2 = replace(PAPER, id="2611.00002")
        with patch("requests.Session.request", side_effect=AssertionError("状态筛选不得联网")):
            app = self.app([(PAPER, Summary("人工测试", "GAT", "Unknown")), (p2, Summary("人工测试", "LSTM", "Yes"))])
            self.assertEqual(app.metric[0].value, "2")
            app.selectbox(key="code_filter").select("No").run()
            self.assertEqual(app.metric[0].value, "0")
            app.selectbox(key="code_filter").select("Unknown").run()
            self.assertEqual(app.metric[0].value, "1")
            self.assertEqual(len(app.exception), 0)

    def test_valid_empty_source_is_not_api_failure(self):
        with patch("src.fetch_papers.fetch_papers", return_value=result_for([], total=0)):
            app = self.app()
            app.button(key="search_button").click().run()
            self.assertEqual(len(app.error), 0)
            self.assertTrue(any("本次检索未找到匹配论文" in item.value for item in app.info))
            self.assertEqual(len(app.exception), 0)

    def test_source_failure_does_not_erase_current_session_results(self):
        with patch("src.fetch_papers.fetch_papers", side_effect=ServiceError("模拟 429 达到重试上限")):
            app = self.app([(PAPER, None)])
            app.button(key="search_button").click().run()
            self.assertTrue(any("联网检索失败" in item.value for item in app.error))
            self.assertEqual(app.metric[0].value, "1")
            self.assertEqual(len(app.exception), 0)

    def test_retry_uses_current_results_only(self):
        p2 = replace(PAPER, id="2611.00002")
        with patch.dict(os.environ, {"AI_BASE_URL": "https://example.invalid", "AI_MODEL": "test", "AI_API_KEY": "synthetic"}), \
             patch("src.ai_jobs.summarize", return_value=Summary("恢复后的测试摘要", "GAT", "Unknown")) as ai:
            app = self.app([(PAPER, None), (p2, Summary("已有测试摘要", "LSTM", "Yes"))])
            app.button(key="retry_all").click().run()
            app.run()
            self.assertEqual(app.metric[1].value, "2")
            self.assertEqual(ai.call_count, 1)
            self.assertEqual(len(app.exception), 0)

    def test_ai_wait_does_not_hide_papers_or_block_ui(self):
        release, started = Event(), Event()
        def slow_summary(paper, config):
            started.set()
            release.wait(5)
            return Summary("人工后台测试摘要", "GAT", "Unknown")
        with patch.dict(os.environ, {"AI_BASE_URL": "https://example.invalid", "AI_MODEL": "test", "AI_API_KEY": "synthetic"}), \
             patch("src.fetch_papers.fetch_papers", return_value=result_for([PAPER])), \
             patch("src.ai_jobs.summarize", side_effect=slow_summary):
            try:
                app = self.app()
                app.button(key="search_button").click().run()
                self.assertTrue(started.is_set())
                self.assertEqual(app.metric[0].value, "1")
                self.assertEqual(app.metric[1].value, "0")
                self.assertTrue(any(PAPER.title in m.value for m in app.markdown))
                self.assertEqual(len(app.button(key="stop_ai").label), len("停止后续摘要"))
                self.assertEqual(len(app.exception), 0)
                app.button(key="reset_filters").click().run()
                self.assertEqual(len(app.metric), 0)
                self.assertEqual(app.session_state["paper_results"].all(), [])
            finally:
                release.set()
