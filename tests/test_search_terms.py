import unittest

from src.fetch_papers import ALL_ARXIV, build_query
from src.search_terms import SearchTermError, keyword_query
from tests.test_core import START, END


class SearchTermTests(unittest.TestCase):
    def test_chinese_gnn_expands_to_english_families(self):
        query = keyword_query("图神经网络")
        for term in ["graph neural network", "graph convolutional network", "graph attention network", "GNN", "GCN", "GAT"]:
            self.assertIn(term, query)
        self.assertNotIn("图神经网络", query)

    def test_composite_concepts_are_combined_with_and(self):
        query = keyword_query("图神经网络 加速度预测")
        self.assertIn(" AND ", query)

    def test_unknown_chinese_is_explained_not_sent_as_empty_search(self):
        with self.assertRaises(SearchTermError):
            keyword_query("量子纠缠")

    def test_all_arxiv_search_does_not_add_vehicle_restriction(self):
        query = build_query(ALL_ARXIV, START, END, "图神经网络")
        self.assertNotIn('abs:"vehicle"', query)
        self.assertNotIn("ANDNOT", query)
        self.assertIn("submittedDate", query)
        with self.assertRaises(SearchTermError):
            build_query(ALL_ARXIV, START, END)
