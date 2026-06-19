# Tests for the graph analysis module.

import pytest
import networkx as nx
from src.graph_analysis import is_strongly_connected, find_scc, connectivity_report


@pytest.fixture
def strong_graph():
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    return G


@pytest.fixture
def weak_graph():
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 2)])  # 2 unreachable from 1,0
    G.add_node(3)  # isolated
    return G


class TestStrongConnectivity:
    def test_strongly_connected(self, strong_graph):
        assert is_strongly_connected(strong_graph) is True

    def test_not_strongly_connected(self, weak_graph):
        assert is_strongly_connected(weak_graph) is False

    def test_empty_graph(self):
        G = nx.DiGraph()
        assert is_strongly_connected(G) is False


class TestSCC:
    def test_single_scc(self, strong_graph):
        comps = find_scc(strong_graph)
        assert len(comps) == 1

    def test_multiple_scc(self, weak_graph):
        comps = find_scc(weak_graph)
        assert len(comps) >= 2

    def test_report_generation(self, strong_graph):
        report = connectivity_report(strong_graph)
        assert "YES" in report
