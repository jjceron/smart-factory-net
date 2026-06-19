# Tests for the shortest path algorithms.

import pytest
import networkx as nx
from src.shortest_paths import dijkstra, astar, compare_algorithms


@pytest.fixture
def simple_graph():
    G = nx.DiGraph()
    G.add_node("A", pos=(0, 0))
    G.add_node("B", pos=(1, 0))
    G.add_node("C", pos=(2, 0))
    G.add_node("D", pos=(3, 0))
    G.add_edge("A", "B", length=10)
    G.add_edge("B", "C", length=10)
    G.add_edge("A", "C", length=25)
    G.add_edge("C", "D", length=10)
    return G


class TestDijkstra:
    def test_shortest_path(self, simple_graph):
        path, dist, exp = dijkstra(simple_graph, "A", "D")
        assert path == ["A", "B", "C", "D"]
        assert dist == 30.0

    def test_direct_path_preferred(self, simple_graph):
        path, dist, exp = dijkstra(simple_graph, "A", "C")
        assert path == ["A", "B", "C"]
        assert dist == 20.0

    def test_unreachable(self, simple_graph):
        path, dist, exp = dijkstra(simple_graph, "D", "A")
        assert path is None
        assert dist == float("inf")


class TestAStar:
    def test_shortest_path(self, simple_graph):
        path, dist, exp = astar(simple_graph, "A", "D")
        assert path == ["A", "B", "C", "D"]
        assert dist == 30.0

    def test_same_as_dijkstra(self, simple_graph):
        p_d, d_d, e_d = dijkstra(simple_graph, "A", "D")
        p_a, d_a, e_a = astar(simple_graph, "A", "D")
        assert p_d == p_a
        assert d_d == d_a


class TestComparison:
    def test_compare_algorithms(self, simple_graph):
        od = [("A", "D", "Test pair")]
        results = compare_algorithms(simple_graph, od)
        assert len(results) == 1
        r = results[0]
        assert r["dijkstra_distance"] == r["astar_distance"]
        assert r["paths_match"] is True
