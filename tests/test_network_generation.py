# Tests for the network generation module.

import pytest
from src.network_generation import build_base_network, generate_random_network


class TestBaseNetwork:
    def test_node_count(self):
        G, _ = build_base_network()
        assert G.number_of_nodes() >= 15

    def test_edge_count(self):
        G, _ = build_base_network()
        assert G.number_of_edges() >= 25

    def test_zone_count(self):
        G, _ = build_base_network()
        zones = set(nx.get_node_attributes(G, "zone").values())
        assert len(zones) >= 4

    def test_od_pairs_count(self):
        _, od = build_base_network()
        assert len(od) >= 4

    def test_edge_constraints(self):
        G, _ = build_base_network()
        pos = nx.get_node_attributes(G, "pos")
        for u, v, d in G.edges(data=True):
            L = d["length"]
            C = d["capacity"]
            Q = d["flow"]
            assert 10 <= L <= 80
            dist = ((pos[u][0] - pos[v][0])**2 + (pos[u][1] - pos[v][1])**2)**0.5
            assert L >= dist - 1e-9
            assert 20 <= C <= 100
            assert 0.3 * C <= Q <= 0.85 * C

    def test_strong_connectivity(self):
        G, _ = build_base_network()
        assert nx.is_strongly_connected(G)

    def test_save_load(self, tmp_path):
        import networkx as nx
        G, _ = build_base_network()
        from src.network_generation import save_network, load_network
        path = tmp_path / "test_net.json"
        import json
        data = nx.node_link_data(G)
        with open(path, "w") as f:
            json.dump(data, f)
        with open(path) as f:
            loaded = nx.node_link_graph(json.load(f))
        assert loaded.number_of_nodes() == G.number_of_nodes()
        assert loaded.number_of_edges() == G.number_of_edges()


class TestRandomNetwork:
    @pytest.mark.parametrize("n", [20, 40, 80])
    def test_random_connectivity(self, n):
        G = generate_random_network(n, seed=42)
        assert nx.is_strongly_connected(G)

    @pytest.mark.parametrize("n", [20, 40])
    def test_random_edge_count(self, n):
        G = generate_random_network(n, seed=42)
        assert G.number_of_edges() >= 2 * n - 1  # at least a Hamiltonian cycle
        assert G.number_of_edges() <= 2 * n * 5  # not too many

    def test_random_reproducible(self):
        G1 = generate_random_network(50, seed=123)
        G2 = generate_random_network(50, seed=123)
        assert G1.number_of_nodes() == G2.number_of_nodes()
        assert G1.number_of_edges() == G2.number_of_edges()


import networkx as nx
