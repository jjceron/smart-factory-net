# Integration test — end-to-end pipeline verification.

import pytest
from src.network_generation import build_base_network
from src.graph_analysis import connectivity_report, is_strongly_connected
from src.shortest_paths import compare_algorithms
from src.newton_raphson import edges_from_graph, compute_edge_velocities
from src.optimization_model import prepare_edge_data, solve_optimization
from src.sensitivity_analysis import run_sensitivity


class TestFullPipeline:
    def test_end_to_end(self):
        """Run the full analysis pipeline on the base network."""
        G, od_pairs = build_base_network()

        # Connectivity
        assert is_strongly_connected(G) is True

        # Shortest paths
        results = compare_algorithms(G, od_pairs)
        assert len(results) == len(od_pairs)
        for r in results:
            assert r["dijkstra_path"] is not None
            assert r["astar_path"] is not None
            assert r["dijkstra_distance"] == r["astar_distance"]
            assert r["paths_match"] is True

        # Newton-Raphson
        edges = edges_from_graph(G)
        vel_results = compute_edge_velocities(edges)
        assert len(vel_results) == G.number_of_edges() * 3

        # Optimization
        df = prepare_edge_data(G)
        prob, opt_results, summary = solve_optimization(df)
        assert summary["status"] == "Optimal"
        assert summary["percentage_reduction"] >= 0

        # Sensitivity
        sens = run_sensitivity(df)
        assert len(sens) == 10
        expected = [round(0.1 * i, 1) for i in range(1, 11)]
        assert (sens["budget_fraction"].round(1) == expected).all()

    def test_report_generation(self):
        """Test that all report functions produce output."""
        G, od_pairs = build_base_network()

        report = connectivity_report(G)
        assert len(report) > 0

        from src.shortest_paths import print_comparison
        results = compare_algorithms(G, od_pairs)
        comparison = print_comparison(results)
        assert len(comparison) > 0
