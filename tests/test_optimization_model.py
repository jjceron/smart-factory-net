# Tests for the MILP optimization model.

import pytest
import pandas as pd
from src.optimization_model import solve_optimization, prepare_edge_data


@pytest.fixture
def sample_edges():
    return pd.DataFrame([
        {"edge_id": "E1", "source": "A", "target": "B", "length": 20,
         "capacity": 50, "flow": 30, "cost": 2000,
         "travel_time": 2.0, "travel_time_modernized": 1.5},
        {"edge_id": "E2", "source": "B", "target": "C", "length": 30,
         "capacity": 60, "flow": 40, "cost": 2700,
         "travel_time": 3.0, "travel_time_modernized": 2.0},
        {"edge_id": "E3", "source": "C", "target": "D", "length": 25,
         "capacity": 45, "flow": 35, "cost": 2150,
         "travel_time": 2.5, "travel_time_modernized": 1.8},
    ])


class TestOptimization:
    def test_solve_returns_summary(self, sample_edges):
        prob, results, summary = solve_optimization(sample_edges, budget=3000)
        assert summary["status"] == "Optimal"
        assert summary["conveyors_total"] == 3
        assert 0 <= summary["conveyors_selected"] <= 3
        assert summary["budget_used"] <= summary["budget_available"]

    def test_no_budget(self, sample_edges):
        prob, results, summary = solve_optimization(sample_edges, budget=0)
        assert summary["conveyors_selected"] == 0

    def test_full_budget(self, sample_edges):
        total = sample_edges["cost"].sum()
        prob, results, summary = solve_optimization(sample_edges, budget=total)
        assert summary["conveyors_selected"] >= 1

    def test_indicator_improvement(self, sample_edges):
        prob, results, summary = solve_optimization(sample_edges, budget=3000)
        assert summary["indicator_final"] <= summary["indicator_initial"]
        assert summary["percentage_reduction"] >= 0
