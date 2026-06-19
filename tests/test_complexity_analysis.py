# Tests for the complexity analysis module.

import pytest
from src.complexity_analysis import run_complexity_analysis


class TestComplexity:
    def test_returns_dataframe(self):
        df = run_complexity_analysis(sizes=(20, 40), seed=42)
        assert len(df) == 4  # 2 sizes * 2 algorithms
        assert list(df.columns) == ["size", "edges", "algorithm", "time_ms",
                                     "expanded_nodes", "distance"]

    @pytest.mark.parametrize("algo", ["Dijkstra", "A*"])
    def test_both_algorithms_present(self, algo):
        df = run_complexity_analysis(sizes=(20,), seed=42)
        assert algo in df["algorithm"].values

    def test_times_positive(self):
        df = run_complexity_analysis(sizes=(20,), seed=42)
        assert (df["time_ms"] >= 0).all()
        assert (df["expanded_nodes"] > 0).all()
