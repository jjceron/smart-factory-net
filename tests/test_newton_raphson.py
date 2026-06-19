# Tests for the Newton-Raphson velocity solver.

import pytest
import numpy as np
from src.newton_raphson import newton_raphson, f, f_prime, compute_edge_velocities


class TestNewtonRaphson:
    def test_f_at_known_point(self):
        # For rho=0.5, compute f(2.0) manually
        val = f(2.0, 0.5)
        expected = 2.0 + 0.4 * np.log(3.0) - 3.0 * (1.0 - 0.25)
        assert abs(val - expected) < 1e-10

    def test_f_prime(self):
        fp = f_prime(2.0)
        expected = 1.0 + 0.4 / 3.0
        assert abs(fp - expected) < 1e-10

    def test_convergence(self):
        vel, iters, error = newton_raphson(0.5, 1.0, tol=1e-6, max_iter=100)
        assert vel is not None
        assert vel > 0
        assert abs(f(vel, 0.5)) < 1e-6

    def test_multiple_initial_guesses(self):
        rho = 0.7
        vels = []
        for v0 in [0.2, 1.0, 5.0]:
            vel, iters, err = newton_raphson(rho, v0, tol=1e-6)
            assert vel is not None
            assert vel > 0
            vels.append(vel)
        # All should converge to the same root
        assert max(vels) - min(vels) < 1e-4

    def test_high_utilization(self):
        vel, iters, error = newton_raphson(0.9, 1.0, tol=1e-6)
        assert vel is not None
        assert vel > 0

    def test_low_utilization(self):
        vel, iters, error = newton_raphson(0.3, 1.0, tol=1e-6)
        assert vel is not None


class TestEdgeVelocities:
    def test_compute_multiple_edges(self):
        edges = [
            {"length": 30, "capacity": 50, "flow": 30, "id": "E0", "source": "A", "target": "B"},
            {"length": 20, "capacity": 40, "flow": 20, "id": "E1", "source": "B", "target": "C"},
        ]
        results = compute_edge_velocities(edges)
        assert len(results) == 6  # 2 edges * 3 initial guesses
        for r in results:
            assert r["converged"] is True
            assert r["velocity"] is not None
            assert r["travel_time"] is not None
