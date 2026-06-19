# Newton-Raphson module for conveyor velocity estimation.
# Author: Ceron Ordonez, J. J., M.Sc (c) in Electrical Engineering
# AI Assistant: DeepSeek V4 Flash
#
# Solves for effective velocity v on each conveyor belt given:
#   f(v) = v + 0.4 * ln(1 + v) - 3 * (1 - rho^2)
# where rho = Q / C is the utilization index.

import numpy as np


def f(v, rho):
    """The velocity equation: f(v) = v + 0.4*ln(1+v) - 3*(1 - rho^2)."""
    if v <= -1:
        return np.inf
    return v + 0.4 * np.log(1 + v) - 3.0 * (1.0 - rho**2)


def f_prime(v):
    """Derivative: f'(v) = 1 + 0.4/(1+v)."""
    return 1.0 + 0.4 / (1.0 + v)


def newton_raphson(rho, v0, tol=1e-6, max_iter=100):
    """Solve for velocity using Newton-Raphson method.
    
    Args:
        rho: Utilization index Q/C (0 < rho < 1)
        v0: Initial guess for velocity
        tol: Convergence tolerance
        max_iter: Maximum iterations
        
    Returns:
        (velocity, iterations, error) or (None, iterations, error) if not converged
    """
    v = float(v0)
    for i in range(max_iter):
        fv = f(v, rho)
        if abs(fv) < tol:
            return v, i + 1, abs(fv)
        fp = f_prime(v)
        if abs(fp) < 1e-15:
            return None, i + 1, abs(fv)
        v_new = v - fv / fp
        if v_new <= 0:
            return None, i + 1, abs(fv)
        v = v_new

    return None, max_iter, abs(f(v, rho))


def compute_edge_velocities(edges, initial_guesses=(0.2, 1.0, 5.0)):
    """Compute velocity for each edge with multiple initial guesses.
    
    Args:
        edges: List of dicts with keys 'length', 'capacity', 'flow'
        initial_guesses: Tuple of initial velocity guesses
        
    Returns:
        List of dicts with velocity results per edge per initial guess
    """
    results = []
    for idx, edge in enumerate(edges):
        L = edge["length"]
        C = edge["capacity"]
        Q = edge["flow"]
        rho = Q / C

        for v0 in initial_guesses:
            vel, iters, error = newton_raphson(rho, v0)
            travel_time = L / vel if (vel is not None and vel > 0) else None

            results.append({
                "edge_index": idx,
                "edge_id": edge.get("id", f"E{idx}"),
                "source": edge.get("source", ""),
                "target": edge.get("target", ""),
                "length": L,
                "capacity": C,
                "flow": Q,
                "utilization": round(rho, 4),
                "initial_guess": v0,
                "converged": vel is not None,
                "velocity": round(vel, 6) if vel is not None else None,
                "iterations": iters,
                "final_error": error,
                "travel_time": round(travel_time, 4) if travel_time else None,
            })
    return results


def print_velocity_results(results):
    """Return formatted table of velocity results."""
    lines = []
    lines.append("=" * 120)
    lines.append("NEWTON-RAPHSON VELOCITY ANALYSIS")
    lines.append("=" * 120)

    # Group by edge
    edges_seen = set()
    for r in results:
        eid = r["edge_id"]
        if eid not in edges_seen:
            edges_seen.add(eid)
            lines.append(f"\nEdge {eid}: {r['source']} -> {r['target']}")
            lines.append(f"  L={r['length']}, C={r['capacity']}, Q={r['flow']}, rho={r['utilization']}")
            lines.append(f"  {'v0':>8} {'Converged':>10} {'v':>12} {'Iter':>6} {'Error':>12} {'T_a':>10}")
            lines.append(f"  {'-'*60}")
        lines.append(f"  {r['initial_guess']:>8.1f} {'YES' if r['converged'] else 'NO':>10} "
                      f"{str(r['velocity']):>12} {r['iterations']:>6} {r['final_error']:>12.2e} "
                      f"{str(r['travel_time']):>10}")

    lines.append("\n" + "=" * 120)
    return "\n".join(lines)


def edges_from_graph(G):
    """Extract edge data list from a NetworkX DiGraph."""
    edges = []
    for i, (u, v, d) in enumerate(G.edges(data=True)):
        edges.append({
            "id": f"({u}->{v})",
            "source": u,
            "target": v,
            "length": d["length"],
            "capacity": d["capacity"],
            "flow": d["flow"],
        })
    return edges
