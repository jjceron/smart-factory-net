# Shortest path algorithms — Dijkstra and A*.
# Author: Ceron Ordonez, J. J., M.Sc (c) in Electrical Engineering
# AI Assistant: DeepSeek V4 Flash
#
# Both algorithms are implemented from scratch using heapq.
# A* uses Euclidean distance as the admissible heuristic.

import heapq
import time
import numpy as np


def dijkstra(G, source, target, weight="length"):
    """Dijkstra's algorithm from scratch.
    
    Args:
        G: NetworkX DiGraph
        source: Source node ID
        target: Target node ID
        weight: Edge attribute for distance
        
    Returns:
        (path, distance, expanded_nodes)
    """
    dist = {source: 0.0}
    prev = {source: None}
    pq = [(0.0, source)]
    expanded = set()

    while pq:
        d, u = heapq.heappop(pq)
        if u in expanded:
            continue
        expanded.add(u)
        if u == target:
            break

        for _, v, data in G.edges(u, data=True):
            w = data.get(weight, 1.0)
            nd = d + w
            if v not in dist or nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if target not in prev:
        return None, float("inf"), len(expanded)

    path = []
    v = target
    while v is not None:
        path.append(v)
        v = prev[v]
    path.reverse()

    return path, dist[target], len(expanded)


def _euclidean_heuristic(pos, target):
    """Return h(node) = Euclidean distance to target."""
    t_pos = pos[target]
    def h(node):
        p = pos[node]
        return np.sqrt((p[0] - t_pos[0])**2 + (p[1] - t_pos[1])**2)
    return h


def astar(G, source, target, weight="length"):
    """A* search algorithm from scratch with Euclidean heuristic.
    
    Args:
        G: NetworkX DiGraph with 'pos' attribute on nodes (x, y)
        source: Source node ID
        target: Target node ID
        weight: Edge attribute for distance
        
    Returns:
        (path, distance, expanded_nodes)
    """
    pos = {n: G.nodes[n].get("pos", (0, 0)) for n in G.nodes()}
    h_func = _euclidean_heuristic(pos, target)

    g_score = {source: 0.0}
    f_score = {source: h_func(source)}
    prev = {source: None}
    pq = [(f_score[source], source)]
    expanded = set()

    while pq:
        _, u = heapq.heappop(pq)
        if u in expanded:
            continue
        expanded.add(u)
        if u == target:
            break

        for _, v, data in G.edges(u, data=True):
            w = data.get(weight, 1.0)
            tentative_g = g_score[u] + w
            if v not in g_score or tentative_g < g_score[v]:
                g_score[v] = tentative_g
                f_score[v] = tentative_g + h_func(v)
                prev[v] = u
                heapq.heappush(pq, (f_score[v], v))

    if target not in prev:
        return None, float("inf"), len(expanded)

    path = []
    v = target
    while v is not None:
        path.append(v)
        v = prev[v]
    path.reverse()

    return path, g_score[target], len(expanded)


def compare_algorithms(G, od_pairs):
    """Compare Dijkstra and A* on all source-destination pairs.
    
    Args:
        G: NetworkX DiGraph
        od_pairs: List of (source, target, description)
        
    Returns:
        List of dicts with comparison results
    """
    results = []
    for src, tgt, desc in od_pairs:
        t0 = time.perf_counter()
        path_d, dist_d, exp_d = dijkstra(G, src, tgt)
        t_d = time.perf_counter() - t0

        t0 = time.perf_counter()
        path_a, dist_a, exp_a = astar(G, src, tgt)
        t_a = time.perf_counter() - t0

        results.append({
            "origin": src,
            "destination": tgt,
            "description": desc,
            "dijkstra_path": path_d,
            "dijkstra_distance": dist_d,
            "dijkstra_expanded": exp_d,
            "dijkstra_time_ms": round(t_d * 1000, 4),
            "astar_path": path_a,
            "astar_distance": dist_a,
            "astar_expanded": exp_a,
            "astar_time_ms": round(t_a * 1000, 4),
            "paths_match": path_d == path_a if path_d and path_a else False,
        })
    return results


def print_comparison(results):
    """Return formatted comparison table string."""
    lines = []
    lines.append("=" * 120)
    lines.append(f"{'OD Pair':<35} {'Dijkstra':>30} {'A*':>30} {'Match':>8}")
    lines.append("-" * 120)
    for r in results:
        label = f"{r['origin']} -> {r['destination']} ({r['description']})"
        d_str = f"d={r['dijkstra_distance']:.1f} e={r['dijkstra_expanded']} t={r['dijkstra_time_ms']:.2f}ms"
        a_str = f"d={r['astar_distance']:.1f} e={r['astar_expanded']} t={r['astar_time_ms']:.2f}ms"
        match = "YES" if r.get("paths_match") else "NO"
        lines.append(f"{label:<35} {d_str:>30} {a_str:>30} {match:>8}")
    lines.append("=" * 120)

    for r in results:
        lines.append(f"\nOD: {r['origin']} -> {r['destination']} ({r['description']})")
        lines.append(f"  Dijkstra path: {' -> '.join(r['dijkstra_path'])}")
        lines.append(f"  A* path:       {' -> '.join(r['astar_path'])}")
        lines.append(f"  Dijkstra: dist={r['dijkstra_distance']:.2f}  expanded={r['dijkstra_expanded']}  time={r['dijkstra_time_ms']:.4f}ms")
        lines.append(f"  A*:       dist={r['astar_distance']:.2f}  expanded={r['astar_expanded']}  time={r['astar_time_ms']:.4f}ms")

    return "\n".join(lines)
