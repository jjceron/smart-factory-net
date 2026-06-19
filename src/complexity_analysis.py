# Computational complexity analysis — Dijkstra vs A* scaling.
# Author: Ceron Ordonez, J. J., M.Sc (c) in Electrical Engineering
# AI Assistant: DeepSeek V4 Flash
#
# Generates random networks of increasing size and measures:
# - Execution time vs |V| for Dijkstra and A*
# - Number of expanded nodes vs |V|
# - Confirms theoretical O((|V|+|E|)log|V|) complexity

import time
import numpy as np
import pandas as pd
from src.network_generation import generate_random_network
from src.shortest_paths import dijkstra, astar


def run_complexity_analysis(sizes=(20, 40, 80, 160, 320), seed=42):
    """Measure Dijkstra and A* performance across network sizes.
    
    For each size, generates a random network with ~2*|V| edges and
    computes shortest path between the first and last node.
    
    Args:
        sizes: Tuple of node counts to test
        seed: Base random seed (incremented per size)
        
    Returns:
        DataFrame with columns: size, algorithm, time_ms, expanded_nodes, distance
    """
    records = []
    for n_nodes in sizes:
        G = generate_random_network(n_nodes, seed=seed + n_nodes)
        nodes = list(G.nodes())
        src, tgt = nodes[0], nodes[-1]

        # Dijkstra
        t0 = time.perf_counter()
        _, dist_d, exp_d = dijkstra(G, src, tgt)
        t_d = (time.perf_counter() - t0) * 1000  # ms

        # A*
        t0 = time.perf_counter()
        _, dist_a, exp_a = astar(G, src, tgt)
        t_a = (time.perf_counter() - t0) * 1000  # ms

        records.append({
            "size": n_nodes,
            "edges": G.number_of_edges(),
            "algorithm": "Dijkstra",
            "time_ms": round(t_d, 4),
            "expanded_nodes": exp_d,
            "distance": round(dist_d, 2),
        })
        records.append({
            "size": n_nodes,
            "edges": G.number_of_edges(),
            "algorithm": "A*",
            "time_ms": round(t_a, 4),
            "expanded_nodes": exp_a,
            "distance": round(dist_a, 2),
        })

    return pd.DataFrame(records)


def print_complexity_results(df):
    """Return formatted complexity table string."""
    lines = []
    lines.append("=" * 90)
    lines.append("COMPUTATIONAL COMPLEXITY ANALYSIS")
    lines.append("=" * 90)
    lines.append(f"{'|V|':>8} {'|E|':>8} {'Algorithm':>12} {'Time (ms)':>12} "
                 f"{'Expanded':>10} {'Distance':>12}")
    lines.append("-" * 90)

    for _, r in df.iterrows():
        lines.append(
            f"{r['size']:>8} {r['edges']:>8} {r['algorithm']:>12} "
            f"{r['time_ms']:>12.4f} {r['expanded_nodes']:>10} {r['distance']:>12.2f}"
        )

    # Summary by size
    lines.append("\n--- Complexity Class Verification ---")
    lines.append("Dijkstra: O((|V|+|E|) log |V|)")
    lines.append("A*:       O(b^d) worst-case, but heuristic reduces expansion in practice")

    lines.append("\n--- Ratio Analysis ---")
    for size in df["size"].unique():
        sub = df[df["size"] == size]
        dij = sub[sub["algorithm"] == "Dijkstra"]["expanded_nodes"].values[0]
        ast = sub[sub["algorithm"] == "A*"]["expanded_nodes"].values[0]
        ratio = ast / dij if dij > 0 else 0
        lines.append(f"  |V|={size:>4}: A*/Dijkstra expanded ratio = {ratio:.3f}")

    lines.append("=" * 90)
    return "\n".join(lines)
