#!/usr/bin/env python3
"""
Smart Factory Transport Network Optimization — Main Pipeline
Author: Ceron Ordonez, J. J., M.Sc (c) in Electrical Engineering
AI Assistant: DeepSeek V4 Flash

Runs the complete analysis:
  1. Build and save base network
  2. Connectivity analysis
  3. Shortest paths (Dijkstra vs A*)
  4. Complexity scaling analysis
  5. Newton-Raphson velocity estimation
  6. MILP conveyor modernization optimization
  7. Budget sensitivity analysis
  8. Generate all figures
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.network_generation import build_base_network, save_network
from src.graph_analysis import connectivity_report
from src.shortest_paths import compare_algorithms, print_comparison
from src.complexity_analysis import run_complexity_analysis, print_complexity_results
from src.newton_raphson import edges_from_graph, compute_edge_velocities, print_velocity_results
from src.optimization_model import prepare_edge_data, solve_optimization, print_optimization_results
from src.sensitivity_analysis import run_sensitivity, print_sensitivity_results
from src.visualization import (
    plot_network,
    plot_complexity_time,
    plot_complexity_nodes,
    plot_sensitivity,
    plot_dijkstra_astar_comparison,
    plot_velocity_vs_utilization,
    plot_optimization_selection,
    plot_marginal_benefit,
)

RESULTS_DIR = PROJECT_ROOT / "results"


def save_text(filename, content):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    path.write_text(content, encoding="utf-8")
    return path


def main():
    t_start = time.perf_counter()
    print("=" * 60)
    print("SMART FACTORY TRANSPORT NETWORK OPTIMIZATION")
    print("=" * 60)

    # 1. Build base network
    print("\n[1/8] Building base network...")
    G, od_pairs = build_base_network()
    net_path = save_network(G)
    print(f"      Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    print(f"      Saved to: {net_path}")

    # 2. Connectivity analysis
    print("\n[2/8] Running connectivity analysis...")
    report = connectivity_report(G)
    print(report)
    save_text("connectivity.txt", report)

    # 3. Shortest paths — Dijkstra vs A*
    print("\n[3/8] Computing shortest paths (5 OD pairs)...")
    sp_results = compare_algorithms(G, od_pairs)
    comparison = print_comparison(sp_results)
    print(comparison)
    save_text("shortest_paths.txt", comparison)

    # 4. Complexity scaling analysis
    print("\n[4/8] Running complexity scaling analysis...")
    print("      Sizes: 20, 40, 80, 160, 320")
    complexity_df = run_complexity_analysis(sizes=(20, 40, 80, 160, 320))
    complexity_text = print_complexity_results(complexity_df)
    print(complexity_text)
    save_text("complexity.txt", complexity_text)
    complexity_df.to_csv(RESULTS_DIR / "complexity.csv", index=False)
    plot_complexity_time(complexity_df)
    plot_complexity_nodes(complexity_df)
    print("      Figures saved to: figures/complexity_time.png, complexity_nodes.png")

    # 5. Newton-Raphson velocity estimation
    print("\n[5/8] Estimating conveyor velocities (Newton-Raphson)...")
    edges = edges_from_graph(G)
    vel_results = compute_edge_velocities(edges, initial_guesses=(0.2, 1.0, 5.0))
    velocity_text = print_velocity_results(vel_results)
    print(velocity_text)
    save_text("newton_raphson.txt", velocity_text)
    plot_velocity_vs_utilization()
    print("      Figure saved to: figures/velocity_vs_utilization.png")

    # 6. MILP optimization
    print("\n[6/8] Solving MILP optimization...")
    opt_df = prepare_edge_data(G)
    _, opt_results, opt_summary = solve_optimization(opt_df)
    opt_text = print_optimization_results(opt_summary, opt_results)
    print(opt_text)
    save_text("optimization.txt", opt_text)
    opt_results.to_csv(RESULTS_DIR / "optimization.csv", index=False)
    plot_optimization_selection(opt_results)
    print("      Figure saved to: figures/optimization_selection.png")

    # 7. Sensitivity analysis
    print("\n[7/8] Running budget sensitivity analysis...")
    sens_df = run_sensitivity(opt_df)
    sens_text = print_sensitivity_results(sens_df)
    print(sens_text)
    save_text("sensitivity.txt", sens_text)
    sens_df.to_csv(RESULTS_DIR / "sensitivity.csv", index=False)
    plot_sensitivity(sens_df)
    plot_marginal_benefit(sens_df)
    print("      Figures saved to: figures/sensitivity.png, marginal_benefit.png")

    # 8. Network visualization + comparison
    print("\n[8/8] Generating all figures...")
    plot_network(G)
    print("      Figure saved to: figures/network_base.png")
    plot_dijkstra_astar_comparison(sp_results)
    print("      Figure saved to: figures/dijkstra_astar_comparison.png")

    # Done
    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 60)
    print(f"PIPELINE COMPLETE -- Time: {elapsed:.2f}s")
    print("=" * 60)
    print("\nOutput files:")
    print(f"  data/     -> {net_path.name}")
    print(f"  results/  -> connectivity.txt, shortest_paths.txt, newton_raphson.txt,")
    print(f"               optimization.txt, sensitivity.txt, complexity.txt (.csv)")
    print(f"  figures/  -> network_base.png, complexity_time.png, complexity_nodes.png,")
    print(f"               sensitivity.png, marginal_benefit.png,")
    print(f"               dijkstra_astar_comparison.png, velocity_vs_utilization.png,")
    print(f"               optimization_selection.png")


if __name__ == "__main__":
    main()
