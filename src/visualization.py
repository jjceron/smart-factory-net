# Visualization module — network graphs and result plots.
# Author: Ceron Ordonez, J. J., M.Sc (c) in Electrical Engineering
# AI Assistant: DeepSeek V4 Flash

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pathlib import Path

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"


def plot_network(G, filename="network_base.png", title="Factory Transport Network"):
    """Plot the directed graph with zone-based coloring and edge labels."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pos = {n: G.nodes[n]["pos"] for n in G.nodes()}
    zones = {n: G.nodes[n]["zone"] for n in G.nodes()}
    zone_colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c", 3: "#d62728"}
    node_colors = [zone_colors[zones[n]] for n in G.nodes()]

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors, edgecolors="black", ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

    # Draw edges with arrows
    nx.draw_networkx_edges(G, pos, arrowstyle="-|>", arrowsize=15,
                           edge_color="gray", width=1.5, ax=ax)

    # Edge labels: length
    edge_labels = {(u, v): f"L={d['length']}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6, ax=ax)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=c, label=name) for name, c in [
            ("Raw Materials", "#1f77b4"), ("Production", "#ff7f0e"),
            ("Quality", "#2ca02c"), ("Finished Goods", "#d62728")
        ]
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    ax.set_title(title, fontsize=14)
    ax.axis("off")
    plt.tight_layout()
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_complexity(df, filename="complexity_scaling.png"):
    """Plot execution time and expanded nodes vs network size."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for algo, color, marker in [("Dijkstra", "#1f77b4", "o"), ("A*", "#ff7f0e", "s")]:
        sub = df[df["algorithm"] == algo]
        axes[0].plot(sub["size"], sub["time_ms"], marker=marker, color=color,
                     label=f"{algo} O((V+E)logV)", linewidth=2)
        axes[1].plot(sub["size"], sub["expanded_nodes"], marker=marker, color=color,
                     label=algo, linewidth=2)

    axes[0].set_xlabel("Number of nodes |V|")
    axes[0].set_ylabel("Execution time (ms)")
    axes[0].set_title("Time vs Network Size")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Number of nodes |V|")
    axes[1].set_ylabel("Expanded nodes")
    axes[1].set_title("Expanded Nodes vs Network Size")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_sensitivity(df, filename="sensitivity.png"):
    """Plot indicator vs budget fraction to show diminishing returns."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df["budget_fraction"] * 100, df["indicator_final"],
            marker="o", color="#1f77b4", linewidth=2, markersize=8)
    ax.plot(df["budget_fraction"] * 100, df["indicator_initial"],
            "--", color="gray", label="Initial indicator (no modernization)", linewidth=1.5)

    # Annotate
    for _, r in df.iterrows():
        ax.annotate(f"{r['percentage_reduction']:.1f}%",
                     (r["budget_fraction"] * 100, r["indicator_final"]),
                     textcoords="offset points", xytext=(0, 10),
                     fontsize=7, ha="center")

    ax.set_xlabel("Budget (% of total modernization cost)")
    ax.set_ylabel("Performance indicator I = sum(Q_a * T_a)")
    ax.set_title("Sensitivity Analysis — Budget vs Performance Indicator")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
