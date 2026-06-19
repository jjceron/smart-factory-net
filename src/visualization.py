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


def plot_network(G, filename="network_base.png"):
    """Plot the directed graph with zone-based coloring.
    Titles removed for manual caption in report. Larger elements."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pos = {n: G.nodes[n]["pos"] for n in G.nodes()}
    zones = {n: G.nodes[n]["zone"] for n in G.nodes()}
    names = {n: G.nodes[n]["name"] for n in G.nodes()}
    zone_colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c", 3: "#d62728"}
    node_colors = [zone_colors[zones[n]] for n in G.nodes()]

    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    nx.draw_networkx_nodes(G, pos, node_size=900, node_color=node_colors,
                           edgecolors="black", linewidths=1.5, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", ax=ax)

    nx.draw_networkx_edges(G, pos, arrowstyle="-|>", arrowsize=25,
                           edge_color="gray", width=2.0, min_source_margin=20,
                           min_target_margin=20, ax=ax)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=c, label=name, edgecolor="black")
        for name, c in [
            ("Raw Materials", "#1f77b4"), ("Production", "#ff7f0e"),
            ("Quality", "#2ca02c"), ("Finished Goods", "#d62728")
        ]
    ]
    ax.legend(handles=legend_elements, loc="upper right",
              fontsize=12, framealpha=0.9, edgecolor="black")

    ax.axis("off")
    plt.tight_layout()
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_complexity_time(df, filename="complexity_time.png"):
    """Plot execution time vs network size (single panel)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    for algo, color, marker in [("Dijkstra", "#1f77b4", "o"), ("A*", "#ff7f0e", "s")]:
        sub = df[df["algorithm"] == algo]
        ax.plot(sub["size"], sub["time_ms"], marker=marker, color=color,
                label=algo, linewidth=2, markersize=8)
        for _, r in sub.iterrows():
            ax.annotate(f"{r['time_ms']:.2f}",
                        (r["size"], r["time_ms"]),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=7, color=color)

    ax.set_xlabel("Number of nodes $|V|$", fontsize=12)
    ax.set_ylabel("Execution time (ms)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_complexity_nodes(df, filename="complexity_nodes.png"):
    """Plot expanded nodes vs network size (single panel)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    for algo, color, marker in [("Dijkstra", "#1f77b4", "o"), ("A*", "#ff7f0e", "s")]:
        sub = df[df["algorithm"] == algo]
        ax.plot(sub["size"], sub["expanded_nodes"], marker=marker, color=color,
                label=algo, linewidth=2, markersize=8)
        for _, r in sub.iterrows():
            ax.annotate(f"{r['expanded_nodes']}",
                        (r["size"], r["expanded_nodes"]),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=7, color=color)

    ax.set_xlabel("Number of nodes $|V|$", fontsize=12)
    ax.set_ylabel("Expanded nodes", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_sensitivity(df, filename="sensitivity.png"):
    """Plot indicator vs budget fraction to show diminishing returns."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df["budget_fraction"] * 100, df["indicator_final"],
            marker="o", color="#1f77b4", linewidth=2.5, markersize=10)
    ax.plot(df["budget_fraction"] * 100, df["indicator_initial"],
            "--", color="gray", label="Initial indicator $I_0$ (no modernization)",
            linewidth=2)

    for _, r in df.iterrows():
        ax.annotate(f"{r['percentage_reduction']:.1f}%",
                     (r["budget_fraction"] * 100, r["indicator_final"]),
                     textcoords="offset points", xytext=(0, 12),
                     fontsize=9, ha="center", fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                               edgecolor="gray", alpha=0.8))

    ax.set_xlabel("Budget (% of total modernization cost)", fontsize=12)
    ax.set_ylabel("Performance indicator $I = \\sum Q_a \\cdot T_a$", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_dijkstra_astar_comparison(results, filename="dijkstra_astar_comparison.png"):
    """Bar chart: expanded nodes per OD pair for Dijkstra vs A*."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))

    labels = [r["origin"] + " → " + r["destination"] for r in results]
    x = np.arange(len(labels))
    width = 0.35

    dijk_vals = [r["dijkstra_expanded"] for r in results]
    astar_vals = [r["astar_expanded"] for r in results]

    bars1 = ax.bar(x - width/2, dijk_vals, width, label="Dijkstra",
                   color="#1f77b4", edgecolor="black", linewidth=0.8)
    bars2 = ax.bar(x + width/2, astar_vals, width, label="A*",
                   color="#ff7f0e", edgecolor="black", linewidth=0.8)

    for bar, val in zip(bars1, dijk_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(val), ha="center", fontsize=9, fontweight="bold")
    for bar, val in zip(bars2, astar_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(val), ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Expanded nodes", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_velocity_vs_utilization(filename="velocity_vs_utilization.png"):
    """Plot velocity v(rho) for rho in [0.1, 0.95] using Newton-Raphson."""
    from src.newton_raphson import newton_raphson

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    rhos = np.linspace(0.1, 0.95, 100)
    velocities = []
    for rho in rhos:
        v, _, _ = newton_raphson(rho, 1.0, tol=1e-6, max_iter=100)
        velocities.append(v if v else 0)

    ax.plot(rhos, velocities, color="#2ca02c", linewidth=2.5)

    # Annotate key points
    key_rhos = [0.3, 0.5, 0.7, 0.9]
    for kr in key_rhos:
        kv, _, _ = newton_raphson(kr, 1.0, tol=1e-6)
        if kv:
            ax.plot(kr, kv, "o", color="red", markersize=6)
            ax.annotate(f"$\\rho={kr:.1f}$\n$v={kv:.2f}$",
                        (kr, kv), textcoords="offset points",
                        xytext=(10, 5), fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow",
                                  alpha=0.7))

    ax.set_xlabel("Utilization index $\\rho = Q / C$", fontsize=12)
    ax.set_ylabel("Velocity $v$ (m/s)", fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_optimization_selection(opt_df, filename="optimization_selection.png"):
    """Horizontal bar chart: cost vs benefit per conveyor, colored by zone.
    Selected conveyors are highlighted with a border."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    zone_colors = {"Raw Materials": "#1f77b4", "Production": "#ff7f0e",
                   "Quality": "#2ca02c", "Finished Goods": "#d62728"}

    df = opt_df.copy()

    zone_map = {"R01": "Raw Materials", "R02": "Raw Materials", "R03": "Raw Materials",
                "P04": "Production", "P05": "Production", "P06": "Production",
                "P07": "Production", "P08": "Production", "P09": "Production",
                "P10": "Production",
                "Q11": "Quality", "Q12": "Quality", "Q13": "Quality",
                "F14": "Finished Goods", "F15": "Finished Goods", "F16": "Finished Goods"}
    df["zone"] = df["source"].map(zone_map)

    df["benefit"] = df["flow"] * (df["travel_time"] - df["travel_time_modernized"])
    df["label"] = df["source"] + "→" + df["target"]
    df.sort_values("benefit", inplace=True)

    colors = [zone_colors[z] for z in df["zone"]]
    edgecolors = ["black" if s == 1 else "none" for s in df["selected"]]
    linewidths = [2.0 if s == 1 else 0.5 for s in df["selected"]]

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(range(len(df)), df["benefit"], color=colors,
                   edgecolor=edgecolors, linewidth=linewidths)

    # Bold label for selected conveyors
    for i, (_, row) in enumerate(df.iterrows()):
        label = row["label"]
        ax.text(row["benefit"] + 5, i, label,
                fontsize=7, va="center",
                fontweight="bold" if row["selected"] else "normal")

    ax.set_xlabel("Benefit $w_a = Q_a \\cdot \\Delta T_a$", fontsize=12)
    ax.set_ylabel("Conveyor (sorted by benefit)", fontsize=12)
    ax.set_yticks([])

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=c, label=name, edgecolor="black")
        for name, c in [
            ("Raw Materials", "#1f77b4"), ("Production", "#ff7f0e"),
            ("Quality", "#2ca02c"), ("Finished Goods", "#d62728")
        ]
    ]
    legend_elements.append(
        Patch(facecolor="white", edgecolor="black", linewidth=2,
              label="Selected (modernized)"))
    ax.legend(handles=legend_elements, fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_marginal_benefit(df, filename="marginal_benefit.png"):
    """Bar chart: marginal improvement per 10% budget increment."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    margins = []
    labels = []
    prev_pct = 0
    prev_val = df.iloc[0]["indicator_initial"]
    for _, r in df.iterrows():
        marginal = round(prev_val - r["indicator_final"], 2)
        margins.append(marginal)
        labels.append(f"{r['budget_fraction']*100:.0f}%")
        prev_val = r["indicator_final"]

    first_diffs = []
    first_labels = []
    for i in range(1, len(margins)):
        first_diffs.append(margins[i] - margins[i-1])
        first_labels.append(f"{labels[i-1]} → {labels[i]}")

    bar_colors = ["#1f77b4"] * len(first_diffs)
    # Gradient: darker for larger marginal benefit
    max_diff = max(first_diffs) if first_diffs else 1
    for i in range(len(first_diffs)):
        intensity = first_diffs[i] / max_diff
        bar_colors[i] = plt.cm.Blues(0.3 + 0.7 * intensity)

    bars = ax.bar(range(len(first_diffs)), first_diffs, color=bar_colors,
                  edgecolor="black", linewidth=1.0)

    for bar, val in zip(bars, first_diffs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f"{val:.2f}", ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(range(len(first_diffs)))
    ax.set_xticklabels(first_labels, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Marginal improvement in $I$", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path
