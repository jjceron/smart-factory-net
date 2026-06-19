# Network generation module for the Smart Factory Transport Network.
# Author: Ceron Ordonez, J. J., M.Sc (c) in Electrical Engineering
# AI Assistant: DeepSeek V4 Flash
#
# This module provides functions to:
# 1. Generate the manual base network (16 nodes, 4 zones, ~30 edges, 5 OD pairs)
# 2. Generate random strongly connected networks for complexity analysis

import numpy as np
import networkx as nx
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

BASE_NODES = [
    ("R01", "Receiving Dock",      0, 10, 10),
    ("R02", "Raw Storage",         0,  5, 25),
    ("R03", "Warehouse",           0,  5, 40),
    ("P04", "Cutting",             1, 20, 10),
    ("P05", "Machining-1",         1, 20, 25),
    ("P06", "Machining-2",         1, 20, 40),
    ("P07", "Sub-Assembly",        1, 35, 15),
    ("P08", "Assembly-1",          1, 35, 30),
    ("P09", "Assembly-2",          1, 35, 45),
    ("P10", "Heat Treatment",      1, 50, 10),
    ("Q11", "Inspection",          2, 50, 25),
    ("Q12", "Rework",              2, 50, 40),
    ("Q13", "Testing Lab",         2, 65, 20),
    ("F14", "Packaging",           3, 65, 35),
    ("F15", "FG Storage",          3, 80, 25),
    ("F16", "Shipping",            3, 80, 40),
]

ZONE_NAMES = {0: "Raw Materials", 1: "Production", 2: "Quality", 3: "Finished Goods"}

BASE_EDGES = [
    ("R01", "R02", 18, 45, 30),
    ("R01", "R03", 32, 50, 35),
    ("R02", "R03", 16, 35, 25),
    ("R03", "P04", 34, 60, 42),
    ("R02", "P04", 22, 55, 38),
    ("P04", "P05", 16, 50, 40),
    ("P04", "P06", 32, 45, 35),
    ("P05", "P06", 16, 40, 30),
    ("P05", "P07", 20, 55, 40),
    ("P06", "P07", 30, 50, 35),
    ("P06", "P09", 18, 45, 30),
    ("P07", "P08", 16, 60, 48),
    ("P08", "P09", 16, 55, 42),
    ("P08", "Q11", 18, 50, 38),
    ("P09", "Q11", 26, 50, 40),
    ("P04", "P10", 32, 35, 20),
    ("P10", "P07", 18, 40, 25),
    ("Q11", "Q12", 16, 45, 30),
    ("Q11", "Q13", 18, 40, 28),
    ("Q12", "Q13", 26, 35, 22),
    ("Q12", "P05", 34, 30, 18),
    ("Q12", "P04", 43, 25, 12),
    ("Q13", "F14", 16, 55, 45),
    ("F14", "F15", 20, 50, 40),
    ("F14", "F16", 18, 55, 45),
    ("F15", "F16", 16, 50, 42),
    ("R03", "Q11", 48, 30, 15),
    ("Q13", "P06", 50, 25, 10),
    ("F16", "R01", 78, 40, 20),
    ("P10", "F14", 30, 30, 15),
]

BASE_OD_PAIRS = [
    ("R01", "F16", "Full product flow"),
    ("P04", "Q11", "Cutting to inspection"),
    ("P06", "P08", "Machining to assembly"),
    ("Q12", "P05", "Rework to machining"),
    ("R03", "P04", "Warehouse to production"),
]


def _euclidean(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def build_base_network():
    """Build the manually-designed base factory network graph."""
    pos = {}
    G = nx.DiGraph()
    for nid, name, zid, x, y in BASE_NODES:
        G.add_node(nid, name=name, zone=zid, zone_name=ZONE_NAMES[zid], pos=(float(x), float(y)))
        pos[nid] = (float(x), float(y))

    for u, v, length, cap, flow in BASE_EDGES:
        dist = _euclidean(pos[u], pos[v])
        if not (10 <= length <= 80):
            raise ValueError(f"({u}->{v}) length {length} out of [10,80]")
        if length < dist - 1e-9:
            raise ValueError(f"({u}->{v}) length {length} < Euclidean dist {dist:.1f}")
        if not (20 <= cap <= 100):
            raise ValueError(f"({u}->{v}) capacity {cap} out of [20,100]")
        if not (0.3 * cap <= flow <= 0.85 * cap):
            raise ValueError(f"({u}->{v}) flow {flow} not in [0.3C,0.85C]")
        cost = 50 * length + 20 * cap
        G.add_edge(u, v, length=length, capacity=cap, flow=flow, cost=cost)

    return G, BASE_OD_PAIRS


def save_network(G, filename="network_base.json"):
    """Save graph to JSON file in data/."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data = nx.node_link_data(G)
    path = DATA_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def load_network(filename="network_base.json"):
    """Load graph from JSON file in data/."""
    path = DATA_DIR / filename
    with open(path) as f:
        data = json.load(f)
    return nx.node_link_graph(data)


def generate_random_network(n_nodes, seed=42):
    """Generate a random strongly connected directed graph for scaling analysis.

    Creates a Hamiltonian cycle to guarantee strong connectivity, then
    adds extra edges to reach approximately 2 * |V| edges.

    Returns:
        G: NetworkX DiGraph with same edge attributes as the base network.
    """
    rng = np.random.default_rng(seed)
    G = nx.DiGraph()

    for i in range(n_nodes):
        nid = f"N{i:04d}"
        G.add_node(nid, name=nid, zone=0, zone_name="Random",
                   pos=(float(rng.uniform(0, 100)), float(rng.uniform(0, 100))))

    nodes = list(G.nodes())
    pos = nx.get_node_attributes(G, "pos")
    target_edges = 2 * n_nodes

    shuffled = nodes.copy()
    rng.shuffle(shuffled)
    for i in range(n_nodes):
        u = shuffled[i]
        v = shuffled[(i + 1) % n_nodes]
        dist = _euclidean(pos[u], pos[v])
        L = max(10, min(80, round(dist + float(rng.uniform(1, 10)))))
        C = int(rng.integers(20, 101))
        Q = round(C * float(rng.uniform(0.3, 0.85)), 1)
        G.add_edge(u, v, length=L, capacity=C, flow=Q, cost=50 * L + 20 * C)

    attempts = 0
    while G.number_of_edges() < target_edges and attempts < target_edges * 5:
        attempts += 1
        u, v = rng.choice(nodes, size=2, replace=False)
        if G.has_edge(u, v):
            continue
        dist = _euclidean(pos[u], pos[v])
        L = max(10, min(80, round(dist + float(rng.uniform(1, 10)))))
        C = int(rng.integers(20, 101))
        Q = round(C * float(rng.uniform(0.3, 0.85)), 1)
        G.add_edge(u, v, length=L, capacity=C, flow=Q, cost=50 * L + 20 * C)

    return G
