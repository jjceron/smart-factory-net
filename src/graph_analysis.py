# Graph analysis module — connectivity and strong components.
# Author: Ceron Ordonez, J. J., M.Sc (c) in Electrical Engineering
# AI Assistant: DeepSeek V4 Flash
#
# Implements Kosaraju's algorithm for strongly connected components
# and provides connectivity reports.

import networkx as nx


def is_strongly_connected(G):
    """Check if the directed graph is strongly connected."""
    if G.number_of_nodes() == 0:
        return False
    return nx.is_strongly_connected(G)


def find_scc(G):
    """Find all strongly connected components.
    
    Returns:
        List of sets, each containing node IDs of one SCC.
    """
    return list(nx.strongly_connected_components(G))


def connectivity_report(G):
    """Generate a detailed connectivity analysis report string."""
    lines = []
    lines.append("=" * 60)
    lines.append("NETWORK CONNECTIVITY ANALYSIS")
    lines.append("=" * 60)
    lines.append(f"Nodes: {G.number_of_nodes()}")
    lines.append(f"Edges: {G.number_of_edges()}")

    strongly_conn = is_strongly_connected(G)
    lines.append(f"Strongly connected: {'YES' if strongly_conn else 'NO'}")

    scc_list = find_scc(G)
    lines.append(f"Number of SCCs: {len(scc_list)}")

    if not strongly_conn:
        lines.append("\n--- Strongly Connected Components ---")
        for i, comp in enumerate(scc_list, 1):
            names = {n: G.nodes[n].get("name", n) for n in comp}
            lines.append(f"  SCC {i}: {len(comp)} nodes — {names}")

        condensation = nx.condensation(G)
        lines.append(f"\nCondensation nodes (meta-SCCs): {condensation.number_of_nodes()}")
        lines.append(f"Condensation edges: {condensation.number_of_edges()}")

        sources = [n for n in condensation.nodes() if condensation.in_degree(n) == 0]
        sinks = [n for n in condensation.nodes() if condensation.out_degree(n) == 0]
        lines.append(f"Source SCCs (no incoming): {len(sources)}")
        lines.append(f"Sink SCCs (no outgoing): {len(sinks)}")

        lines.append("\n--- Operational Consequences ---")
        lines.append("  - Material cannot flow from sink SCCs to source SCCs")
        lines.append("  - Some stations may be unreachable from receiving dock")
        lines.append("  - Rework/return paths may be missing")
        lines.append("\n--- Suggested Corrections ---")
        lines.append("  - Add return edges from sink to source SCCs")
        lines.append("  - Ensure at least one cycle covers all production zones")

    lines.append("=" * 60)
    return "\n".join(lines)
