# MILP optimization module for conveyor belt modernization.
# Author: Ceron Ordonez, J. J., M.Sc (c) in Electrical Engineering
# AI Assistant: DeepSeek V4 Flash
#
# Formulates and solves the binary optimization problem:
#   Minimize sum(Q_a * Te_a)
#   Subject to:
#     sum(c_a * y_a) <= B0
#     Te_a = T_a0 - DeltaT_a * y_a
#     y_a in {0, 1}

import pulp as pl
import pandas as pd


def solve_optimization(edges_df, budget=None):
    """Solve the MILP for conveyor modernization.
    
    Args:
        edges_df: DataFrame with columns:
            edge_id, source, target, length, capacity, flow,
            cost, travel_time (T_a0), travel_time_modernized (T_a1)
        budget: Budget constraint. If None uses B0 = 0.3 * sum(c_a)
        
    Returns:
        (model, results_df, summary_dict)
    """
    df = edges_df.copy()
    total_cost = df["cost"].sum()
    if budget is None:
        budget = 0.3 * total_cost

    B0 = budget

    # Problem
    prob = pl.LpProblem("ConveyorModernization", pl.LpMinimize)

    # Decision variables
    y = {
        row["edge_id"]: pl.LpVariable(f"y_{row['edge_id']}", cat=pl.LpBinary)
        for _, row in df.iterrows()
    }

    # Te_a = T_a0 - DeltaT_a * y_a
    Te = {}
    for _, row in df.iterrows():
        eid = row["edge_id"]
        T0 = row["travel_time"]
        T1 = row["travel_time_modernized"]
        DeltaT = T0 - T1
        Te[eid] = T0 - DeltaT * y[eid]

    # Objective: minimize sum(Q_a * Te_a)
    prob += pl.lpSum(df.loc[df["edge_id"] == eid, "flow"].values[0] * Te[eid]
                     for eid in df["edge_id"]), "TotalCost"

    # Budget constraint
    prob += pl.lpSum(df.loc[df["edge_id"] == eid, "cost"].values[0] * y[eid]
                     for eid in df["edge_id"]) <= B0, "Budget"

    # Solve
    prob.solve(pl.PULP_CBC_CMD(msg=False))

    # Extract results
    results = df.copy()
    results["selected"] = results["edge_id"].map(lambda eid: int(pl.value(y[eid])))
    results["travel_time_effective"] = results.apply(
        lambda r: r["travel_time"] if r["selected"] == 0 else r["travel_time_modernized"], axis=1
    )
    results["cost_incurred"] = results["selected"] * results["cost"]

    indicator_initial = (results["flow"] * results["travel_time"]).sum()
    indicator_final = (results["flow"] * results["travel_time_effective"]).sum()

    summary = {
        "budget_available": round(B0, 2),
        "budget_used": round(results["cost_incurred"].sum(), 2),
        "total_conveyor_cost": round(total_cost, 2),
        "conveyors_selected": int(results["selected"].sum()),
        "conveyors_total": len(results),
        "indicator_initial": round(indicator_initial, 4),
        "indicator_final": round(indicator_final, 4),
        "absolute_reduction": round(indicator_initial - indicator_final, 4),
        "percentage_reduction": round(
            (indicator_initial - indicator_final) / indicator_initial * 100, 2
        ) if indicator_initial > 0 else 0,
        "status": pl.LpStatus[prob.status],
    }

    return prob, results, summary


def print_optimization_results(summary, results_df):
    """Return formatted optimization results string."""
    lines = []
    lines.append("=" * 60)
    lines.append("MILP OPTIMIZATION RESULTS")
    lines.append("=" * 60)
    lines.append(f"Solver status: {summary['status']}")
    lines.append(f"Total conveyor cost:       {summary['total_conveyor_cost']:>12.2f}")
    lines.append(f"Budget available (B0):     {summary['budget_available']:>12.2f}")
    lines.append(f"Budget used:               {summary['budget_used']:>12.2f}")
    lines.append(f"Conveyors selected:        {summary['conveyors_selected']:>12} / {summary['conveyors_total']}")
    lines.append(f"Indicator initial (I0):    {summary['indicator_initial']:>12.4f}")
    lines.append(f"Indicator final (I):       {summary['indicator_final']:>12.4f}")
    lines.append(f"Absolute reduction:        {summary['absolute_reduction']:>12.4f}")
    lines.append(f"Percentage reduction:      {summary['percentage_reduction']:>12.2f}%")

    selected = results_df[results_df["selected"] == 1]
    if not selected.empty:
        lines.append("\n--- Selected Conveyors ---")
        lines.append(f"{'Edge':<15} {'Source':>8} {'Target':>8} {'Cost':>10} {'T0':>8} {'T1':>8}")
        lines.append("-" * 60)
        for _, r in selected.iterrows():
            lines.append(f"{r['edge_id']:<15} {r['source']:>8} {r['target']:>8} "
                         f"{r['cost']:>10.1f} {r['travel_time']:>8.4f} {r['travel_time_modernized']:>8.4f}")

    lines.append("=" * 60)
    return "\n".join(lines)


def prepare_edge_data(G):
    """Extract edge data from a NetworkX graph and compute travel times.
    
    The modernization scenario assumes C'_a = 1.3 * C_a, which reduces
    utilization and thus increases velocity (shorter travel time).
    """
    import numpy as np
    from src.newton_raphson import newton_raphson

    records = []
    for u, v, d in G.edges(data=True):
        L = d["length"]
        C = d["capacity"]
        Q = d["flow"]
        c_a = d["cost"]

        rho = Q / C
        vel, _, _ = newton_raphson(rho, 1.0)
        T0 = L / vel if vel else float("inf")

        # Modernized: C' = 1.3 * C
        rho_mod = Q / (1.3 * C)
        vel_mod, _, _ = newton_raphson(rho_mod, 1.0)
        T1 = L / vel_mod if vel_mod else float("inf")

        records.append({
            "edge_id": f"({u}->{v})",
            "source": u,
            "target": v,
            "length": L,
            "capacity": C,
            "flow": Q,
            "cost": c_a,
            "travel_time": round(T0, 6),
            "travel_time_modernized": round(T1, 6),
        })

    return pd.DataFrame(records)
