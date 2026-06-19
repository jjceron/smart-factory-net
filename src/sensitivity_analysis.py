# Sensitivity analysis module — budget vs. performance.
# Author: Ceron Ordonez, J. J., M.Sc (c) in Electrical Engineering
# AI Assistant: DeepSeek V4 Flash
#
# Evaluates the MILP model under increasing budget levels from 10% to 100%
# of total modernization cost, producing data for the diminishing returns chart.

import pandas as pd
from src.optimization_model import solve_optimization


def run_sensitivity(edges_df):
    """Run sensitivity analysis across budget levels.
    
    Evaluates budgets at {0.1, 0.2, ..., 1.0} * total_cost.
    
    Args:
        edges_df: DataFrame with edge data (same as optimization_model)
        
    Returns:
        DataFrame with one row per budget level containing:
        - budget_fraction, budget_available, budget_used, conveyors_selected,
          indicator_final, absolute_reduction, percentage_reduction
    """
    total_cost = edges_df["cost"].sum()
    fractions = [round(0.1 * i, 1) for i in range(1, 11)]

    records = []
    for frac in fractions:
        budget = frac * total_cost
        _, _, summary = solve_optimization(edges_df, budget=budget)
        records.append({
            "budget_fraction": frac,
            "budget_available": summary["budget_available"],
            "budget_used": summary["budget_used"],
            "conveyors_selected": summary["conveyors_selected"],
            "indicator_initial": summary["indicator_initial"],
            "indicator_final": summary["indicator_final"],
            "absolute_reduction": summary["absolute_reduction"],
            "percentage_reduction": summary["percentage_reduction"],
        })

    return pd.DataFrame(records)


def print_sensitivity_results(df):
    """Return formatted sensitivity table string."""
    lines = []
    lines.append("=" * 100)
    lines.append("SENSITIVITY ANALYSIS — Budget vs. Performance")
    lines.append("=" * 100)
    lines.append(f"{'Budget %':>10} {'Budget':>12} {'Used':>12} {'Selected':>10} "
                 f"{'I_final':>12} {'Reduction':>12} {'Red %':>10}")
    lines.append("-" * 100)
    for _, r in df.iterrows():
        lines.append(
            f"{r['budget_fraction']*100:>9.0f}% {r['budget_available']:>12.2f} "
            f"{r['budget_used']:>12.2f} {r['conveyors_selected']:>10} "
            f"{r['indicator_final']:>12.4f} {r['absolute_reduction']:>12.4f} "
            f"{r['percentage_reduction']:>9.2f}%"
        )
    lines.append("=" * 100)
    return "\n".join(lines)
