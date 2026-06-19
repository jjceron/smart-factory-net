# Smart Factory Transport Network Optimization

Analysis and optimization of an internal transport network in a manufacturing plant using graph theory, numerical methods, and MILP optimization.

## Materials

- **Base network**: 16 stations, 30 directed conveyors, 4 functional zones (Raw Materials, Production, Quality, Finished Goods), 5 origin-destination pairs
- **Constraints**: $L_a \in [10,80]$, $C_a \in [20,100]$, $Q_a \in [0.3C_a, 0.85C_a]$, $L_a \geq \|p_i-p_j\|$, cost $c_a = 50L_a + 20C_a$

## Methods

1. **Graph analysis**: Strong connectivity via Kosaraju's algorithm
2. **Shortest paths**: Dijkstra and A* (Euclidean heuristic) — from scratch
3. **Complexity scaling**: Random networks with $|V| = \{20,40,80,160,320\}$, $|E| \approx 2|V|$
4. **Velocity estimation**: Newton-Raphson solving $v + 0.4\ln(1+v) - 3(1-\rho^2) = 0$, $\rho = Q/C$
5. **MILP optimization**: Binary conveyor modernization $y_a \in \{0,1\}$ minimizing $\sum Q_a T_{e,a}$ s.t. $\sum c_a y_a \leq B_0$
6. **Sensitivity**: 10 budget levels (10%–100% of total cost)

## Technologies

- **Language**: Python 3.11
- **Libraries**: NetworkX, NumPy, SciPy, Pandas, Matplotlib, PuLP (CBC solver)
- **Testing**: pytest (42 tests)

## How to Run

```bash
# Setup
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt

# Full pipeline
python main.py

# Run tests
python -m pytest tests/ -v
```

Output directories: `data/` (network JSON), `results/` (tables), `figures/` (8 PNG charts).

## Author

Cerón Ordoñez, J. J., M.Sc (c) in Electrical Engineering
AI Assistant: DeepSeek V4 Flash
