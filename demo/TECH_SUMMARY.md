# cuFOLIO — Unlocking Real-Time Backtesting

## Demo Details

Interactive Streamlit demo showcasing GPU-accelerated portfolio optimization and dynamic rebalancing backtesting. Compares NVIDIA cuOpt GPU solver against open-source CPU solvers side-by-side in real time with progressive plot updates.

Two demo apps:
- **Portfolio Optimizer** (`cvar_streamlit_app.py`) — single-period Mean-CVaR or Mean-Variance optimization
- **Rebalancing Strategies** (`rebalancing_streamlit_app.py`) — multi-period dynamic rebalancing with trigger-based re-optimization

---

## NVIDIA Technologies

- **NVIDIA cuOpt** — GPU-accelerated LP/QP/MILP solver for portfolio optimization
- **NVIDIA cuML** — GPU-accelerated Kernel Density Estimation (KDE) for scenario generation
- **CUDA 12 / 13** — GPU compute backend

## 3rd Party Technologies / Applications

- **CVXPY** — convex optimization modeling framework (CPU solvers: HiGHS, CLARABEL)
- **Streamlit** — interactive web application framework
- **Plotly** — interactive charts (rebalancing backtest, dataset visualization)
- **squarify** — treemap visualization for live portfolio heatmap
- **scikit-learn** — CPU-based KDE (fallback when GPU unavailable)
- **matplotlib / seaborn** — animation frame rendering

---

## Target Audience

- Portfolio managers exploring systematic rebalancing strategies
- Quantitative analysts benchmarking optimization solvers
- Financial engineers evaluating GPU acceleration for risk management
- DLI workshop participants learning GPU-accelerated portfolio optimization

---

## Setup & Installation

### Required Equipment & Software

- 1x workstation or cloud instance with NVIDIA GPU (B200, H100, A100, or similar)
- NVIDIA Driver 550+ with CUDA 12.x or 13.x
- Python 3.10+
- ~2 GB disk space (code + datasets)
- Modern web browser (Chrome, Firefox, Edge)

For CPU-only demo (no GPU required):
- Any machine with Python 3.10+
- GPU columns will show "GPU solver not available"

### File Location

```
quantitative-portfolio-optimization/
├── demo/
│   ├── cvar_streamlit_app.py          # Portfolio optimizer app
│   ├── rebalancing_streamlit_app.py   # Rebalancing strategies app
│   ├── app_parameters.py             # Shared UI parameters
│   ├── requirements.txt              # Demo-specific dependencies
│   └── diagrams/                     # Images, GIFs, QR codes
├── src/                              # cuFOLIO core library
├── data/stock_data/                  # Stock price CSV datasets
├── notebooks/                        # Jupyter notebooks
├── .streamlit/config.toml            # NVIDIA dark theme + developer toolbar
└── pyproject.toml                    # Package config
```

### Installation

```bash
# Clone the repository
git clone https://github.com/NVIDIA-AI-Blueprints/quantitative-portfolio-optimization.git
cd quantitative-portfolio-optimization

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install core package
pip install -e .

# Install GPU support
pip install -e ".[cuda12]"    # For CUDA 12
# or
pip install -e ".[cuda13]"    # For CUDA 13

# Install demo dependencies
pip install -r demo/requirements.txt
```

### Startup

```bash
source .venv/bin/activate

# Rebalancing Strategies (main demo)
streamlit run demo/rebalancing_streamlit_app.py

# Portfolio Optimizer
streamlit run demo/cvar_streamlit_app.py
```

App opens at `http://localhost:8501` by default.

---

## Performance Tuning

- **Scenario count**: 5,000–10,000 for live demos (fast); 20,000+ for accuracy
- **Dataset size**: dow30 (~30 assets) for quick demos; sp500 (~400 assets) for GPU speedup showcase
- **Look-forward window**: 21 days (monthly) for moderate demo length; 63 days (quarterly) for shorter runs
- **CPU solver**: HiGHS (fastest CPU baseline) or CLARABEL (interior point)

Typical solve times (sp500, 10k scenarios, B200 GPU):
| Solver | Single Solve | 30-Portfolio Frontier |
|--------|-------------|----------------------|
| GPU (cuOpt) | ~0.4s | ~12s |
| CPU (HiGHS) | ~17s | ~8 min |

---

## Control Summary

**Sidebar controls:**
- Dataset selector (masked as Dataset 1, 2, … by default)
- Date range (start/end, constrained to dataset availability)
- Portfolio allocation range slider (min/max weight)
- Cash reserve range slider
- Max leverage slider
- Risk sensitivity slider (default 1.2)
- Tail-risk confidence slider
- Simulation count slider (5k–20k)
- Rebalancing trigger selector (loss threshold, drift, drawdown, buy & hold)
- Heatmap display — portfolio notional ($) for dollar amounts in the live treemap
- CPU solver selector (always masked as CPU Solver 1, CPU Solver 2)
- Advanced mode toggle (return type, threshold, windows, transaction costs, constraints)

**Main panel tabs:**
- Overview — intro + animated GIF demo
- Dataset — interactive normalized price chart and returns chart (Plotly, dark theme)
- Live Demo — GPU vs CPU side-by-side progressive results with live portfolio heatmap
- Architecture — pipeline diagram
- Benchmarks — B200 performance chart
- References — GTC workshop QR codes + academic citations

**Live demo features:**
- Smooth day-by-day animation using matplotlib static frames (no flashing)
- Both buy & hold and rebalancing lines progress in sync
- Each solver independently switches to interactive Plotly chart when done
- Live treemap heatmap showing portfolio composition updated every rebalancing period
- NVIDIA brand color palette: green gradient for long positions, red for shorts, gold for cash
- Post-run detailed results with date selector to browse portfolio heatmaps per period

**Name masking:**
- Dataset and ticker names are masked by default to avoid specific financial suggestions
- Solver names (CPU Solver 1, CPU Solver 2) are always masked
- To unmask dataset and ticker names, append `?mask=false` to the URL
- To re-enable masking, use `?mask=true` or remove the parameter
- The "About" entry in the upper-right hamburger menu documents this toggle

---

## Connection Diagram

```
[Browser] <--HTTP--> [Streamlit Server (port 8501)]
                          |
              +-----------+-----------+
              |                       |
        [GPU Thread]           [CPU Subprocess]
        cuOpt + cuML KDE      HiGHS/CLARABEL + sklearn KDE
        (main process)        (spawn-isolated process)
```

---

## Dimensions / Weight

Software-only demo. No hardware props required.

For portable demo setups:
- Laptop with NVIDIA RTX GPU (e.g. RTX 4090 Laptop) is sufficient
- External display recommended for conference presentations (1080p+)
