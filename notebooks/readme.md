# Portfolio Optimization Notebooks

This directory contains Jupyter notebooks that demonstrate GPU-accelerated portfolio optimization techniques using NVIDIA technologies.

## Getting Started

If you haven't already set up your Jupyter kernel, please refer to the [Getting Started](../README.md#installation-on-pytorch-container) section in the main README.

## Recommended Order

### 1. Start Here: Basic Workflow

**[`cvar_basic.ipynb`](cvar_basic.ipynb)** - Complete Portfolio Optimization Walkthrough

This is the **recommended starting point** for understanding the portfolio optimization workflow. This notebook provides a comprehensive, step-by-step walkthrough covering:

- Data preparation and preprocessing
- Scenario generation
- Mean-CVaR (Conditional Value-at-Risk) portfolio optimization
- Implementing real-world constraints (concentration limits, leverage, turnover)
- Portfolio construction and analysis
- Performance evaluation and backtesting

**Start with this notebook** to build a solid foundation before exploring advanced topics.

### 2. Advanced Workflows

Once you're comfortable with the basic workflow, explore these advanced topics in any order:

#### [`efficient_frontier.ipynb`](efficient_frontier.ipynb) - Efficient Frontier Analysis

This notebook demonstrates how to:
- Generate the efficient frontier by solving multiple optimization problems
- Visualize the risk-return tradeoff across different portfolio configurations
- Compare portfolios along the efficient frontier
- Leverage GPU acceleration to quickly compute multiple optimal portfolios

#### [`rebalancing_strategies.ipynb`](rebalancing_strategies.ipynb) - Dynamic Portfolio Rebalancing

This notebook introduces dynamic portfolio management techniques:
- Time-series backtesting framework
- Testing various rebalancing strategies (periodic, threshold-based, etc.)
- Evaluating the impact of transaction costs on portfolio performance
- Analyzing strategy performance over different market conditions
- Comparing multiple rebalancing approaches

## Additional Resources

For questions or issues, please visit:
- [GitHub Issues](https://github.com/NVIDIA-AI-Blueprints/quantitative-portfolio-optimization/issues)
- [GitHub Discussions](https://github.com/NVIDIA-AI-Blueprints/quantitative-portfolio-optimization/discussions)

