# GPU-Accelerated Portfolio Optimization

## Overview

This project provides GPU-accelerated solutions for portfolio optimization, a computationally intensive task central to portfolio management. It leverages NVIDIA technologies including **cuOpt**, **RAPIDS**, and the **HPC SDK** to deliver substantial performance improvements over traditional CPU-based approaches.

Portfolio Optimization (PO) involves solving high-dimensional, non-linear numerical optimization problems that balance risk and return to meet investment goals. By introducing GPU-accelerated building blocks, this project significantly reduces computation times while improving solution quality, making sophisticated portfolio optimization accessible and practical for large-scale applications.

### Key Features

- **GPU-Accelerated Optimization**: Leverages NVIDIA cuOpt LP/QP solvers for dramatic speedups
- **Multiple Risk Measures**: Supports mean-variance and Conditional Value-at-Risk (CVaR) optimization
- **Real-World Constraints**: Implements concentration limits, leverage constraints, turnover limits, and cardinality constraints
- **Scenario Generation**: Uses GPU-accelerated KDE (Kernel Density Estimation) for return distribution modeling
- **Backtesting Framework**: Comprehensive tools for evaluating portfolio performance

### Why Portfolio Optimization Matters

Portfolio Management (PM) is crucial for all investment-focused institutions, including buy-side and sell-side operations across asset classes. As financial markets grow in complexity and diversity, the demand for sophisticated PM techniques increases. Currently, most institutions rely on CPUs and third-party solvers, which can be slow and costly. GPU acceleration offers a path to faster, more efficient, and more scalable portfolio optimization solutions.

### Optimization Methods

This project focuses on two prominent risk measures used in practice:

**Mean-Variance Optimization**: Pioneered by Harry Markowitz, this approach evaluates portfolios by balancing expected return against variance, providing a fundamental trade-off between risk and return.

**CVaR Optimization**: Conditional Value-at-Risk offers a more nuanced assessment of risk by focusing on the tail end of the loss distribution. Unlike Value-at-Risk (VaR), CVaR accounts for the average of losses that exceed a threshold, providing a more comprehensive measure of extreme risks. This approach is particularly valuable for portfolios with asymmetric return distributions.

### Portfolio Optimization Workflow
![Portfolio Optimization workflow](./images/po_workflow.png)

### NVIDIA Technology Stack
<p align="center">
    <img src="./images/nvidia_po_stack.png" alt="NVIDIA stack for PO" width="750"/>
</p>

## Get Started

### Prerequisites
- NVIDIA GPU with CUDA support
- Docker with NVIDIA Container Toolkit (for containerized deployment)

### Installation on PyTorch Container

To install cuFOLIO dependencies on the NVIDIA PyTorch container:

```bash
# Start the container
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:25.08-py3

# Clone the repository
git clone https://github.com/NVIDIA-AI-Blueprints/portfolio-optimization.git
cd portfolio-optimization

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install cuFOLIO with all dependencies
pip install -e .

# Optional: Install development tools
pip install -e ".[dev]"
```

**Note:** The PyTorch container already includes CUDA and other GPU dependencies. This installation adds the optimization and ML libraries (cuOpt, cuML).

### Quick Start

Explore the example notebooks in the `deploy/` directory:
- **`cvar_basic.ipynb`**: Complete walkthrough of Mean-CVaR portfolio optimization with GPU acceleration

## Contribution Guidelines

We welcome contributions to this project! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on:
- Code of conduct
- How to submit issues and feature requests
- Pull request process
- Coding standards and best practices

## Community

For questions, discussions, and community support:
- **Issues**: Report bugs and request features via [GitHub Issues](https://github.com/NVIDIA-AI-Blueprints/portfolio-optimization/issues)
- **Discussions**: Join conversations in [GitHub Discussions](https://github.com/NVIDIA-AI-Blueprints/portfolio-optimization/discussions)

## References

- [NVIDIA cuOpt Documentation](https://docs.nvidia.com/cuopt/)
- [RAPIDS cuML](https://docs.rapids.ai/api/cuml/stable/)
- Markowitz, H. (1952). "Portfolio Selection". *The Journal of Finance*, 7(1), 77-91.
- Rockafellar, R. T., & Uryasev, S. (2000). "Optimization of conditional value-at-risk". *Journal of Risk*, 2, 21-42.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.