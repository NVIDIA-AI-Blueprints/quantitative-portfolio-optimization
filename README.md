# Quantitative Portfolio Optimization developer example

## Overview

Quantitative Portfolio Optimization developer example is a GPU-accelerated solution designed to enable fast, scalable, and real-time portfolio optimization for financial institutions. Leveraging NVIDIA® cuOpt™ and RAPIDS, this developer example delivers near-real-time solutions for large-scale Mean-CVaR portfolio optimization problems, allowing enterprises to model advanced risk measures and optimize complex portfolios in accelerated time.

Portfolio Optimization (PO) involves solving high-dimensional, non-linear numerical optimization problems that balance risk and return to meet investment goals. By introducing GPU-accelerated building blocks, this project significantly reduces computation times while improving solution quality, making sophisticated portfolio optimization accessible and practical for large-scale applications.

<p align="center">
    <img src="./docs/arch_diagram.png" alt="architecture diagram for PO" width="750"/>
</p>

### Key Features

- **End-to-End GPU Workflow**: Accelerates the portfolio allocation problem using NVIDIA® cuOpt™, delivering near-real-time optimization.
- **Flexible Financial Model Building**: Fully customizable constraints, including CVaR, leverage, budgets, turnover, and cardinality limits.
- **Data-driven Risk Modeling**: Utilizes CVaR as a risk measure that is scenario-based and requires no assumptions about the asset return distribution.
- **Full Pipeline Support**: Provides tools for performance evaluation, strategy backtesting, benchmarking, and visualization.
- **Easy Benchmarking**: Streamlines the process of benchmarking against CP-based libraries and solvers.
- **Scalable & Efficient**: Excels at solving large LP and MILP problems, leveraging NVIDIA libraries for pre- and post-optimization acceleration.

## Get Started

### System Requirements
<details>
<summary><b>Minimum Requirements</b></summary>

- **System Architecture**:
  - x86-64
  - ARM64
- **GPU**:
  - Volta architecture or better (Compute Capability >=7.0)
- **CPU**:
  - 4+ cores
- **System Memory**:
  - 16+ GB RAM
- **NVMe SSD Storage**:
  - 100+ GB free space
- **CUDA**:
  - 12.0+
- **Python**:
  - 3.10.x - 3.13.x
- **NVIDIA Drivers**:
  - 525.60.13+ (Linux)
  - 527.41+ (Windows)
- **OS**:
  - Linux distributions with glibc>=2.28 (released in August 2018):
    - Arch Linux (minimum version 2018-08-02)
    - Debian (minimum version 10.0)
    - Fedora (minimum version 29)
    - Linux Mint (minimum version 20)
    - Rocky Linux / Alma Linux / RHEL (minimum version 8)
    - Ubuntu (minimum version 20.04)
  - Windows 11 with WSL2
- **CUDA & NVIDIA Driver Combinations**:
  - CUDA 12.0 with Driver 525.60.13+
  - CUDA 12.2 with Driver 535.86.10+
  - CUDA 12.5 with Driver 555.42.06+
  - CUDA 12.9 with Driver 570.42.01+
  - CUDA 13.0 with Driver 580.65.06+

</details>

<details>
<summary><b>Recommended Requirements for Best Performance</b></summary>

- **System Architecture**:
  - x86-64
  - ARM64
- **GPU**:
  - NVIDIA H100 SXM (compute capability >= 9.0) and above
- **CPU**:
  - 32+ cores
- **System Memory**:
  - 64+ GB RAM
- **NVMe SSD Storage**:
  - 100+ GB free space
- **CUDA**:
  - 13.0
- **NVIDIA Drivers**:
  - Latest NVIDIA drivers (580.65.06+)
- **OS**:
  - Linux distributions with glibc>=2.28 (released in August 2018):
    - Arch Linux (minimum version 2018-08-02)
    - Debian (minimum version 10.0)
    - Fedora (minimum version 29)
    - Linux Mint (minimum version 20)
    - Rocky Linux / Alma Linux / RHEL (minimum version 8)

The above configuration will provide optimal performance for large-scale optimization problems.

</details>

<details>
<summary><b>Container Requirements</b></summary>

- **nvidia-container-toolkit** needs to be installed for Docker deployment

</details>

### Installation on PyTorch Container

To install dependencies on the NVIDIA PyTorch container:

```bash
# Start the container
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:25.08-py3

# Clone the repository
git clone https://github.com/NVIDIA-AI-Blueprints/quantitative-portfolio-optimization.git
cd quantitative-portfolio-optimization

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install with all dependencies using uv
uv sync

# Optional: Install development tools (including ipykernel)
UV_CACHE_DIR=/path/to/cache/directory uv sync --extra dev

# Install Jupyter and JupyterLab
uv pip install jupyter jupyterlab ipykernel

# Create a Jupyter kernel for this environment
uv run python -m ipykernel install --user --name=portfolio-opt --display-name "Portfolio Optimization"

# Launch Jupyter Lab
uv run jupyter lab
```

**Note:** The PyTorch container already includes CUDA and other GPU dependencies. This installation adds the optimization and ML libraries (cuOpt, cuML). The `uv sync` command automatically creates a virtual environment and installs all dependencies from `uv.lock`.

**Important Notes:**
- If you encounter "No space left on device" errors, use the `--cache-dir` flag to specify an alternate cache location with sufficient disk space (at least 10GB free recommended)
- You can also set the `UV_CACHE_DIR` environment variable instead of using the flag: `export UV_CACHE_DIR=/path/to/cache/directory`
- The project is configured to resolve dependencies only for x86_64 platforms to avoid compatibility issues

#### Using the Jupyter Kernel

After launching Jupyter Lab:
1. Navigate to the [`notebooks/`](notebooks/) directory
2. Open any notebook (e.g., `cvar_basic.ipynb`)
3. Select the "Portfolio Optimization" kernel from the kernel selector in the top-right corner
4. If the kernel is not visible, refresh the page or restart Jupyter Lab

To list all available kernels:
```bash
jupyter kernelspec list
```

To remove the kernel later (if needed):
```bash
jupyter kernelspec uninstall portfolio-opt
```

### Setting Up Data with Git LFS

This project uses Git Large File Storage (LFS) to manage large data files, including `sp500.csv` stored in the `data/` directory.

#### Installing Git LFS

If you don't have Git LFS installed, follow these steps:

**On Ubuntu/Debian:**
```bash
sudo apt-get install git-lfs
```

**On Fedora/RHEL:**
```bash
sudo dnf install git-lfs
```

**For other systems**, visit the [Git LFS website](https://git-lfs.github.com/) for installation instructions.

#### Pulling LFS Files

After installing Git LFS, initialize it and pull the data files:

```bash
# Initialize Git LFS (one-time setup)
git lfs install

# Pull all LFS files (including sp500.csv)
git lfs pull
```

The `sp500.csv` file will be downloaded to the `data/` directory and is ready to use in the notebooks.

**Note:** If you cloned the repository before installing Git LFS, you'll need to run `git lfs pull` to fetch the actual data files. Otherwise, you'll only have pointer files instead of the actual data.

### Quick Start

Explore the example notebooks in the [`notebooks/`](notebooks/) directory:
- **`cvar_basic.ipynb`**: Complete walkthrough of Mean-CVaR portfolio optimization with GPU acceleration
- **`efficient_frontier.ipynb`**: A quick tutorial on how to generate efficient frontier.
- **`rebalancing_strategies.ipynb`** Introduction to dynamic re-balancing and examples of testing strategies

## Contribution Guidelines

We welcome contributions to this project! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on:
- Code of conduct
- How to submit issues and feature requests
- Pull request process
- Coding standards and best practices

## Community

For questions, discussions, and community support:
- **Issues**: Report bugs and request features via [GitHub Issues](https://github.com/NVIDIA-AI-Blueprints/quantitative-portfolio-optimization/issues)
- **Discussions**: Join conversations in [GitHub Discussions](https://github.com/NVIDIA-AI-Blueprints/quantitative-portfolio-optimization/discussions)

## References

- [NVIDIA cuOpt Documentation](https://docs.nvidia.com/cuopt/)
- [RAPIDS cuML](https://docs.rapids.ai/api/cuml/stable/)
- Markowitz, H. (1952). "Portfolio Selection". *The Journal of Finance*, 7(1), 77-91.
- Rockafellar, R. T., & Uryasev, S. (2000). "Optimization of conditional value-at-risk". *Journal of Risk*, 2, 21-42.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
