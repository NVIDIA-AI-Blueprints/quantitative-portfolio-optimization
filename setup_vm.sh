#!/bin/bash

################################################################################
# Portfolio Optimization VM Setup Script
# 
# This script sets up the environment for the Quantitative Portfolio 
# Optimization developer example on a VM after the environment is created.
#
# Usage: ./setup_vm.sh [--skip-git-lfs] [--skip-jupyter] [--dev]
################################################################################

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command line arguments
SKIP_GIT_LFS=false
SKIP_JUPYTER=false
INSTALL_DEV=false

for arg in "$@"; do
    case $arg in
        --skip-git-lfs)
            SKIP_GIT_LFS=true
            shift
            ;;
        --skip-jupyter)
            SKIP_JUPYTER=true
            shift
            ;;
        --dev)
            INSTALL_DEV=true
            shift
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: ./setup_vm.sh [--skip-git-lfs] [--skip-jupyter] [--dev]"
            exit 1
            ;;
    esac
done

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo ""
    echo -e "${GREEN}================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}================================${NC}"
}

################################################################################
# Step 1: Check system requirements
################################################################################
print_header "Step 1: Checking System Requirements"

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    print_error "This script is designed for Linux systems only"
    exit 1
fi
print_success "Running on Linux"

# Check if NVIDIA GPU is available
if ! command -v nvidia-smi &> /dev/null; then
    print_warning "nvidia-smi not found. GPU may not be available."
else
    print_info "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
    print_success "GPU check passed"
fi

# Check Python version
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
print_info "Python version: $PYTHON_VERSION"

# Verify Python version is >= 3.10
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    print_error "Python 3.10 or higher is required (found $PYTHON_VERSION)"
    exit 1
fi
print_success "Python version check passed"

################################################################################
# Step 2: Install Git LFS and pull data files
################################################################################
if [ "$SKIP_GIT_LFS" = false ]; then
    print_header "Step 2: Setting Up Git LFS"
    
    # Check if git-lfs is installed
    if ! command -v git-lfs &> /dev/null; then
        print_info "Installing Git LFS..."
        
        # Detect distribution and install accordingly
        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y git-lfs
        elif command -v dnf &> /dev/null; then
            sudo dnf install -y git-lfs
        elif command -v yum &> /dev/null; then
            sudo yum install -y git-lfs
        else
            print_error "Could not detect package manager. Please install git-lfs manually."
            print_info "Visit: https://git-lfs.github.com/"
            exit 1
        fi
        print_success "Git LFS installed"
    else
        print_success "Git LFS already installed"
    fi
    
    # Initialize Git LFS
    print_info "Initializing Git LFS..."
    git lfs install
    print_success "Git LFS initialized"
    
    # Pull LFS files
    print_info "Pulling Git LFS files (this may take a few minutes)..."
    git lfs pull
    print_success "Git LFS files pulled"
    
    # Verify data file exists
    if [ -f "data/stock_data/sp500.csv" ]; then
        FILE_SIZE=$(du -h data/stock_data/sp500.csv | cut -f1)
        print_success "Data file verified: sp500.csv ($FILE_SIZE)"
    else
        print_warning "Data file sp500.csv not found after Git LFS pull"
    fi
else
    print_header "Step 2: Skipping Git LFS Setup"
fi

################################################################################
# Step 3: Install uv package manager
################################################################################
print_header "Step 3: Installing uv Package Manager"

if ! command -v uv &> /dev/null; then
    print_info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Add uv to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"
    
    # Verify installation
    if ! command -v uv &> /dev/null; then
        print_error "uv installation failed. Please check your internet connection."
        exit 1
    fi
    print_success "uv installed successfully"
else
    print_success "uv already installed"
fi

# Display uv version
UV_VERSION=$(uv --version)
print_info "uv version: $UV_VERSION"

################################################################################
# Step 4: Install Python dependencies
################################################################################
print_header "Step 4: Installing Python Dependencies"

print_info "Installing project dependencies with uv sync..."
print_info "This may take several minutes as it downloads GPU-accelerated libraries..."

# Install main dependencies
uv sync

print_success "Main dependencies installed"

# Install development dependencies if requested
if [ "$INSTALL_DEV" = true ]; then
    print_info "Installing development dependencies..."
    uv sync --extra dev
    print_success "Development dependencies installed"
fi

################################################################################
# Step 5: Setup Jupyter (optional)
################################################################################
if [ "$SKIP_JUPYTER" = false ]; then
    print_header "Step 5: Setting Up Jupyter"
    
    print_info "Installing Jupyter and IPython kernel..."
    uv pip install jupyter jupyterlab ipykernel
    print_success "Jupyter packages installed"
    
    # Create Jupyter kernel
    print_info "Creating Jupyter kernel for portfolio optimization..."
    uv run python -m ipykernel install --user --name=portfolio-opt --display-name "Portfolio Optimization"
    print_success "Jupyter kernel 'portfolio-opt' created"
    
    # List available kernels
    print_info "Available Jupyter kernels:"
    jupyter kernelspec list
else
    print_header "Step 5: Skipping Jupyter Setup"
fi

################################################################################
# Step 6: Verify Installation
################################################################################
print_header "Step 6: Verifying Installation"

print_info "Testing Python imports..."

# Create a test script to verify imports
cat > /tmp/test_imports.py << 'EOF'
import sys
import importlib

packages = [
    'numpy',
    'pandas',
    'cvxpy',
    'sklearn',
    'seaborn',
    'cuml',
    'cuopt',
    'highspy'
]

failed = []
for package in packages:
    try:
        if package == 'sklearn':
            mod = importlib.import_module('sklearn')
        else:
            mod = importlib.import_module(package)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✓ {package:15s} {version}")
    except ImportError as e:
        print(f"✗ {package:15s} FAILED")
        failed.append(package)

if failed:
    print(f"\nFailed to import: {', '.join(failed)}")
    sys.exit(1)
else:
    print("\nAll packages imported successfully!")
    sys.exit(0)
EOF

# Run the test script
if uv run python /tmp/test_imports.py; then
    print_success "All Python packages verified"
else
    print_error "Some Python packages failed to import"
    rm /tmp/test_imports.py
    exit 1
fi

# Clean up test script
rm /tmp/test_imports.py

################################################################################
# Step 7: Display Summary
################################################################################
print_header "Installation Complete!"

echo ""
echo -e "${GREEN}✓ Setup completed successfully!${NC}"
echo ""
echo "Next steps:"
echo ""
echo "1. To start Jupyter Lab, run:"
echo -e "   ${BLUE}uv run jupyter lab${NC}"
echo ""
echo "2. Open one of the example notebooks in the notebooks/ directory:"
echo "   - cvar_basic.ipynb: Complete Mean-CVaR optimization walkthrough"
echo "   - efficient_frontier.ipynb: Generate efficient frontier"
echo "   - rebalancing_strategies.ipynb: Dynamic re-balancing strategies"
echo ""
echo "3. Select the 'Portfolio Optimization' kernel in Jupyter"
echo ""
echo "4. To activate the virtual environment manually:"
echo -e "   ${BLUE}source .venv/bin/activate${NC}"
echo ""

if [ "$INSTALL_DEV" = true ]; then
    echo "Development tools installed. Available commands:"
    echo "  - black: Code formatting"
    echo "  - isort: Import sorting"
    echo "  - flake8: Linting"
    echo "  - pre-commit: Git hooks"
    echo ""
fi

echo "For more information, see README.md"
echo ""
print_success "Happy optimizing! 🚀"

