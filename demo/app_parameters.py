#!/usr/bin/env python3
"""
Efficient Frontier Streamlit App Parameters

Configuration file containing all parameters for the efficient frontier app.
Modify values here to customize app behavior without changing the main code.
"""

from datetime import datetime

import pandas as pd

# =============================================================================
# DEFAULT INPUT VALUES
# =============================================================================


class DefaultValues:
    """Default values for all input controls"""

    BLUEPRINT_NAME = "Quantitative Portfolio Optimization developer example"

    # Dataset Settings
    DATASET_NAME = "sp500"  # Default dataset (if available)
    START_DATE = pd.to_datetime("2022-07-01")
    END_DATE = pd.to_datetime("2023-12-31")
    REGIME_NAME = "Selected Period"
    RETURN_TYPE = "LOG"  # "LOG" or "SIMPLE"

    # CVaR Parameters
    W_MIN = -0.3  # Minimum weight
    W_MAX = 0.8  # Maximum weight
    C_MIN = 0.1  # Minimum cash
    C_MAX = 0.4  # Maximum cash
    CONFIDENCE = 0.95  # CVaR confidence level
    NUM_SCEN = 10000  # Number of scenarios
    L_TAR = 1.6  # Leverage target
    FIT_TYPE = "kde"  # "kde" or "empirical"
    RISK_AVERSION = 1.2  # Default risk aversion for rebalancing

    # Efficient Frontier Settings
    RA_NUM = 20  # Number of portfolios
    MIN_RISK_AVERSION = 0.1  # Minimum risk aversion
    MAX_RISK_AVERSION = 100.0  # Maximum risk aversion

    # Discretized Portfolios Settings
    SHOW_DISCRETIZED = False  # Show discretized portfolios by default
    WEIGHT_DISCRETIZATION = 10  # Number of weight steps
    MAX_ASSETS = 5  # Maximum assets to use
    USE_GPU_DISCRETIZATION = False  # Use GPU acceleration for discretization

    # CPU Solver Settings
    DEFAULT_CPU_SOLVER = "HIGHS"  # Default CPU solver

    # Optional Constraints (for rebalancing app)
    ENABLE_TURNOVER_CONSTRAINT = False  # Enable turnover constraint by default
    TURNOVER_LIMIT = 1.0  # Default turnover limit (T_tar)
    ENABLE_CARDINALITY_CONSTRAINT = False  # Enable cardinality constraint by default
    CARDINALITY_LIMIT = 10  # Default maximum number of assets
    ENABLE_CVAR_LIMIT = False  # Enable hard CVaR limit by default
    CVAR_HARD_LIMIT = 0.02  # Default hard CVaR limit


# =============================================================================
# INPUT CONTROL LIMITS
# =============================================================================


class InputLimits:
    """Limits and constraints for input controls"""

    # CVaR Parameter Limits
    W_MIN_RANGE = (-2.0, 2.0)  # (min_value, max_value)
    W_MAX_RANGE = (0.0, 2.0)  # Will be dynamically set to (w_min, 2.0)
    C_MIN_RANGE = (0.0, 1.0)
    C_MAX_RANGE = (-1.0, 1.0)  # Will be dynamically set to (c_min, 1.0)
    CONFIDENCE_RANGE = (0.8, 0.99)
    NUM_SCEN_RANGE = (1000, 50000)
    L_TAR_RANGE = (0.5, 2.0)

    # Efficient Frontier Limits
    RA_NUM_RANGE = (5, 50)
    MIN_RISK_AVERSION_RANGE = (0.1, 10.0)
    MAX_RISK_AVERSION_RANGE = (0.1, 100.0)

    # Discretization Limits
    WEIGHT_DISCRETIZATION_RANGE = (2, 20)
    MAX_ASSETS_RANGE = (2, None)  # None = will be set to dataset size

    # Optional Constraint Limits (for rebalancing app)
    TURNOVER_LIMIT_RANGE = (0.01, 2.0)  # Min and max turnover constraint
    CARDINALITY_LIMIT_RANGE = (1, None)  # Min cardinality (max set by dataset size)
    CVAR_HARD_LIMIT_RANGE = (0.001, 1.0)  # Min and max hard CVaR limit

    # Step Sizes
    W_STEP = 0.1
    C_STEP = 0.1
    CONFIDENCE_STEP = 0.01
    NUM_SCEN_STEP = 1000
    L_TAR_STEP = 0.1
    RA_NUM_STEP = 5
    MIN_RISK_AVERSION_STEP = 0.1
    MAX_RISK_AVERSION_STEP = 1.0
    WEIGHT_DISCRETIZATION_STEP = 1
    TURNOVER_LIMIT_STEP = 0.1
    CARDINALITY_LIMIT_STEP = 1
    CVAR_HARD_LIMIT_STEP = 0.001


# =============================================================================
# VISUAL STYLING PARAMETERS
# =============================================================================


class PlotStyling:
    """Plot appearance and styling parameters"""

    # Color Schemes
    COLOR_SCHEMES = {
        "nvidia": {
            "frontier": "#7cd7fe",  # Light blue - matches rebalance.py
            "benchmark": ["#ef9100", "#ff8181", "#0d8473"],  # NVIDIA orange, red, dark teal - matches rebalance.py
            "assets": "#c359ef",  # Purple - matches rebalance.py
            "custom": "#fc79ca",  # Pink - matches rebalance.py
            "background": "#FFFFFF",  # White - matches rebalance.py
            "grid": "#E0E0E0",
        },
        "modern": {
            "frontier": "#7cd7fe",  # Light blue - matches rebalance.py
            "benchmark": ["#ef9100", "#ff8181", "#0d8473"],  # NVIDIA orange, red, dark teal - matches rebalance.py
            "assets": "#c359ef",  # Purple - matches rebalance.py
            "custom": "#fc79ca",  # Pink - matches rebalance.py
            "background": "#FFFFFF",  # White - matches rebalance.py
            "grid": "#E0E0E0",
        },
        "classic": {
            "frontier": "#1f77b4",
            "benchmark": ["#ff7f0e", "#2ca02c", "#d62728"],
            "assets": "#9467bd",
            "custom": "#e377c2",
            "background": "white",
            "grid": "#E8E8E8",
        },
        "vibrant": {
            "frontier": "#FF6B35",
            "benchmark": ["#F7931E", "#FFD23F", "#06FFA5"],
            "assets": "#B6244F",
            "custom": "#7209B7",
            "background": "#FFFEF7",
            "grid": "#E5E5E5",
        },
    }

    # Default color scheme to use
    DEFAULT_COLOR_SCHEME = "nvidia"

    # Figure Parameters
    FIGURE_SIZE = (16, 12)  # Larger size for better detail
    REBALANCING_FIGURE_SIZE = (16, 8)  # Shorter height so heatmap is visible below
    FIGURE_DPI = 300  # High resolution optimized for retina displays

    # Font Parameters
    XLABEL_FONTSIZE = 14
    YLABEL_FONTSIZE = 14
    TITLE_FONTSIZE = 16
    LEGEND_FONTSIZE = 12
    COLORBAR_FONTSIZE = 11
    ANNOTATION_FONTSIZE = 10
    FONT_WEIGHT = "bold"
    TITLE_PAD = 20

    # Line and Marker Parameters
    FRONTIER_LINEWIDTH = 3.5
    FRONTIER_ALPHA = 0.9
    FRONTIER_POINT_SIZE = 120
    FRONTIER_EDGE_WIDTH = 2.5
    FRONTIER_POINT_ALPHA = 0.8
    FRONTIER_FILL_ALPHA = 0.1

    # Discretized Portfolio Parameters
    DISCRETIZED_POINT_SIZE = 50
    DISCRETIZED_ALPHA = 0.6
    DISCRETIZED_EDGE_WIDTH = 0.6
    DISCRETIZED_COLORMAP = "plasma"

    # Special Portfolio Parameters
    SPECIAL_POINT_SIZE = 150
    SPECIAL_EDGE_WIDTH = 2.5
    SPECIAL_MARKERS = ["s", "^", "o"]  # square, triangle, circle

    # Grid and Spine Parameters
    GRID_ALPHA = 0.3
    SPINE_COLOR = "#CCCCCC"

    # Legend Parameters
    LEGEND_LOCATION = "upper left"
    LEGEND_FRAMEON = True
    LEGEND_FANCYBOX = True
    LEGEND_SHADOW = True
    LEGEND_FRAMEALPHA = 0.9

    # Colorbar Parameters
    COLORBAR_SHRINK = 0.8
    COLORBAR_PAD = 0.02
    COLORBAR_ROTATION = 270
    COLORBAR_LABELPAD = 15

    # Annotation Parameters
    ANNOTATION_OFFSET = (10, 10)
    ANNOTATION_BOXSTYLE = "round,pad=0.3"
    ANNOTATION_ALPHA = 0.8


# =============================================================================
# PERFORMANCE PARAMETERS
# =============================================================================


class PerformanceParams:
    """Performance and timing related parameters"""

    # Sleep Delays (in seconds)
    THREAD_SYNC_DELAY = 0.05  # Delay to ensure thread synchronization (reduced)
    UI_UPDATE_DELAY = 0.0  # Delay between portfolio updates (removed for GPU speed)
    INITIALIZATION_DELAY = 0.5  # Delay before starting optimization threads (reduced)
    MAIN_LOOP_DELAY = 0.01  # Delay in main progress checking loop
    ANIMATION_DELAY = 0.1  # Delay for plot animation effects (reduced)

    # GPU Optimization Parameters (separated progress bar from plot updates)
    ENABLE_REAL_TIME_PLOTS = (
        True  # Enable/disable real-time plot updates during optimization
    )
    PLOT_UPDATE_FREQUENCY = (
        1  # Update plots every N portfolios (1=every portfolio, 2=every other, etc.)
    )

    # Validation Thresholds
    MAX_COMBINATIONS_THRESHOLD = 1e10  # Maximum allowed portfolio combinations
    LARGE_COMBINATIONS_THRESHOLD = 1e7  # Threshold for "large" combinations warning
    MEDIUM_COMBINATIONS_THRESHOLD = 1e5  # Threshold for "medium" combinations info

    # Progress Display
    PROGRESS_PRECISION = 3  # Decimal places for timing displays
    TIME_PRECISION = 2  # Decimal places for total time displays


# =============================================================================
# SOLVER CONFIGURATION
# =============================================================================


class SolverConfig:
    """Solver options and configurations"""

    # Available CPU Solvers
    CPU_SOLVER_OPTIONS = {
        "HIGHS": "HiGHS (Fast LP/QP solver)",
        "CLARABEL": "Clarabel (Interior point)",
        "ECOS": "ECOS (Embedded conic)",
        "OSQP": "OSQP (Quadratic programming)",
        "SCS": "SCS (Splitting conic solver)",
    }

    # Default CPU solver
    DEFAULT_CPU_SOLVER = "HIGHS"

    # Solver Settings
    SOLVER_VERBOSE = False
    PRINT_RESULTS = False


# =============================================================================
# UI TEXT AND MESSAGES
# =============================================================================


class UIText:
    """Text messages and labels used in the UI"""

    # Section Headers
    DATASET_HEADER = "📊 Dataset Settings"
    CVAR_HEADER = "⚙️ CVaR Parameters"
    FRONTIER_HEADER = "📈 Efficient Frontier Settings"
    SOLVER_HEADER = "🚀 GPU vs CPU Comparison"
    DISCRETIZED_HEADER = "🎯 Discretized Portfolios"
    CONSTRAINTS_HEADER = "🔧 Optional Constraints"

    # Error Messages
    IMPOSSIBLE_CONSTRAINTS = "❌ Impossible constraints! Adjust Min Weight or Min Cash."
    TOO_MANY_COMBINATIONS = (
        "❌ Too many combinations! Reduce weight steps or max assets."
    )
    MIN_MAX_RISK_AVERSION_ERROR = "⚠️ Min must be ≤ Max Risk Aversion"
    MIN_RISK_AVERSION_WARNING = "⚠️ Min Risk Aversion must be > 0"

    # Info Messages
    STARTING_GPU = "🚀 Starting GPU (cuOpt)..."
    STARTING_CPU = "🖥️ Starting CPU ({})..."
    GPU_SYNCHRONIZED = "🚀 GPU synchronized and ready to race!"
    CPU_SYNCHRONIZED = "🖥️ CPU ({}) synchronized and ready to race!"
    RACE_STARTED_GPU = "🏁 GPU: Race started!"
    RACE_STARTED_CPU = "🏁 CPU: Race started!"

    # Instructions
    CONFIGURE_INSTRUCTION = "👈 **Configure parameters in the sidebar and click 'Generate Efficient Frontier' to start!**"

    # Button Labels
    RUN_BUTTON = "🚀 Generate Efficient Frontier"

    # Help Text
    CPU_SOLVER_HELP = "Select the CPU solver to compare against GPU cuOpt"
    GPU_ACCELERATION_HELP = "Use CuPy Numeric for GPU-accelerated portfolio evaluation"
    MAX_ASSETS_HELP = "Dataset {} contains {} assets"
    TURNOVER_CONSTRAINT_HELP = "Limit portfolio turnover (L1 distance from previous weights) - useful for controlling transaction costs"
    CARDINALITY_CONSTRAINT_HELP = "Limit the maximum number of assets with non-zero weights - creates more focused portfolios"
    CVAR_LIMIT_HELP = "Set a hard upper limit on portfolio CVaR risk - ensures risk stays below threshold"


# =============================================================================
# MATPLOTLIB STYLE CONFIGURATION
# =============================================================================


class MatplotlibConfig:
    """Matplotlib style and context settings"""

    STYLE = "seaborn-v0_8-whitegrid"
    CONTEXT = "paper"
    FONT_SCALE = 1.2


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_color_scheme(scheme_name=None):
    """Get color scheme by name, defaults to DEFAULT_COLOR_SCHEME"""
    if scheme_name is None:
        scheme_name = PlotStyling.DEFAULT_COLOR_SCHEME
    return PlotStyling.COLOR_SCHEMES.get(
        scheme_name, PlotStyling.COLOR_SCHEMES[PlotStyling.DEFAULT_COLOR_SCHEME]
    )


def get_cpu_solver_display_name(solver_key):
    """Get display name for CPU solver"""
    return SolverConfig.CPU_SOLVER_OPTIONS.get(solver_key, solver_key)


def validate_risk_aversion_range(min_ra, max_ra):
    """Validate risk aversion range"""
    if min_ra > max_ra:
        return False, UIText.MIN_MAX_RISK_AVERSION_ERROR
    elif min_ra <= 0:
        return False, UIText.MIN_RISK_AVERSION_WARNING
    return True, None


def validate_combination_count(total_combinations):
    """Validate discretization combination count"""
    if total_combinations > PerformanceParams.MAX_COMBINATIONS_THRESHOLD:
        return False, UIText.TOO_MANY_COMBINATIONS
    return True, None
