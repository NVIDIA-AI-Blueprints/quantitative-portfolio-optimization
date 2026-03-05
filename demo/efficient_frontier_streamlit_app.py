# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Efficient Frontier Streamlit App

Interactive web app for efficient frontier visualization with progressive portfolio solving
to showcase GPU optimization speed.

Author: phuo-nv
"""

import queue
import sys
import threading
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Import app parameters
from app_parameters import (
    DefaultValues,
    InputLimits,
    MatplotlibConfig,
    PerformanceParams,
    PlotStyling,
    SolverConfig,
    UIText,
    get_color_scheme,
    get_cpu_solver_display_name,
    validate_combination_count,
    validate_risk_aversion_range,
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure matplotlib for high-quality rendering
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['lines.antialiased'] = True
plt.rcParams['patch.antialiased'] = True
plt.rcParams['text.antialiased'] = True
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1

# Add workspace root to path for imports (package structure: cufolio = "src")
script_dir = Path(__file__).parent.absolute()
workspace_root = script_dir.parent  # This is the workspace root
sys.path.insert(0, str(workspace_root))
cvar_dir = workspace_root  # For backward compatibility with path references

try:
    # Import cufolio package
    import cvxpy as cp
    from cufolio import cvar_optimizer, cvar_utils, utils
    from cufolio.cvar_parameters import CvarParameters
    from cufolio.settings import (
        KDESettings,
        ReturnsComputeSettings,
        ScenarioGenerationSettings,
    )

    IMPORTS_OK = True
except ImportError as e:
    IMPORTS_OK = False
    IMPORT_ERROR = str(e)

# Set page config
st.set_page_config(
    page_title=f"{DefaultValues.blueprint_name} Efficient Frontier",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better appearance
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #76b900;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #76b900;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #76b900;
    }
    .stProgress .st-bo {
        background-color: #76b900;
    }
    .stAlert[data-baseweb="notification"] {
        border-left-color: #76b900 !important;
    }
    div[data-testid="stSuccess"] {
        background-color: rgba(118, 185, 0, 0.1) !important;
        border-left-color: #76b900 !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


def get_available_datasets():
    """Get list of available datasets"""
    try:
        data_dir = workspace_root / "data" / "stock_data"
        if data_dir.exists():
            datasets = [f.stem for f in data_dir.glob("*.csv")]
            return sorted(datasets)
        return ["sp500", "sp100"]  # fallback
    except Exception:
        return ["sp500", "sp100"]  # fallback


def get_dataset_num_assets(dataset_name):
    """Get the number of assets in a dataset"""
    try:
        dataset_path = workspace_root / "data" / "stock_data" / f"{dataset_name}.csv"
        if dataset_path.exists():
            df = pd.read_csv(dataset_path)
            # Subtract 1 for date column, assuming first column is date
            return len(df.columns) - 1
        return 30  # Default fallback
    except Exception:
        return 30  # Default fallback


def display_special_portfolios_table(results_df):
    """Display a table of special portfolios (min variance, max Sharpe, max return)."""
    if results_df is None or len(results_df) == 0:
        st.warning("No portfolio data available")
        return

    try:
        # Calculate portfolio metrics
        returns = results_df["return"].values
        risks = results_df["risk"].values  # CVaR values
        variances = risks**2  # Approximate variance using CVaR
        sharpe_ratios = np.where(risks > 0, returns / risks, 0)

        # Find special portfolio indices
        min_var_idx = np.argmin(variances)
        max_sharpe_idx = np.argmax(sharpe_ratios)
        max_return_idx = np.argmax(returns)

        # Create special portfolios data
        special_data = []

        # Min Variance Portfolio
        special_data.append(
            {
                "Portfolio Type": "🔷 Min Variance",
                "Return (%)": f"{returns[min_var_idx] * 100:.2f}",
                "CVaR (%)": f"{risks[min_var_idx] * 100:.2f}",
                "Sharpe Ratio": f"{sharpe_ratios[min_var_idx]:.3f}",
                "Risk Aversion": f"{results_df.iloc[min_var_idx]['risk_aversion']:.4f}",
            }
        )

        # Max Sharpe Portfolio
        special_data.append(
            {
                "Portfolio Type": "🔸 Max Sharpe",
                "Return (%)": f"{returns[max_sharpe_idx] * 100:.2f}",
                "CVaR (%)": f"{risks[max_sharpe_idx] * 100:.2f}",
                "Sharpe Ratio": f"{sharpe_ratios[max_sharpe_idx]:.3f}",
                "Risk Aversion": f"{results_df.iloc[max_sharpe_idx]['risk_aversion']:.4f}",
            }
        )

        # Max Return Portfolio
        special_data.append(
            {
                "Portfolio Type": "🔹 Max Return",
                "Return (%)": f"{returns[max_return_idx] * 100:.2f}",
                "CVaR (%)": f"{risks[max_return_idx] * 100:.2f}",
                "Sharpe Ratio": f"{sharpe_ratios[max_return_idx]:.3f}",
                "Risk Aversion": f"{results_df.iloc[max_return_idx]['risk_aversion']:.4f}",
            }
        )

        # Display as DataFrame
        special_df = pd.DataFrame(special_data)
        st.dataframe(special_df, hide_index=True, width="stretch")

        # Show portfolio characteristics
        st.caption(
            "💡 **Legend:** Min Variance (lowest risk), Max Sharpe (best risk-adjusted return), Max Return (highest return)"
        )

    except Exception as e:
        st.error(f"Error calculating special portfolios: {str(e)}")


def create_efficient_frontier_progressive(
    dataset_path,
    regime_dict,
    returns_compute_settings,
    scenario_generation_settings,
    cvar_params,
    solver_settings,
    ra_num,
    min_risk_aversion,
    max_risk_aversion,
    show_discretized_portfolios,
    discretization_params,
    solver_name,
    progress_queue,
    result_queue,
    start_event,
    optimization_start_event,
):
    """Progressive efficient frontier creation with queue communication."""
    try:
        # Send minimal initial status for synchronization
        progress_queue.put(
            {
                "solver": solver_name,
                "status": "initializing",
                "message": f"Initializing {solver_name}...",
            }
        )

        # Get color scheme from parameters
        colors = get_color_scheme()

        # Set beautiful styling
        plt.style.use(MatplotlibConfig.STYLE)
        sns.set_context(
            MatplotlibConfig.CONTEXT, font_scale=MatplotlibConfig.FONT_SCALE
        )

        # Initialize the plot with beautiful styling and consistent layout
        fig, ax = plt.subplots(
            figsize=PlotStyling.FIGURE_SIZE,
            dpi=PlotStyling.FIGURE_DPI,
            facecolor=colors["background"],
            tight_layout=True,
        )
        ax.set_facecolor(colors["background"])

        # Beautiful labels and title
        ax.set_xlabel(
            f"{cvar_params.confidence:.0%} CVaR (percentage)",
            fontsize=PlotStyling.XLABEL_FONTSIZE,
            fontweight=PlotStyling.FONT_WEIGHT,
        )
        ax.set_ylabel(
            "Expected Return (percentage)",
            fontsize=PlotStyling.YLABEL_FONTSIZE,
            fontweight=PlotStyling.FONT_WEIGHT,
        )
        ax.set_title(
            f'Efficient Frontier - {regime_dict["name"]} ({solver_name})',
            fontsize=PlotStyling.TITLE_FONTSIZE,
            fontweight=PlotStyling.FONT_WEIGHT,
            pad=PlotStyling.TITLE_PAD,
        )

        # Beautiful grid and styling
        ax.grid(True, alpha=PlotStyling.GRID_ALPHA, color=colors["grid"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(PlotStyling.SPINE_COLOR)
        ax.spines["bottom"].set_color(PlotStyling.SPINE_COLOR)

        # Ensure consistent margins and layout
        fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)

        # Step 1: Calculate returns
        progress_queue.put(
            {
                "solver": solver_name,
                "status": "calculating_returns",
                "message": f"{solver_name}: Calculating returns...",
            }
        )

        # Compute returns from price data
        returns_dict = utils.calculate_returns(
            dataset_path, regime_dict, returns_compute_settings
        )

        # Generate return scenarios from KDE or other fit types
        returns_dict = cvar_utils.generate_cvar_data(
            returns_dict,
            scenario_generation_settings
        )

        # Step 2: Generate risk aversion levels using logarithmic spacing (reversed order)
        if min_risk_aversion <= 0:
            min_risk_aversion = 0.0001

        risk_aversion_list = np.logspace(
            np.log10(min_risk_aversion), np.log10(max_risk_aversion), ra_num
        )[::-1]

        # Step 3: Wait for synchronization signal before starting any plotting
        progress_queue.put(
            {
                "solver": solver_name,
                "status": "ready",
                "message": f"{solver_name}: Ready to start - waiting for synchronization...",
            }
        )

        # Wait for start signal to ensure simultaneous beginning
        start_event.wait()

        # Step 4a: Initial plot is now handled externally for true simultaneity
        # Send ready status to indicate thread is synchronized and starting

        # Step 4b: NOW show discretized portfolios (if enabled) - AFTER synchronization
        discretized_portfolios = None
        if show_discretized_portfolios:
            try:
                progress_queue.put(
                    {
                        "solver": solver_name,
                        "status": "discretized_portfolios",
                        "message": f"{solver_name}: Generating discretized portfolios...",
                    }
                )

                discretized_portfolios = cvar_utils.evaluate_all_linear_combinations(
                    returns_dict, cvar_params, **discretization_params
                )

                # Plot beautiful discretized portfolios with variance as color
                scatter = ax.scatter(
                    discretized_portfolios["CVaR"] * 100,
                    discretized_portfolios["return"] * 100,
                    s=PlotStyling.DISCRETIZED_POINT_SIZE,
                    c=discretized_portfolios["variance"],
                    cmap=PlotStyling.DISCRETIZED_COLORMAP,
                    alpha=PlotStyling.DISCRETIZED_ALPHA,
                    edgecolor="white",
                    linewidth=PlotStyling.DISCRETIZED_EDGE_WIDTH,
                    label="Discretized Portfolios",
                    zorder=2,
                )

                # Add beautiful colorbar for portfolio variance
                cbar = plt.colorbar(
                    scatter,
                    ax=ax,
                    shrink=PlotStyling.COLORBAR_SHRINK,
                    pad=PlotStyling.COLORBAR_PAD,
                )
                cbar.set_label(
                    "Portfolio Variance",
                    rotation=PlotStyling.COLORBAR_ROTATION,
                    labelpad=PlotStyling.COLORBAR_LABELPAD,
                    fontweight=PlotStyling.FONT_WEIGHT,
                )
                cbar.ax.tick_params(labelsize=PlotStyling.COLORBAR_FONTSIZE)

                # Beautiful legend styling
                ax.legend(
                    loc=PlotStyling.LEGEND_LOCATION,
                    frameon=PlotStyling.LEGEND_FRAMEON,
                    fancybox=PlotStyling.LEGEND_FANCYBOX,
                    shadow=PlotStyling.LEGEND_SHADOW,
                    framealpha=PlotStyling.LEGEND_FRAMEALPHA,
                    fontsize=PlotStyling.LEGEND_FONTSIZE,
                )

                # Send the discretized plot immediately after creation for simultaneous display
                progress_queue.put(
                    {
                        "solver": solver_name,
                        "status": "discretized_complete",
                        "message": f"{solver_name}: Discretized portfolios ready",
                        "figure": fig,
                    }
                )

            except Exception as e:
                discretized_portfolios = None

        # Step 4c: Always send a plot update and synchronization checkpoint before optimization
        # This ensures both solvers are at the same point regardless of discretization settings
        progress_queue.put(
            {
                "solver": solver_name,
                "status": "plot_ready",
                "message": f"{solver_name}: Plot initialized - ready for optimization",
                "figure": fig,
            }
        )

        # CRITICAL SYNCHRONIZATION: Wait for both solvers to reach this point
        # This ensures the optimization race starts simultaneously, regardless of discretization time
        progress_queue.put(
            {
                "solver": solver_name,
                "status": "waiting_for_race_start",
                "message": f"{solver_name}: Waiting for race start signal...",
            }
        )

        optimization_start_event.wait()  # Wait for signal to start optimization race

        progress_queue.put(
            {
                "solver": solver_name,
                "status": "starting_optimization",
                "message": f"{solver_name}: Starting efficient frontier optimization NOW!",
            }
        )

        cvar_problem = cvar_optimizer.CVaR(
            returns_dict=returns_dict, cvar_params=cvar_params
        )

        # Initialize results
        results_data = []
        efficient_risks = []
        efficient_returns = []
        total_time = 0
        total_solver_time = 0

        # Solve each portfolio progressively with optimized updates
        for i, ra_value in enumerate(risk_aversion_list):
            start_time = time.time()

            # Update parameters and setup
            # cvar_problem.params.update_risk_aversion(ra_value)
            # cvar_problem._setup_optimization_problem()

            cvar_problem.risk_aversion_param.value = (
                ra_value * cvar_problem._risk_aversion_scalar
            )

            # Solve optimization
            result_row, portfolio = cvar_problem.solve_optimization_problem(
                solver_settings=solver_settings,
                print_results=SolverConfig.PRINT_RESULTS,
            )

            total_time_for_portfolio = time.time() - start_time
            solve_time = result_row[
                "solve time"
            ]  # Extract actual solver time from CVaR optimizer
            total_time += total_time_for_portfolio
            total_solver_time += solve_time

            # Extract results
            risk = result_row["CVaR"]
            ret = result_row["return"]

            efficient_risks.append(risk)
            efficient_returns.append(ret)
            results_data.append(
                {
                    "risk_aversion": ra_value,
                    "risk": risk,
                    "return": ret,
                    "solve_time": solve_time,
                    "portfolio": portfolio.print_clean(),
                }
            )

            # Update plot with new point using beautiful colors
            ax.scatter(
                [risk * 100],
                [ret * 100],
                c=colors["frontier"],
                s=PlotStyling.FRONTIER_POINT_SIZE,
                marker="o",
                zorder=5,
                edgecolors="white",
                linewidth=PlotStyling.FRONTIER_EDGE_WIDTH,
                alpha=PlotStyling.FRONTIER_POINT_ALPHA,
            )

            # Draw beautiful line connecting points (if more than one point)
            if len(efficient_risks) > 1:
                ax.plot(
                    [r * 100 for r in efficient_risks],
                    [r * 100 for r in efficient_returns],
                    color=colors["frontier"],
                    linewidth=PlotStyling.FRONTIER_LINEWIDTH,
                    alpha=PlotStyling.FRONTIER_ALPHA,
                    zorder=4,
                    label="Efficient Frontier",
                )

                # Add beautiful gradient fill under the frontier
                # ax.fill_between(
                #     [r * 100 for r in efficient_risks],
                #     [r * 100 for r in efficient_returns],
                #     alpha=PlotStyling.FRONTIER_FILL_ALPHA,
                #     color=colors["frontier"],
                #     zorder=1,
                # )

                # Update legend with beautiful styling
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(
                    by_label.values(),
                    by_label.keys(),
                    loc=PlotStyling.LEGEND_LOCATION,
                    frameon=PlotStyling.LEGEND_FRAMEON,
                    fancybox=PlotStyling.LEGEND_FANCYBOX,
                    shadow=PlotStyling.LEGEND_SHADOW,
                    framealpha=PlotStyling.LEGEND_FRAMEALPHA,
                    fontsize=PlotStyling.LEGEND_FONTSIZE,
                )

            # Send immediate progress update (without heavy figure) for responsive progress bar
            progress_queue.put(
                {
                    "solver": solver_name,
                    "status": "portfolio_progress",
                    "portfolio_num": i + 1,
                    "total_portfolios": ra_num,
                    "risk_aversion": ra_value,
                    "risk": risk,
                    "return": ret,
                    "solve_time": solve_time,
                    "total_time": total_time,
                    "total_solver_time": total_solver_time,
                    "message": f"{solver_name}: Solved portfolio {i+1}/{ra_num} (RA: {ra_value:.3f})",
                }
            )

            # Send plot update every few portfolios to avoid lag (configurable frequency)
            if (
                i + 1
            ) % PerformanceParams.PLOT_UPDATE_FREQUENCY == 0 or i == ra_num - 1:
                progress_queue.put(
                    {
                        "solver": solver_name,
                        "status": "portfolio_plot_update",
                        "portfolio_num": i + 1,
                        "total_portfolios": ra_num,
                        "figure": fig,
                        "message": f"{solver_name}: Plot updated - portfolio {i+1}/{ra_num}",
                    }
                )

            # No delay for GPU optimization - let it run at full speed

        # Create results DataFrame
        results_df = pd.DataFrame(results_data)

        # Calculate and mark special portfolios on the efficient frontier
        if len(results_data) > 0:
            # Calculate portfolio metrics for special portfolio identification
            returns = np.array([r["return"] for r in results_data])
            risks = np.array([r["risk"] for r in results_data])
            variances = risks**2  # Assuming CVaR approximates variance for this purpose
            sharpe_ratios = np.where(risks > 0, returns / risks, 0)  # Return/Risk ratio

            # Find special portfolio indices
            min_var_idx = np.argmin(variances)
            max_sharpe_idx = np.argmax(sharpe_ratios)
            max_return_idx = np.argmax(returns)

            # Add markers for special portfolios
            special_portfolios = [
                {
                    "type": "Min Variance",
                    "idx": min_var_idx,
                    "risk": risks[min_var_idx] * 100,
                    "return": returns[min_var_idx] * 100,
                    "color": "blue",
                    "marker": "s",  # square
                    "size": 150,
                },
                {
                    "type": "Max Sharpe",
                    "idx": max_sharpe_idx,
                    "risk": risks[max_sharpe_idx] * 100,
                    "return": returns[max_sharpe_idx] * 100,
                    "color": "gold",
                    "marker": "^",  # triangle
                    "size": 150,
                },
                {
                    "type": "Max Return",
                    "idx": max_return_idx,
                    "risk": risks[max_return_idx] * 100,
                    "return": returns[max_return_idx] * 100,
                    "color": "green",
                    "marker": "o",  # circle
                    "size": 150,
                },
            ]

            # Plot beautiful special portfolio markers
            for i, portfolio in enumerate(special_portfolios):
                ax.scatter(
                    portfolio["risk"],
                    portfolio["return"],
                    s=PlotStyling.SPECIAL_POINT_SIZE,
                    color=colors["benchmark"][i],
                    marker=PlotStyling.SPECIAL_MARKERS[i],
                    edgecolor="white",
                    linewidth=PlotStyling.SPECIAL_EDGE_WIDTH,
                    zorder=10,
                    label=f"{portfolio['type']} Portfolio",
                )

                # Add beautiful annotation with rounded box
                ax.annotate(
                    f"{portfolio['type']}\nReturn: {portfolio['return']:.2f}%\nCVaR: {portfolio['risk']:.2f}%",
                    (portfolio["risk"], portfolio["return"]),
                    xytext=PlotStyling.ANNOTATION_OFFSET,
                    textcoords="offset points",
                    bbox=dict(
                        boxstyle=PlotStyling.ANNOTATION_BOXSTYLE,
                        facecolor=colors["benchmark"][i],
                        alpha=PlotStyling.ANNOTATION_ALPHA,
                        edgecolor="white",
                    ),
                    fontsize=PlotStyling.ANNOTATION_FONTSIZE,
                    color="white",
                    fontweight=PlotStyling.FONT_WEIGHT,
                    ha="left",
                    zorder=11,
                )

            # Update legend to include special portfolios with beautiful styling
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(
                by_label.values(),
                by_label.keys(),
                loc=PlotStyling.LEGEND_LOCATION,
                frameon=PlotStyling.LEGEND_FRAMEON,
                fancybox=PlotStyling.LEGEND_FANCYBOX,
                shadow=PlotStyling.LEGEND_SHADOW,
                framealpha=PlotStyling.LEGEND_FRAMEALPHA,
                fontsize=PlotStyling.LEGEND_FONTSIZE,
            )

            # Send update with marked portfolios
            progress_queue.put(
                {
                    "solver": solver_name,
                    "status": "special_portfolios_marked",
                    "message": f"{solver_name}: Special portfolios marked on efficient frontier",
                    "figure": fig,
                    "special_portfolios": special_portfolios,
                }
            )

        # Send final completion status
        progress_queue.put(
            {
                "solver": solver_name,
                "status": "completed",
                "message": f"{solver_name}: Completed! {len(results_data)} portfolios in {total_time:.2f}s",
            }
        )

        # Send final result
        result_queue.put(
            {
                "success": True,
                "solver_name": solver_name,
                "results_df": results_df,
                "fig": fig,
                "ax": ax,
                "total_time": total_time,
                "total_solver_time": total_solver_time,
                "discretized_portfolios": discretized_portfolios,
                "error": None,
            }
        )

    except Exception as e:
        # Send error status
        progress_queue.put(
            {
                "solver": solver_name,
                "status": "error",
                "message": f"{solver_name}: Error - {str(e)}",
            }
        )

        result_queue.put(
            {
                "success": False,
                "solver_name": solver_name,
                "results_df": None,
                "fig": None,
                "ax": None,
                "total_time": 0,
                "total_solver_time": 0,
                "discretized_portfolios": None,
                "error": str(e),
            }
        )


def run_progressive_efficient_frontiers(
    dataset_path,
    regime_dict,
    returns_compute_settings,
    scenario_generation_settings,
    cvar_params,
    ra_num,
    min_risk_aversion,
    max_risk_aversion,
    show_discretized_portfolios,
    discretization_params,
    gpu_plot_container,
    cpu_plot_container,
    gpu_progress_placeholder,
    cpu_progress_placeholder,
    cpu_solver_choice,
    blog_mode=True,
):
    """Run GPU and CPU efficient frontiers in parallel with progressive updates."""

    # Setup solver configurations
    gpu_settings = {"solver": cp.CUOPT, "verbose": SolverConfig.SOLVER_VERBOSE}

    # Map CPU solver choice to CVXPY solver
    cpu_solver_map = {
        "HIGHS": cp.HIGHS,
        "CLARABEL": cp.CLARABEL,
        "ECOS": cp.ECOS,
        "OSQP": cp.OSQP,
        "SCS": cp.SCS,
    }

    cpu_settings = {
        "solver": cpu_solver_map.get(cpu_solver_choice, cp.HIGHS),
        "verbose": SolverConfig.SOLVER_VERBOSE,
    }

    # Create queues for communication
    gpu_progress_queue = queue.Queue()
    cpu_progress_queue = queue.Queue()
    gpu_result_queue = queue.Queue()
    cpu_result_queue = queue.Queue()

    # Start both progress indicators
    with gpu_progress_placeholder.container():
        st.info(UIText.STARTING_GPU)
    with cpu_progress_placeholder.container():
        st.info(UIText.STARTING_CPU.format(cpu_solver_choice))

    # Create and display empty plots immediately for simultaneous appearance
    # Create identical empty plots for both GPU and CPU
    def create_empty_plot(solver_name):
        plt.style.use(MatplotlibConfig.STYLE)
        sns.set_context(
            MatplotlibConfig.CONTEXT, font_scale=MatplotlibConfig.FONT_SCALE
        )
        colors = get_color_scheme()

        # Create figure with consistent sizing and layout
        fig, ax = plt.subplots(
            figsize=PlotStyling.FIGURE_SIZE,
            dpi=PlotStyling.FIGURE_DPI,
            facecolor=colors["background"],
            tight_layout=True,
        )
        ax.set_facecolor(colors["background"])

        ax.set_xlabel(
            f"{cvar_params.confidence:.0%} CVaR (percentage)",
            fontsize=PlotStyling.XLABEL_FONTSIZE,
            fontweight=PlotStyling.FONT_WEIGHT,
        )
        ax.set_ylabel(
            "Expected Return (percentage)",
            fontsize=PlotStyling.YLABEL_FONTSIZE,
            fontweight=PlotStyling.FONT_WEIGHT,
        )
        ax.set_title(
            f'Efficient Frontier - {regime_dict["name"]}',
            fontsize=PlotStyling.TITLE_FONTSIZE,
            fontweight=PlotStyling.FONT_WEIGHT,
            pad=PlotStyling.TITLE_PAD,
        )

        ax.grid(True, alpha=PlotStyling.GRID_ALPHA, color=colors["grid"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(PlotStyling.SPINE_COLOR)
        ax.spines["bottom"].set_color(PlotStyling.SPINE_COLOR)

        # Ensure consistent margins and layout
        fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)

        return fig

    # Create and display both empty plots simultaneously
    # In blog mode, hide CPU solver name in title
    cpu_display_name = "CPU" if blog_mode else f"CPU ({cpu_solver_choice})"
    gpu_empty_fig = create_empty_plot("GPU (cuOpt)")
    cpu_empty_fig = create_empty_plot(cpu_display_name)

    # Display both plots at exactly the same time with consistent sizing
    gpu_plot_container.pyplot(gpu_empty_fig, width="stretch")
    cpu_plot_container.pyplot(cpu_empty_fig, width="stretch")

    # Clean up the temporary figures
    plt.close(gpu_empty_fig)
    plt.close(cpu_empty_fig)

    # Create synchronization events to ensure simultaneous start
    start_event = threading.Event()  # Initial synchronization
    optimization_start_event = threading.Event()  # Optimization race synchronization

    # Start threads
    gpu_thread = threading.Thread(
        target=create_efficient_frontier_progressive,
        args=(
            dataset_path,
            regime_dict,
            returns_compute_settings,
            scenario_generation_settings,
            cvar_params,
            gpu_settings,
            ra_num,
            min_risk_aversion,
            max_risk_aversion,
            show_discretized_portfolios,
            discretization_params,
            "GPU (cuOpt)",
            gpu_progress_queue,
            gpu_result_queue,
            start_event,
            optimization_start_event,
        ),
    )

    cpu_thread = threading.Thread(
        target=create_efficient_frontier_progressive,
        args=(
            dataset_path,
            regime_dict,
            returns_compute_settings,
            scenario_generation_settings,
            cvar_params,
            cpu_settings,
            ra_num,
            min_risk_aversion,
            max_risk_aversion,
            show_discretized_portfolios,
            discretization_params,
            cpu_display_name,
            cpu_progress_queue,
            cpu_result_queue,
            start_event,
            optimization_start_event,
        ),
    )

    gpu_thread.start()
    cpu_thread.start()

    # Minimal wait for both threads to initialize, then signal them to start
    time.sleep(PerformanceParams.INITIALIZATION_DELAY)

    # Show synchronization message
    with gpu_progress_placeholder.container():
        st.info(UIText.GPU_SYNCHRONIZED)
    with cpu_progress_placeholder.container():
        st.info(UIText.CPU_SYNCHRONIZED.format(cpu_solver_choice))

    start_event.set()  # Signal both threads to start optimization simultaneously

    # Track completion status
    gpu_completed = False
    cpu_completed = False
    results = {}
    gpu_started = False
    cpu_started = False
    gpu_waiting_for_race = False
    cpu_waiting_for_race = False
    race_started = False

    # Main update loop - check for progress updates independently and in real-time
    while not (gpu_completed and cpu_completed):

        # Process ALL progress updates immediately, but only one plot update per loop

        # Check GPU progress - process ALL pending progress updates for immediate response
        gpu_processed_plot = False
        try:
            while True:  # Process all pending GPU updates
                gpu_update = gpu_progress_queue.get_nowait()

                if gpu_update["status"] == "portfolio_progress":
                    # Update GPU progress bar immediately (lightweight update)
                    with gpu_progress_placeholder.container():
                        progress = (
                            gpu_update["portfolio_num"] / gpu_update["total_portfolios"]
                        )
                        st.progress(progress)
                        avg_time_per_portfolio = (
                            gpu_update["total_time"] / gpu_update["portfolio_num"]
                        )
                        avg_solver_time = (
                            gpu_update["total_solver_time"]
                            / gpu_update["portfolio_num"]
                        )
                        st.caption(
                            f"Last Solver: {gpu_update['solve_time']:.3f}s | Avg Solver: {avg_solver_time:.3f}s | Total: {gpu_update['total_time']:.2f}s"
                        )

                elif (
                    gpu_update["status"] == "portfolio_plot_update"
                    and not gpu_processed_plot
                ):
                    # Update GPU plot (heavier update, only one per loop iteration)
                    if "figure" in gpu_update and gpu_update["figure"] is not None:
                        gpu_plot_container.pyplot(gpu_update["figure"], width="stretch")
                        gpu_processed_plot = True

                elif gpu_update["status"] == "portfolio_solved":
                    # Legacy support - handle both progress and plot in one update
                    if (
                        "figure" in gpu_update
                        and gpu_update["figure"] is not None
                        and not gpu_processed_plot
                    ):
                        gpu_plot_container.pyplot(gpu_update["figure"], width="stretch")
                        gpu_processed_plot = True

                    with gpu_progress_placeholder.container():
                        progress = (
                            gpu_update["portfolio_num"] / gpu_update["total_portfolios"]
                        )
                        st.progress(progress)
                        avg_time_per_portfolio = (
                            gpu_update["total_time"] / gpu_update["portfolio_num"]
                        )
                        # Check if total_solver_time exists (for backward compatibility)
                        if "total_solver_time" in gpu_update:
                            avg_solver_time = (
                                gpu_update["total_solver_time"]
                                / gpu_update["portfolio_num"]
                            )
                            st.caption(
                                f"Last Solver: {gpu_update['solve_time']:.3f}s | Avg Solver: {avg_solver_time:.3f}s | Total: {gpu_update['total_time']:.2f}s"
                            )
                        else:
                            st.caption(
                                f"Last: {gpu_update['solve_time']:.3f}s | Avg: {avg_time_per_portfolio:.3f}s | Total: {gpu_update['total_time']:.2f}s"
                            )

                elif gpu_update["status"] == "special_portfolios_marked":
                    # Update GPU plot with special portfolio markers (always show - this is important final result)
                    if "figure" in gpu_update:
                        gpu_plot_container.pyplot(gpu_update["figure"], width="stretch")
                    with gpu_progress_placeholder.container():
                        st.success(gpu_update["message"])

                elif gpu_update["status"] == "completed":
                    with gpu_progress_placeholder.container():
                        st.success(gpu_update["message"])
                    gpu_completed = True
                    break  # Exit the while loop to stop processing more updates

                elif gpu_update["status"] == "error":
                    with gpu_progress_placeholder.container():
                        st.error(gpu_update["message"])
                    gpu_completed = True
                    break  # Exit the while loop to stop processing more updates

                elif gpu_update["status"] == "ready":
                    with gpu_progress_placeholder.container():
                        st.warning(gpu_update["message"])
                    gpu_started = True

                elif gpu_update["status"] == "discretized_complete":
                    # Display the discretized portfolios plot immediately (always show - important initial visualization)
                    if "figure" in gpu_update:
                        gpu_plot_container.pyplot(gpu_update["figure"], width="stretch")
                    with gpu_progress_placeholder.container():
                        st.success(gpu_update["message"])

                elif gpu_update["status"] == "plot_ready":
                    # Display the plot ready state - ensures synchronization checkpoint (always show)
                    if "figure" in gpu_update:
                        gpu_plot_container.pyplot(gpu_update["figure"], width="stretch")
                    with gpu_progress_placeholder.container():
                        st.success(gpu_update["message"])

                elif gpu_update["status"] == "waiting_for_race_start":
                    # GPU is ready and waiting for race to start
                    gpu_waiting_for_race = True
                    with gpu_progress_placeholder.container():
                        st.warning(gpu_update["message"])

                elif gpu_update["status"] in [
                    "initializing",
                    "calculating_returns",
                    "discretized_portfolios",
                    "starting_optimization",
                ]:
                    with gpu_progress_placeholder.container():
                        st.info(gpu_update["message"])

        except queue.Empty:
            pass

        # Check CPU progress - process ALL pending progress updates for immediate response
        cpu_processed_plot = False
        try:
            while True:  # Process all pending CPU updates
                cpu_update = cpu_progress_queue.get_nowait()

                if cpu_update["status"] == "portfolio_progress":
                    # Update CPU progress bar immediately (lightweight update)
                    with cpu_progress_placeholder.container():
                        progress = (
                            cpu_update["portfolio_num"] / cpu_update["total_portfolios"]
                        )
                        st.progress(progress)
                        avg_time_per_portfolio = (
                            cpu_update["total_time"] / cpu_update["portfolio_num"]
                        )
                        avg_solver_time = (
                            cpu_update["total_solver_time"]
                            / cpu_update["portfolio_num"]
                        )
                        st.caption(
                            f"Last Solver: {cpu_update['solve_time']:.3f}s | Avg Solver: {avg_solver_time:.3f}s | Total: {cpu_update['total_time']:.2f}s"
                        )

                elif (
                    cpu_update["status"] == "portfolio_plot_update"
                    and not cpu_processed_plot
                ):
                    # Update CPU plot (heavier update, only one per loop iteration)
                    if "figure" in cpu_update and cpu_update["figure"] is not None:
                        cpu_plot_container.pyplot(cpu_update["figure"], width="stretch")
                        cpu_processed_plot = True

                elif cpu_update["status"] == "portfolio_solved":
                    # Legacy support - handle both progress and plot in one update
                    if (
                        "figure" in cpu_update
                        and cpu_update["figure"] is not None
                        and not cpu_processed_plot
                    ):
                        cpu_plot_container.pyplot(cpu_update["figure"], width="stretch")
                        cpu_processed_plot = True

                    with cpu_progress_placeholder.container():
                        progress = (
                            cpu_update["portfolio_num"] / cpu_update["total_portfolios"]
                        )
                        st.progress(progress)
                        avg_time_per_portfolio = (
                            cpu_update["total_time"] / cpu_update["portfolio_num"]
                        )
                        # Check if total_solver_time exists (for backward compatibility)
                        if "total_solver_time" in cpu_update:
                            avg_solver_time = (
                                cpu_update["total_solver_time"]
                                / cpu_update["portfolio_num"]
                            )
                            st.caption(
                                f"Last Solver: {cpu_update['solve_time']:.3f}s | Avg Solver: {avg_solver_time:.3f}s | Total: {cpu_update['total_time']:.2f}s"
                            )
                        else:
                            st.caption(
                                f"Last: {cpu_update['solve_time']:.3f}s | Avg: {avg_time_per_portfolio:.3f}s | Total: {cpu_update['total_time']:.2f}s"
                            )

                elif cpu_update["status"] == "special_portfolios_marked":
                    # Update CPU plot with special portfolio markers (always show - this is important final result)
                    if "figure" in cpu_update:
                        cpu_plot_container.pyplot(cpu_update["figure"], width="stretch")
                    with cpu_progress_placeholder.container():
                        st.success(cpu_update["message"])

                elif cpu_update["status"] == "completed":
                    with cpu_progress_placeholder.container():
                        st.success(cpu_update["message"])
                    cpu_completed = True
                    break  # Exit the while loop to stop processing more updates

                elif cpu_update["status"] == "error":
                    with cpu_progress_placeholder.container():
                        st.error(cpu_update["message"])
                    cpu_completed = True
                    break  # Exit the while loop to stop processing more updates

                elif cpu_update["status"] == "ready":
                    with cpu_progress_placeholder.container():
                        st.warning(cpu_update["message"])
                    cpu_started = True

                elif cpu_update["status"] == "discretized_complete":
                    # Display the discretized portfolios plot immediately (always show - important initial visualization)
                    if "figure" in cpu_update:
                        cpu_plot_container.pyplot(cpu_update["figure"], width="stretch")
                    with cpu_progress_placeholder.container():
                        st.success(cpu_update["message"])

                elif cpu_update["status"] == "plot_ready":
                    # Display the plot ready state - ensures synchronization checkpoint (always show)
                    if "figure" in cpu_update:
                        cpu_plot_container.pyplot(cpu_update["figure"], width="stretch")
                    with cpu_progress_placeholder.container():
                        st.success(cpu_update["message"])

                elif cpu_update["status"] == "waiting_for_race_start":
                    # CPU is ready and waiting for race to start
                    cpu_waiting_for_race = True
                    with cpu_progress_placeholder.container():
                        st.warning(cpu_update["message"])

                elif cpu_update["status"] in [
                    "initializing",
                    "calculating_returns",
                    "discretized_portfolios",
                    "starting_optimization",
                ]:
                    with cpu_progress_placeholder.container():
                        st.info(cpu_update["message"])

        except queue.Empty:
            pass

        # Check if both are waiting for race start and signal them to begin
        if (
            gpu_waiting_for_race
            and cpu_waiting_for_race
            and not race_started
            and not gpu_completed
            and not cpu_completed
        ):
            # Show synchronized race start message
            with gpu_progress_placeholder.container():
                st.success("🏁 GPU: Ready! Starting race NOW!")
            with cpu_progress_placeholder.container():
                st.success("🏁 CPU: Ready! Starting race NOW!")

            # Signal both threads to start optimization race simultaneously
            optimization_start_event.set()
            race_started = True

        # Check if both are ready and show synchronized start message
        if gpu_started and cpu_started and not gpu_completed and not cpu_completed:
            with gpu_progress_placeholder.container():
                st.success(UIText.RACE_STARTED_GPU)
            with cpu_progress_placeholder.container():
                st.success(UIText.RACE_STARTED_CPU)
            gpu_started = False  # Reset to avoid repeated messages
            cpu_started = False

        # Reduced delay to improve responsiveness while preventing excessive CPU usage
        time.sleep(PerformanceParams.MAIN_LOOP_DELAY)

    # Wait for threads to complete
    gpu_thread.join()
    cpu_thread.join()

    # Collect final results
    try:
        gpu_result = gpu_result_queue.get_nowait()
        results["GPU"] = gpu_result
    except queue.Empty:
        results["GPU"] = {"success": False, "error": "No result received"}

    try:
        cpu_result = cpu_result_queue.get_nowait()
        results["CPU"] = cpu_result
    except queue.Empty:
        results["CPU"] = {"success": False, "error": "No result received"}

    return results


def main():
    """Main Streamlit app"""

    # Header
    st.markdown(
        f'<div class="main-header">{DefaultValues.blueprint_name} - Efficient Frontier</div>',
        unsafe_allow_html=True,
    )

    # Check imports
    if not IMPORTS_OK:
        st.error(f"❌ Import Error: {IMPORT_ERROR}")
        st.error("Please ensure CVaR src folder is accessible")
        st.stop()

    # Sidebar for parameters
    with st.sidebar:
        st.markdown(
            '<div class="section-header">🎛️ Parameters</div>', unsafe_allow_html=True
        )

        # Dataset selection
        st.subheader(UIText.DATASET_HEADER)
        datasets = get_available_datasets()
        # Set default dataset if available, otherwise use first dataset
        default_index = (
            datasets.index(DefaultValues.DATASET_NAME)
            if DefaultValues.DATASET_NAME in datasets
            else 0
        )
        dataset_name = st.selectbox("Dataset", datasets, index=default_index)

        # Date range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=DefaultValues.START_DATE)
        with col2:
            end_date = st.date_input("End Date", value=DefaultValues.END_DATE)

        regime_name = st.text_input("Regime Name", value=DefaultValues.REGIME_NAME)
        return_type = st.selectbox(
            "Return Type",
            ["LOG", "SIMPLE"],
            index=0 if DefaultValues.RETURN_TYPE == "LOG" else 1,
        )

        # CVaR Parameters
        st.subheader(UIText.CVAR_HEADER)
        col1, col2 = st.columns(2)
        with col1:
            w_min = st.number_input(
                "Min Weight",
                value=DefaultValues.W_MIN,
                min_value=InputLimits.W_MIN_RANGE[0],
                max_value=InputLimits.W_MIN_RANGE[1],
                step=InputLimits.W_STEP,
            )
            c_min = st.number_input(
                "Min Cash",
                value=DefaultValues.C_MIN,
                min_value=InputLimits.C_MIN_RANGE[0],
                max_value=InputLimits.C_MIN_RANGE[1],
                step=InputLimits.C_STEP,
            )
            confidence = st.number_input(
                "Confidence",
                value=DefaultValues.CONFIDENCE,
                min_value=InputLimits.CONFIDENCE_RANGE[0],
                max_value=InputLimits.CONFIDENCE_RANGE[1],
                step=InputLimits.CONFIDENCE_STEP,
            )
        with col2:
            w_max = st.number_input(
                "Max Weight",
                value=DefaultValues.W_MAX,
                min_value=w_min,
                max_value=InputLimits.W_MAX_RANGE[1],
                step=InputLimits.W_STEP,
            )
            c_max = st.number_input(
                "Max Cash",
                value=DefaultValues.C_MAX,
                min_value=c_min,
                max_value=InputLimits.C_MAX_RANGE[1],
                step=InputLimits.C_STEP,
            )
            num_scen = st.number_input(
                "Scenarios",
                value=DefaultValues.NUM_SCEN,
                min_value=InputLimits.NUM_SCEN_RANGE[0],
                max_value=InputLimits.NUM_SCEN_RANGE[1],
                step=InputLimits.NUM_SCEN_STEP,
            )

        L_tar = st.number_input(
            "Leverage Target",
            value=DefaultValues.L_TAR,
            min_value=InputLimits.L_TAR_RANGE[0],
            max_value=InputLimits.L_TAR_RANGE[1],
            step=InputLimits.L_TAR_STEP,
        )
        fit_type = st.selectbox(
            "Fit Type",
            ["kde", "empirical"],
            index=0 if DefaultValues.FIT_TYPE == "kde" else 1,
        )

        # Efficient Frontier Settings
        st.subheader(UIText.FRONTIER_HEADER)
        col1, col2 = st.columns(2)
        with col1:
            ra_num = st.number_input(
                "Number of Portfolios",
                value=DefaultValues.RA_NUM,
                min_value=InputLimits.RA_NUM_RANGE[0],
                max_value=InputLimits.RA_NUM_RANGE[1],
                step=InputLimits.RA_NUM_STEP,
            )
            min_risk_aversion = st.number_input(
                "Min Risk Aversion",
                value=DefaultValues.MIN_RISK_AVERSION,
                min_value=InputLimits.MIN_RISK_AVERSION_RANGE[0],
                max_value=InputLimits.MIN_RISK_AVERSION_RANGE[1],
                step=InputLimits.MIN_RISK_AVERSION_STEP,
                format="%.4f",
            )
        with col2:
            max_risk_aversion = st.number_input(
                "Max Risk Aversion",
                value=DefaultValues.MAX_RISK_AVERSION,
                min_value=InputLimits.MAX_RISK_AVERSION_RANGE[0],
                max_value=InputLimits.MAX_RISK_AVERSION_RANGE[1],
                step=InputLimits.MAX_RISK_AVERSION_STEP,
            )

            # Validation for risk aversion range using utility function
            is_valid, error_msg = validate_risk_aversion_range(
                min_risk_aversion, max_risk_aversion
            )
            if not is_valid:
                if "must be ≤" in error_msg:
                    st.error(error_msg)
                else:
                    st.warning(error_msg)


        # Display Mode (moved before CPU solver to check blog_mode first)
        st.markdown("---")
        st.subheader("📝 Display Mode")
        blog_mode = st.checkbox(
            "Blog Mode",
            value=True,
            help="When enabled, hides CPU solver names in plot titles for cleaner presentation",
        )

        # CPU solver selection (hidden in blog mode, defaults to HIGHS)
        if blog_mode:
            cpu_solver_choice = "HIGHS"
        else:
            # Solver comparison info
            st.subheader(UIText.SOLVER_HEADER)
            cpu_solver_choice = st.selectbox(
                "CPU Solver",
                list(SolverConfig.CPU_SOLVER_OPTIONS.keys()),
                index=list(SolverConfig.CPU_SOLVER_OPTIONS.keys()).index(
                    SolverConfig.DEFAULT_CPU_SOLVER
                ),
                format_func=lambda x: SolverConfig.CPU_SOLVER_OPTIONS[x],
                help=UIText.CPU_SOLVER_HELP,
            )

        # Discretized portfolios
        st.subheader(UIText.DISCRETIZED_HEADER)

        # Add educational info about discretized portfolios
        with st.expander("ℹ️ What are Discretized Portfolios?", expanded=False):
            st.markdown(
                """
            **Discretized portfolios** show all possible portfolio combinations using a finite grid of asset weights.

            **How it works:** Create weight levels (e.g. 0%, 25%, 50%, 75%, 100%) → Generate all combinations → Filter valid portfolios → Calculate performance metrics

            **Caveat:** Searching for portfolio optimizations by brute force means exhaustively evaluating all discretized weight combinations, which rapidly becomes
            computationally infeasible as the number of assets grows due to exponential complexity. Such exhaustive methods are only practical for very small
            portfolios (typically fewer than five assets), while smart algorithms scale far better and yield results much faster.

            **Visual:** Each colored dot = one feasible portfolio. Colors show variance levels (purple=low, yellow=high). Efficient frontier appears as red line on the upper boundary.

            """
            )

        show_discretized = st.checkbox(
            "Show Discretized Portfolios", value=DefaultValues.SHOW_DISCRETIZED
        )
        if show_discretized:
            weight_discretization = st.number_input(
                "Weight Steps",
                value=DefaultValues.WEIGHT_DISCRETIZATION,
                min_value=InputLimits.WEIGHT_DISCRETIZATION_RANGE[0],
                max_value=InputLimits.WEIGHT_DISCRETIZATION_RANGE[1],
                step=InputLimits.WEIGHT_DISCRETIZATION_STEP,
            )

            # Get the number of assets in the selected dataset
            num_assets_in_dataset = get_dataset_num_assets(dataset_name)
            max_assets = st.number_input(
                "Max Assets",
                value=min(DefaultValues.MAX_ASSETS, num_assets_in_dataset),
                min_value=InputLimits.MAX_ASSETS_RANGE[0],
                max_value=num_assets_in_dataset,
                step=1,
                help=UIText.MAX_ASSETS_HELP.format(dataset_name, num_assets_in_dataset),
            )

            # GPU acceleration options for discretized portfolios
            col1, col2 = st.columns(2)
            with col1:
                use_gpu_discretization = st.checkbox(
                    "🚀 GPU Acceleration",
                    value=DefaultValues.USE_GPU_DISCRETIZATION,
                    help=UIText.GPU_ACCELERATION_HELP,
                )
            with col2:
                st.write("")  # Empty space for layout

            total_combinations = weight_discretization**max_assets
            st.info(
                f"📊 Total combinations: {weight_discretization}^{max_assets} = {total_combinations:,}"
            )

            # Check constraint feasibility
            min_possible_weight_sum = max_assets * w_min
            max_possible_weight_sum = max_assets * w_max
            min_required_weight_sum = 1.0 - c_max
            max_allowed_weight_sum = 1.0 - c_min

            # Constraint validation feedback
            if min_possible_weight_sum > max_allowed_weight_sum:
                st.error(UIText.IMPOSSIBLE_CONSTRAINTS)
                show_discretized = False
            elif max_possible_weight_sum < min_required_weight_sum:
                st.error(UIText.IMPOSSIBLE_CONSTRAINTS)
                show_discretized = False

            # Validate combination count
            is_valid, error_msg = validate_combination_count(total_combinations)
            if not is_valid:
                st.error(error_msg)
                show_discretized = False

            if show_discretized:
                discretization_params = {
                    "weight_discretization": weight_discretization,
                    "max_assets": max_assets,
                    "min_weight": w_min,
                    "max_weight": w_max,
                    "use_gpu": use_gpu_discretization,
                }
            else:
                discretization_params = {}
        else:
            discretization_params = {}

        # Run button
        st.markdown("---")
        run_optimization = st.button(UIText.RUN_BUTTON, type="primary", width="stretch")

    # Main content area
    if run_optimization:
        # Validate parameters before running
        if min_risk_aversion > max_risk_aversion:
            st.error(
                "❌ Cannot run optimization: Min Risk Aversion is greater than Max Risk Aversion"
            )
            st.stop()
        if min_risk_aversion <= 0:
            st.warning(
                "⚠️ Min Risk Aversion ≤ 0, adjusting to 0.0001 for logarithmic scale"
            )
            min_risk_aversion = 0.1

        # # Create side-by-side layout for GPU and CPU results
        # st.markdown(
        #     '<div class="section-header">🚀 GPU vs CPU Efficient Frontier Comparison</div>',
        #     unsafe_allow_html=True,
        # )

        # Create columns for side-by-side display with exact equal spacing
        gpu_col, cpu_col = st.columns([1, 1], gap="medium")

        with gpu_col:
            st.markdown("### 🚀 GPU (cuOpt) Results")
            # Use container to ensure consistent sizing
            with st.container():
                gpu_plot_container = st.empty()
                gpu_progress_placeholder = st.empty()

        with cpu_col:
            # In blog mode, hide CPU solver name
            cpu_header = (
                "### 🖥️ CPU Results"
                if blog_mode
                else f"### 🖥️ CPU ({cpu_solver_choice}) Results"
            )
            st.markdown(cpu_header)
            # Use container to ensure consistent sizing
            with st.container():
                cpu_plot_container = st.empty()
                cpu_progress_placeholder = st.empty()

        # Progress at the bottom
        progress_container = st.container()

        # Prepare parameters
        regime_dict = {
            "name": regime_name,
            "range": (start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")),
        }

        cvar_params = CvarParameters(
            w_min=w_min,
            w_max=w_max,
            c_min=c_min,
            c_max=c_max,
            L_tar=L_tar,
            T_tar=None,
            cvar_limit=None,
            risk_aversion=1.0,  # Will be varied
            confidence=confidence,
        )

        # Construct dataset path
        dataset_path = workspace_root / "data" / "stock_data" / f"{dataset_name}.csv"

        if not dataset_path.exists():
            st.error(f"❌ Dataset not found: {dataset_path}")
            st.stop()
        returns_compute_settings = ReturnsComputeSettings(
            return_type=return_type, freq=1
        )
        scenario_generation_settings = ScenarioGenerationSettings(
            num_scen=num_scen,
            fit_type='kde',
            kde_settings=KDESettings(
                bandwidth=0.01,
                kernel='gaussian',
                device='GPU'
            ),
            verbose=False
        )

        # Run progressive optimizations
        with progress_container:
            st.info(
                "🚀 Initializing both GPU and CPU solvers for synchronized start..."
            )

            parallel_results = run_progressive_efficient_frontiers(
                str(dataset_path),
                regime_dict,
                returns_compute_settings,
                scenario_generation_settings,
                cvar_params,
                ra_num,
                min_risk_aversion,
                max_risk_aversion,
                show_discretized,
                discretization_params,
                gpu_plot_container,
                cpu_plot_container,
                gpu_progress_placeholder,
                cpu_progress_placeholder,
                cpu_solver_choice,
                blog_mode,
            )

            st.success("🏁 Synchronized optimization completed!")

        # Display results side-by-side
        gpu_result = parallel_results.get("GPU")
        cpu_result = parallel_results.get("CPU")

        # Display final results and metrics
        if gpu_result and gpu_result["success"]:
            with gpu_col:
                st.markdown("**📊 GPU Performance Metrics**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("⏱️ Total Time", f"{gpu_result['total_time']:.2f}s")
                with col2:
                    st.metric(
                        "⚡ Total Solver Time",
                        f"{gpu_result['total_solver_time']:.2f}s",
                    )
                with col3:
                    avg_solver_time = gpu_result["results_df"]["solve_time"].mean()
                    st.metric("⚡ Avg Solver Time", f"{avg_solver_time:.3f}s")
        else:
            with gpu_col:
                error_msg = gpu_result["error"] if gpu_result else "Unknown error"
                st.error(f"❌ GPU optimization failed: {error_msg}")

        # Display CPU results
        if cpu_result and cpu_result["success"]:
            with cpu_col:
                st.markdown("**📊 CPU Performance Metrics**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("⏱️ Total Time", f"{cpu_result['total_time']:.2f}s")
                with col2:
                    st.metric(
                        "⚡ Total Solver Time",
                        f"{cpu_result['total_solver_time']:.2f}s",
                    )
                with col3:
                    avg_solver_time = cpu_result["results_df"]["solve_time"].mean()
                    st.metric("⚡ Avg Solver Time", f"{avg_solver_time:.3f}s")
        else:
            with cpu_col:
                error_msg = cpu_result["error"] if cpu_result else "Unknown error"
                st.error(f"❌ CPU optimization failed: {error_msg}")

        # Performance comparison summary
        if (
            gpu_result
            and gpu_result["success"]
            and cpu_result
            and cpu_result["success"]
        ):
            with progress_container:
                # In blog mode, hide CPU solver name
                cpu_name = "CPU" if blog_mode else f"CPU ({cpu_solver_choice})"
                st.success(f"✅ Solver comparison completed! GPU (cuOpt) vs {cpu_name}")
                st.markdown(
                    '<div class="section-header">📊 Performance Comparison</div>',
                    unsafe_allow_html=True,
                )

                col1, col2, col3 = st.columns(3)

                with col1:
                    gpu_solver_time = gpu_result["total_solver_time"]
                    cpu_solver_time = cpu_result["total_solver_time"]

                    st.metric("🎯 Number of Portfolios", len(gpu_result["results_df"]))

                    if gpu_solver_time < cpu_solver_time:
                        speedup = cpu_solver_time / gpu_solver_time
                        st.metric("⚡ GPU Solver Speedup", f"{speedup:.1f}x faster")
                    else:
                        speedup = gpu_solver_time / cpu_solver_time
                        st.metric(
                            "⚡ CPU Solver Performance",
                            f"{speedup:.1f}x faster than GPU",
                        )


                with col2:
                    gpu_total_time = gpu_result["total_time"]
                    cpu_total_time = cpu_result["total_time"]
                    st.metric("🚀 GPU Total Time", f"{gpu_total_time:.2f}s")
                    st.metric("🖥️ CPU Total Time", f"{cpu_total_time:.2f}s")

                with col3:
                    st.metric("⚡ GPU Solver Time", f"{gpu_solver_time:.2f}s")
                    st.metric("⚡ CPU Solver Time", f"{cpu_solver_time:.2f}s")

                # Show detailed results
                with st.expander("📋 Detailed Comparison", expanded=False):
                    if (
                        gpu_result["results_df"] is not None
                        and cpu_result["results_df"] is not None
                    ):
                        gpu_display = gpu_result["results_df"][
                            ["risk_aversion", "risk", "return", "solve_time"]
                        ].copy()
                        gpu_display.columns = [
                            "Risk Aversion",
                            "GPU CVaR",
                            "GPU Return",
                            "GPU Time (s)",
                        ]

                        cpu_display = cpu_result["results_df"][
                            ["risk_aversion", "risk", "return", "solve_time"]
                        ].copy()
                        cpu_display.columns = [
                            "Risk Aversion",
                            "CPU CVaR",
                            "CPU Return",
                            "CPU Time (s)",
                        ]

                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**GPU (cuOpt) Results**")
                            st.dataframe(gpu_display, hide_index=True)
                        with col2:
                            # In blog mode, hide CPU solver name
                            cpu_results_header = (
                                "**CPU Results**"
                                if blog_mode
                                else f"**CPU ({cpu_solver_choice}) Results**"
                            )
                            st.markdown(cpu_results_header)
                            st.dataframe(cpu_display, hide_index=True)

                # Special Portfolios Analysis
                st.markdown(
                    '<div class="section-header">🎯 Special Portfolios Analysis</div>',
                    unsafe_allow_html=True,
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**🚀 GPU Special Portfolios**")
                    display_special_portfolios_table(gpu_result["results_df"])

                with col2:
                    # In blog mode, hide CPU solver name
                    cpu_special_header = (
                        "**🖥️ CPU Special Portfolios**"
                        if blog_mode
                        else f"**🖥️ CPU ({cpu_solver_choice}) Special Portfolios**"
                    )
                    st.markdown(cpu_special_header)
                    display_special_portfolios_table(cpu_result["results_df"])

    else:
        # Show instructions
        st.info(UIText.CONFIGURE_INSTRUCTION)

        st.markdown(
            """
        ### 🚀 **Features:**
        - **Synchronized Racing**: Both GPU and CPU solvers initialize, then start optimization simultaneously for fair comparison
        - **Multiple CPU Solvers**: Choose from HIGHS, CLARABEL, ECOS, OSQP, or SCS
        - **Real-Time Visualization**: See portfolios being added to efficient frontiers as they're solved
        - **Special Portfolio Markers**: Automatically identify and mark Min Variance, Max Sharpe, and Max Return portfolios
        - **Progressive Racing**: Compare GPU vs CPU speed with live progress bars and timing
        - **Discretized Portfolios**: Educational visualization of all feasible portfolio combinations
        - **Logarithmic Risk Aversion**: Use logarithmically spaced risk aversion values for better coverage
        - **Live Metrics**: Real-time solve times, progress tracking, and performance comparison

        ### 📊 **How it works:**
        1. **Setup**: Configure your dataset, date range, CVaR parameters, and CPU solver
        2. **Discretization**: (Optional) Show all possible portfolio combinations
        3. **Risk Aversion Range**: Set logarithmic range of risk aversion values
        4. **Synchronized Start**: Both solvers initialize, then begin racing simultaneously
        5. **Live Racing**: Watch real-time performance differences as portfolios are solved
        6. **Special Portfolios**: See highlighted Min Variance (🔷), Max Sharpe (🔸), and Max Return (🔹) portfolios
        """
        )


if __name__ == "__main__":
    main()
