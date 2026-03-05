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
CVaR Rebalancing Strategies - Streamlit App

Interactive app to simulate dynamic rebalancing strategies with CVaR optimization.
Mirrors the efficient frontier app layout: left panel for inputs, right panel shows
GPU vs CPU side-by-side progressive results. Per period, the app displays the
backtest performance, whether re-optimization was triggered, and updates the plot.

Author: phuo-nv
"""

from __future__ import annotations

import copy
import queue
import sys
import threading
import time
import traceback
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# App parameters (reuse styling and controls from efficient frontier app)
from app_parameters import (
    DefaultValues,
    InputLimits,
    MatplotlibConfig,
    PerformanceParams,
    PlotStyling,
    SolverConfig,
    UIText,
    get_color_scheme,
)

warnings.filterwarnings("ignore")

# Global lock for matplotlib operations to prevent thread conflicts
_matplotlib_lock = threading.Lock()

# Add workspace root to path for imports (package structure: cufolio = "src")
script_dir = Path(__file__).parent.absolute()
workspace_root = script_dir.parent  # This is the workspace root
sys.path.insert(0, str(workspace_root))
cvar_dir = workspace_root  # For backward compatibility with path references

try:
    # Import cufolio package
    import cvxpy as cp
    from cufolio import backtest, cvar_optimizer, cvar_utils, rebalance, utils
    from cufolio.cvar_parameters import CvarParameters
    from cufolio.settings import (
        KDESettings,
        ReturnsComputeSettings,
        ScenarioGenerationSettings,
    )

    IMPORTS_OK = True
except Exception as e:
    IMPORTS_OK = False
    IMPORT_ERROR = str(e)


# Page config
st.set_page_config(
    page_title="Backtesting Rebalancing Strategies",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Custom CSS for better appearance (matching efficient frontier app)
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
    try:
        data_dir = workspace_root / "data" / "stock_data"
        if data_dir.exists():
            return sorted([f.stem for f in data_dir.glob("*.csv")])
        return ["dow30", "sp500", "sp100", "baby_dataset"]
    except Exception:
        return ["dow30", "sp500", "sp100", "baby_dataset"]


def get_dataset_num_assets(dataset_name):
    """Get number of assets in dataset with fallback."""
    try:
        dataset_path = workspace_root / "data" / "stock_data" / f"{dataset_name}.csv"
        if dataset_path.exists():
            df = pd.read_csv(dataset_path)
            return max(1, len(df.columns) - 1)
    except Exception:
        pass
    return 30  # Fallback


def _create_error_result(solver_name, error_msg):
    """Create standardized error result for rebalancing."""
    return {
        "success": False,
        "solver_name": solver_name,
        "results_df": None,
        "cumulative_series": None,
        "baseline_series": None,
        "fig": None,
        "total_solve_time": 0.0,
        "total_elapsed_time": 0.0,
        "rebal_count": 0,
        "error": error_msg,
    }


def _set_axis_limits(ax, cum_series, bh_series):
    """Set dynamic axis limits to fit both curves with padding."""
    all_values = list(cum_series.values) + list(bh_series.values)
    y_min, y_max = min(all_values), max(all_values)
    y_range = y_max - y_min
    padding = y_range * 0.05
    ax.set_ylim(y_min - padding, y_max + padding)

    all_dates = list(cum_series.index) + list(bh_series.index)
    if all_dates:
        min_date, max_date = min(all_dates), max(all_dates)
        date_range = (max_date - min_date).days
        date_padding = pd.Timedelta(days=max(1, date_range * 0.02))
        ax.set_xlim(min_date - date_padding, max_date + date_padding)


def _init_rebalancing_figure(title_suffix: str, xlim=None, ylim=None):
    """Initialize rebalancing figure with matplotlib lock for thread safety."""
    with _matplotlib_lock:
        colors = get_color_scheme()
        plt.style.use(MatplotlibConfig.STYLE)
        sns.set_context(MatplotlibConfig.CONTEXT, font_scale=MatplotlibConfig.FONT_SCALE)

        fig, ax = plt.subplots(
            figsize=PlotStyling.FIGURE_SIZE,
            dpi=PlotStyling.FIGURE_DPI,
            facecolor=colors["background"],
            tight_layout=True,
        )
        ax.set_facecolor(colors["background"])
        ax.grid(True, alpha=PlotStyling.GRID_ALPHA, color=colors["grid"])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(PlotStyling.SPINE_COLOR)
        ax.spines["bottom"].set_color(PlotStyling.SPINE_COLOR)
        ax.set_xlabel(
            "Date", fontsize=PlotStyling.XLABEL_FONTSIZE, fontweight=PlotStyling.FONT_WEIGHT
        )
        ax.set_ylabel(
            "Cumulative Portfolio Value",
            fontsize=PlotStyling.YLABEL_FONTSIZE,
            fontweight=PlotStyling.FONT_WEIGHT,
        )
        ax.set_title(
            f"Rebalancing Strategy {title_suffix}",
            fontsize=PlotStyling.TITLE_FONTSIZE,
            fontweight=PlotStyling.FONT_WEIGHT,
            pad=PlotStyling.TITLE_PAD,
        )

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        fig.subplots_adjust(left=0.1, right=0.95, top=0.88, bottom=0.12)
    return fig, ax, colors


def create_rebalancing_progressive(
    dataset_path: str,
    trading_range: tuple,
    returns_compute_settings: dict,
    scenario_generation_settings: dict,
    cvar_params: CvarParameters,
    solver_settings: dict,
    look_back_window: int,
    look_forward_window: int,
    re_optimize_criteria: dict,
    transaction_cost_factor: float,
    solver_name: str,
    strategy_display_name: str,
    progress_queue: "queue.Queue",
    result_queue: "queue.Queue",
    start_event: threading.Event,
):
    """Run progressive rebalancing with per-period updates via queues."""
    try:
        # Initialize CUDA context for GPU thread
        is_gpu = solver_settings.get("solver") == cp.CUOPT
        if is_gpu:
            try:
                import cupy as cupy_cuda
                cupy_cuda.cuda.Device(0).use()
            except Exception:
                pass

        progress_queue.put(
            {
                "solver": solver_name,
                "status": "initializing",
                "message": f"{solver_name}: Initializing...",
            }
        )

        start_date, end_date = trading_range

        # Initialize figure with dynamic axis limits
        fig, ax, colors = _init_rebalancing_figure(
            f"- {strategy_display_name}", None, None
        )

        # Create CVaR parameters for initial portfolio (without turnover constraint for baseline)
        # The baseline computation should not use turnover since there's no previous portfolio
        initial_cvar_params = copy.deepcopy(cvar_params)
        initial_cvar_params.T_tar = (
            None  # No turnover constraint for initial optimization
        )

        # Create rebalancing object (this computes baseline with initial_cvar_params)
        r = rebalance.rebalance_portfolio(
            dataset_directory=dataset_path,
            returns_compute_settings=returns_compute_settings,
            scenario_generation_settings=scenario_generation_settings,
            trading_start=start_date,
            trading_end=end_date,
            look_forward_window=look_forward_window,
            look_back_window=look_back_window,
            cvar_params=initial_cvar_params,  # Use params without turnover for baseline
            solver_settings=solver_settings,
            re_optimize_criteria=re_optimize_criteria,
            print_opt_result=False,
        )

        # But keep the original cvar_params (with turnover) for reoptimizations
        r.cvar_params = cvar_params

        # Baseline (buy & hold) line matching rebalance.py color scheme
        bh_series = r.buy_and_hold_cumulative_portfolio_value
        ax.plot(
            bh_series.index,
            bh_series.values,
            linewidth=2.5,
            color=colors["benchmark"][0],  # Orange - matches rebalance.py
            linestyle="-",
            alpha=0.8,
            label="Buy & Hold",
            zorder=2,
        )

        # Prepare dynamic state
        cumulative_values = np.array([])
        cumulative_dates = []
        rebal_dates = []

        # Compute total periods for progress
        total_periods = 0
        idx_tmp = 0
        while idx_tmp < len(r.dates_range) - look_forward_window:
            total_periods += 1
            idx_tmp += look_forward_window

        # Announce ready and wait for sync
        progress_queue.put(
            {
                "solver": solver_name,
                "status": "ready",
                "message": f"{solver_name}: Ready to start - waiting for synchronization",
            }
        )
        start_event.wait()

        progress_queue.put(
            {
                "solver": solver_name,
                "status": "starting",
                "message": f"{solver_name}: Starting rebalancing",
            }
        )

        # Always send a plot update and synchronization checkpoint before optimization
        # This ensures both solvers are at the same point regardless of initialization timing
        progress_queue.put(
            {
                "solver": solver_name,
                "status": "plot_ready",
                "figure": fig,
                "message": f"{solver_name}: Plot initialized - ready for optimization",
            }
        )

        # Minimal delay to ensure both threads reach this point before starting optimization
        time.sleep(PerformanceParams.THREAD_SYNC_DELAY)

        progress_queue.put(
            {
                "solver": solver_name,
                "status": "starting_optimization",
                "message": f"{solver_name}: Starting rebalancing optimization NOW!",
            }
        )

        # Main loop (adapted from cufolio/rebalance.py)
        # Create column names without duplicate pct_change
        metric_column = r.re_optimize_type
        results_df = pd.DataFrame(
            columns=[metric_column, "re_optimized", "portfolio_value", "max_drawdown"]
        )
        current_portfolio = r.initial_portfolio  # set by baseline computation
        prev_portfolio = None  # Initialize previous portfolio
        portfolio_value = 1.0
        backtest_idx = 0
        backtest_date = r.trading_start
        backtest_final_date = pd.Timestamp(r.dates_range[-look_forward_window])
        period_counter = 0
        total_solve_time = 0.0

        # Track total elapsed time for real-time display
        start_time_overall = time.time()

        while pd.Timestamp(backtest_date) < backtest_final_date:
            period_counter += 1
            # Record current state (ensure scalar values)
            results_df.loc[backtest_date, metric_column] = np.nan
            results_df.loc[backtest_date, "re_optimized"] = False
            results_df.loc[backtest_date, "portfolio_value"] = float(portfolio_value)
            results_df.loc[backtest_date, "max_drawdown"] = (
                0.0  # Initialize max_drawdown
            )

            # Determine if we need to (re)optimize at the start of this period
            reopt_triggered = False
            reopt_metric = None
            solve_time = 0.0

            # Use the initial portfolio computed during baseline (buy & hold) creation
            # No need to re-optimize at the start since we already have the optimal portfolio from baseline computation
            if period_counter == 1:
                # Ensure we have the initial portfolio from the rebalancing object
                if current_portfolio is None:
                    raise ValueError(
                        f"{solver_name}: Initial portfolio not found - baseline computation may have failed"
                    )
                # Add message indicating we're reusing the baseline optimization
                progress_queue.put(
                    {
                        "solver": solver_name,
                        "status": "reusing_baseline",
                        "message": f"{solver_name}: Using initial portfolio from baseline computation (no re-optimization needed)",
                    }
                )

            # Backtest this period with current_portfolio
            bt_start = backtest_date.strftime("%Y-%m-%d")
            bt_end = r.dates_range[backtest_idx + look_forward_window].strftime(
                "%Y-%m-%d"
            )
            bt_regime = {"name": "backtest", "range": (bt_start, bt_end)}
            test_returns = utils.calculate_returns(
                dataset_path, bt_regime, returns_compute_settings
            )

            bt = backtest.portfolio_backtester(
                current_portfolio, test_returns, benchmark_portfolios=None
            )
            bt_result = bt.backtest_single_portfolio(current_portfolio)

            # Update cumulative values and dates
            cum_ret_values = (
                bt_result["cumulative returns"].values[0]
                if hasattr(bt_result["cumulative returns"], "values")
                else bt_result["cumulative returns"]
            )
            cur_cum = cum_ret_values * portfolio_value
            cumulative_values = np.concatenate((cumulative_values, cur_cum))
            cumulative_dates.extend(bt._dates)

            # Compute pct change and transaction cost for this period
            pct_change_raw = r._calculate_pct_change(bt_result)
            pct_change = (
                float(pct_change_raw)
                if hasattr(pct_change_raw, "item")
                else float(pct_change_raw)
            )
            if prev_portfolio is None:
                # No transaction cost for the first period
                tx_cost = 0.0
            else:
                tx_cost = r._calculate_transaction_cost(
                    current_portfolio, prev_portfolio, transaction_cost_factor
                )
            portfolio_value = portfolio_value * (1 + pct_change - tx_cost)

            # Store current portfolio as previous for next iteration
            prev_portfolio = current_portfolio
            if r.re_optimize_type == "pct_change":
                # Ensure results_df has proper data types and no missing values for pct_change method
                try:
                    results_df_clean = results_df.copy()
                    results_df_clean[metric_column] = (
                        results_df_clean[metric_column].fillna(0.0).astype(float)
                    )
                    results_df_clean["re_optimized"] = (
                        results_df_clean["re_optimized"].fillna(False).astype(bool)
                    )
                    reopt_metric, reopt_triggered = r._check_pct_change(
                        float(pct_change), bt_result, results_df_clean
                    )
                except Exception:
                    # Fallback: simple threshold check
                    reopt_triggered = pct_change < r.re_optimize_threshold
                    reopt_metric = pct_change
            elif r.re_optimize_type == "drift_from_optimal":
                reopt_metric, reopt_triggered = r._check_drift_from_optimal(
                    current_portfolio, backtest_idx
                )
            elif r.re_optimize_type == "max_drawdown":
                mdd = float(
                    bt_result["max drawdown"].values[0]
                    if hasattr(bt_result["max drawdown"], "values")
                    else bt_result["max drawdown"]
                )
                reopt_triggered = r._check_max_drawdown(mdd)
                reopt_metric = mdd
            elif r.re_optimize_type == "no_re_optimize":
                reopt_triggered = False
                reopt_metric = None

            # If triggered, run re-optimization for the next period's portfolio
            if reopt_triggered:
                opt_start = backtest_date - pd.Timedelta(days=look_back_window)
                opt_regime = {
                    "name": "re-optimize",
                    "range": (
                        opt_start.strftime("%Y-%m-%d"),
                        backtest_date.strftime("%Y-%m-%d"),
                    ),
                }
                # Compute returns from price data
                opt_returns = utils.calculate_returns(
                    dataset_path, opt_regime, returns_compute_settings
                )

                # Generate return scenarios
                opt_returns = cvar_utils.generate_cvar_data(
                    opt_returns,
                    scenario_generation_settings
                )

                # For turnover constraint: pass existing portfolio if T_tar is enabled, None otherwise
                existing_ptf_for_turnover = (
                    current_portfolio if r.cvar_params.T_tar is not None else None
                )

                if r.cvar_params.T_tar is not None:
                    progress_queue.put(
                        {
                            "solver": solver_name,
                            "status": "applying_turnover",
                            "message": f"{solver_name}: Reoptimizing with turnover constraint (T_tar={r.cvar_params.T_tar:.3f})",
                        }
                    )

                opt_problem = cvar_optimizer.CVaR(
                    returns_dict=opt_returns,
                    cvar_params=r.cvar_params,
                    existing_portfolio=existing_ptf_for_turnover,
                )
                t0 = time.time()
                solver_result, current_portfolio = opt_problem.solve_optimization_problem(
                    r.solver_settings, print_results=False
                )
                solve_time = solver_result["solve time"]
                total_solve_time += solve_time
                rebal_dates.append(backtest_date)

            # Record results for this period (ensure scalar values)
            results_df.loc[backtest_date, metric_column] = (
                float(reopt_metric) if reopt_metric is not None else np.nan
            )
            results_df.loc[backtest_date, "re_optimized"] = bool(reopt_triggered)
            results_df.loc[backtest_date, "max_drawdown"] = float(
                bt_result["max drawdown"].values[0]
                if hasattr(bt_result["max drawdown"], "values")
                else bt_result["max drawdown"]
            )
            results_df.loc[backtest_date, "portfolio_value"] = float(portfolio_value)

            # Update plot with new data (with matplotlib lock for thread safety)
            try:
                with _matplotlib_lock:
                    cum_series = pd.Series(
                        cumulative_values,
                        index=pd.to_datetime(cumulative_dates),
                        name="cumulative_portfolio_value",
                    )
                    ax.plot(
                        cum_series.index,
                        cum_series.values,
                        linewidth=3,
                        color=colors["frontier"],
                        alpha=0.9,
                        label="Dynamic Rebalancing" if period_counter == 1 else None,
                        zorder=3,
                    )

                    # Add rebalancing date markers
                    if rebal_dates:
                        rebalancing_values = []
                        rebalancing_dates_clean = []

                        for date in rebal_dates:
                            date_ts = pd.Timestamp(date)
                            if date_ts in cum_series.index:
                                rebalancing_values.append(float(cum_series[date_ts]))
                                rebalancing_dates_clean.append(date_ts)
                            else:
                                # Find nearest date
                                nearest_idx = cum_series.index.get_indexer(
                                    [date_ts], method="nearest"
                                )[0]
                                if nearest_idx >= 0:
                                    nearest_date = cum_series.index[nearest_idx]
                                    rebalancing_values.append(
                                        float(cum_series[nearest_date])
                                    )
                                    rebalancing_dates_clean.append(nearest_date)

                        if rebalancing_dates_clean:
                            # Set axis limits to fit both curves
                            _set_axis_limits(ax, cum_series, bh_series)

                            # Calculate line height for rebalancing markers
                            y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                            line_height = y_range * 0.05

                            # Create small vertical line segments at each rebalancing date
                            for date, value in zip(
                                rebalancing_dates_clean, rebalancing_values
                            ):
                                ax.vlines(
                                    date,
                                    value - line_height,
                                    value + line_height,
                                    color=colors["assets"],
                                    linewidth=2.5,
                                    alpha=0.9,
                                    linestyle="--",
                                    zorder=5,
                                )
                                ax.plot(
                                    date,
                                    value,
                                    "o",
                                    color=colors["assets"],
                                    markersize=7,
                                    markeredgecolor="white",
                                    markeredgewidth=1.5,
                                    zorder=6,
                                )
                    else:
                        # Set axis limits to fit both curves
                        _set_axis_limits(ax, cum_series, bh_series)

                    # Create custom legend
                    from matplotlib.lines import Line2D
                    if ax.legend_:
                        ax.legend_.remove()

                    legend_elements = [
                        Line2D(
                            [0],
                            [0],
                            color=colors["frontier"],
                            linewidth=3,
                            label="Dynamic Rebalancing",
                        ),
                        Line2D(
                            [0],
                            [0],
                            color=colors["benchmark"][0],
                            linewidth=2.5,
                            label="Buy & Hold",
                        ),
                    ]

                    if rebal_dates:
                        legend_elements.append(
                            Line2D(
                                [0],
                                [0],
                                color=colors["assets"],
                                linewidth=2.5,
                                linestyle="--",
                                marker="o",
                                markersize=7,
                                markerfacecolor=colors["assets"],
                                markeredgecolor="white",
                                markeredgewidth=1.5,
                                label="Rebalancing Dates",
                            )
                        )

                    ax.legend(
                        handles=legend_elements,
                        loc=PlotStyling.LEGEND_LOCATION,
                        frameon=PlotStyling.LEGEND_FRAMEON,
                        fancybox=PlotStyling.LEGEND_FANCYBOX,
                        shadow=PlotStyling.LEGEND_SHADOW,
                        framealpha=PlotStyling.LEGEND_FRAMEALPHA,
                        fontsize=PlotStyling.LEGEND_FONTSIZE,
                    )
            except Exception:
                pass

            # Send progress and plot updates
            current_total_time = time.time() - start_time_overall

            # Progress update (every period)
            progress_queue.put(
                {
                    "solver": solver_name,
                    "status": "period_progress",
                    "period": period_counter,
                    "total_periods": total_periods,
                    "reoptimized": bool(reopt_triggered),
                    "solve_time": solve_time,
                    "total_solve_time": total_solve_time,
                    "total_elapsed_time": current_total_time,
                    "metric": reopt_metric,
                    "portfolio_value": float(portfolio_value),
                    "message": f"{solver_name}: Period {period_counter}/{total_periods} | Re-opt: {'Yes' if reopt_triggered else 'No'}",
                }
            )

            # Plot update (controlled frequency)
            if (
                period_counter % max(1, PerformanceParams.PLOT_UPDATE_FREQUENCY) == 0
            ) or (period_counter == total_periods):
                progress_queue.put(
                    {
                        "solver": solver_name,
                        "status": "period_plot_update",
                        "figure": fig,
                        "period": period_counter,
                        "total_periods": total_periods,
                    }
                )

            # Advance
            backtest_idx += look_forward_window
            backtest_date = r.dates_range[backtest_idx]
            time.sleep(PerformanceParams.UI_UPDATE_DELAY)

        # Finalize
        results_df.index.name = "date"
        cum_series = pd.Series(
            cumulative_values,
            index=pd.to_datetime(cumulative_dates),
            name="cumulative_portfolio_value",
        )

        # Calculate final total elapsed time
        final_total_time = time.time() - start_time_overall

        # Send completion with final plot simultaneously to avoid lag
        progress_queue.put(
            {
                "solver": solver_name,
                "status": "completed",
                "total_elapsed_time": final_total_time,
                "total_solve_time": total_solve_time,
                "figure": fig,  # Include final figure with completion message
                "message": f"{solver_name}: Completed {period_counter} periods in {final_total_time:.2f}s",
            }
        )
        result_queue.put(
            {
                "success": True,
                "solver_name": solver_name,
                "results_df": results_df,
                "cumulative_series": cum_series,
                "baseline_series": bh_series,
                "fig": fig,
                "total_solve_time": total_solve_time,
                "total_elapsed_time": final_total_time,
                "rebal_count": len(rebal_dates),
                "error": None,
            }
        )

    except Exception as e:
        progress_queue.put(
            {
                "solver": solver_name,
                "status": "error",
                "message": f"{solver_name}: Error - {str(e)}",
            }
        )
        result_queue.put(_create_error_result(solver_name, traceback.format_exc()))


def run_progressive_rebalancing(
    dataset_path: str,
    trading_range: tuple,
    returns_compute_settings: dict,
    scenario_generation_settings: dict,
    cvar_params: CvarParameters,
    look_back_window: int,
    look_forward_window: int,
    re_optimize_criteria: dict,
    transaction_cost_factor: float,
    strategy_display_name: str,
    gpu_plot_container,
    cpu_plot_container,
    gpu_progress_placeholder,
    cpu_progress_placeholder,
    cpu_solver_choice: str,
    blog_mode: bool = True,
):
    """Run GPU and CPU rebalancing in parallel with progressive updates."""

    # Build solver settings
    gpu_settings = {"solver": cp.CUOPT, "verbose": SolverConfig.SOLVER_VERBOSE}
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

    # Queues
    gpu_progress_q: queue.Queue = queue.Queue()
    cpu_progress_q: queue.Queue = queue.Queue()
    gpu_result_q: queue.Queue = queue.Queue()
    cpu_result_q: queue.Queue = queue.Queue()

    # Dynamic axis limits will be calculated during plotting based on actual data

    # Create and display empty plots immediately for simultaneous appearance
    # Create identical empty plots for both GPU and CPU with strategy name
    def create_empty_plot():
        with _matplotlib_lock:
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
                "Date",
                fontsize=PlotStyling.XLABEL_FONTSIZE,
                fontweight=PlotStyling.FONT_WEIGHT,
            )
            ax.set_ylabel(
                "Cumulative Portfolio Value",
                fontsize=PlotStyling.YLABEL_FONTSIZE,
                fontweight=PlotStyling.FONT_WEIGHT,
            )
            ax.set_title(
                f"Rebalancing Strategy - {strategy_display_name}",
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
    gpu_empty_fig = create_empty_plot()
    cpu_empty_fig = create_empty_plot()

    # Display both plots at exactly the same time with consistent sizing
    gpu_plot_container.pyplot(gpu_empty_fig, width="stretch")
    cpu_plot_container.pyplot(cpu_empty_fig, width="stretch")

    # Clean up the temporary figures
    with _matplotlib_lock:
        plt.close(gpu_empty_fig)
        plt.close(cpu_empty_fig)

    # Create synchronization event to ensure simultaneous start
    start_event = threading.Event()

    # Create separate copies of cvar_params for thread isolation
    gpu_cvar_params = copy.deepcopy(cvar_params)
    cpu_cvar_params = copy.deepcopy(cvar_params)

    gpu_thread = threading.Thread(
        target=create_rebalancing_progressive,
        args=(
            dataset_path,
            trading_range,
            returns_compute_settings,
            scenario_generation_settings,
            gpu_cvar_params,
            gpu_settings,
            look_back_window,
            look_forward_window,
            re_optimize_criteria,
            transaction_cost_factor,
            "GPU (cuOpt)",
            strategy_display_name,
            gpu_progress_q,
            gpu_result_q,
            start_event,
        ),
        name="GPU-Thread",
        daemon=True,
    )

    cpu_display_name = "CPU" if blog_mode else f"CPU ({cpu_solver_choice})"

    cpu_thread = threading.Thread(
        target=create_rebalancing_progressive,
        args=(
            dataset_path,
            trading_range,
            returns_compute_settings,
            scenario_generation_settings,
            cpu_cvar_params,
            cpu_settings,
            look_back_window,
            look_forward_window,
            re_optimize_criteria,
            transaction_cost_factor,
            cpu_display_name,
            strategy_display_name,
            cpu_progress_q,
            cpu_result_q,
            start_event,
        ),
        name="CPU-Thread",
        daemon=True,
    )

    # Start threads with delay to prevent CUDA conflicts
    gpu_thread.start()
    time.sleep(0.1)
    cpu_thread.start()
    time.sleep(PerformanceParams.INITIALIZATION_DELAY)

    # Show synchronization message
    with gpu_progress_placeholder.container():
        st.info(UIText.GPU_SYNCHRONIZED)
    with cpu_progress_placeholder.container():
        st.info(UIText.CPU_SYNCHRONIZED.format(cpu_solver_choice))

    start_event.set()  # Signal both threads to start optimization simultaneously

    # Track completion status
    gpu_done = False
    cpu_done = False
    gpu_started = False
    cpu_started = False

    # Main loop: drain queues frequently
    while not (gpu_done and cpu_done):
        # GPU updates - process ALL pending progress updates for immediate response
        gpu_processed_plot = False
        try:
            while True:  # Process all pending GPU updates
                upd = gpu_progress_q.get_nowait()
                status = upd.get("status")
                if status in ("initializing", "ready", "starting"):
                    with gpu_progress_placeholder.container():
                        st.info(upd.get("message", ""))
                elif status == "period_progress":
                    with gpu_progress_placeholder.container():
                        period = upd.get("period", 0)
                        total = max(1, upd.get("total_periods", 1))
                        st.info(upd.get("message", ""))
                        st.progress(min(1.0, period / total))

                        # Clean progress display
                        st.caption(
                            f"**Period {period}/{total}** | Portfolio: ${upd.get('portfolio_value', 0.0):.3f}"
                        )
                elif status == "period_plot_update" and not gpu_processed_plot:
                    # Update GPU plot (heavier update, only one per loop iteration)
                    fig = upd.get("figure")
                    if fig is not None:
                        gpu_plot_container.pyplot(fig, width="stretch")
                        gpu_processed_plot = True
                elif status == "plot_ready":
                    # Display the plot ready state - ensures synchronization checkpoint (always show)
                    fig = upd.get("figure")
                    if fig is not None:
                        gpu_plot_container.pyplot(fig, width="stretch")
                    with gpu_progress_placeholder.container():
                        st.success(upd.get("message", ""))
                elif status == "ready":
                    with gpu_progress_placeholder.container():
                        st.warning(upd.get("message", ""))
                    gpu_started = True
                elif status == "starting":
                    with gpu_progress_placeholder.container():
                        st.info(upd.get("message", ""))
                    gpu_started = True
                elif status in [
                    "starting_optimization",
                    "reusing_baseline",
                    "applying_turnover",
                ]:
                    with gpu_progress_placeholder.container():
                        st.info(upd.get("message", ""))
                elif status == "completed":
                    # Display final plot immediately with completion (if included)
                    fig = upd.get("figure")
                    if fig is not None:
                        gpu_plot_container.pyplot(fig, width="stretch")

                    with gpu_progress_placeholder.container():
                        st.success(
                            f"GPU completed in {upd.get('total_elapsed_time', 0.0):.2f}s"
                        )
                    gpu_done = True
                    break
                elif status == "error":
                    with gpu_progress_placeholder.container():
                        st.error(upd.get("message", "GPU error"))
                    gpu_done = True
                    break
        except queue.Empty:
            pass

        # CPU updates - process ALL pending progress updates for immediate response
        cpu_processed_plot = False
        try:
            while True:  # Process all pending CPU updates
                upd = cpu_progress_q.get_nowait()
                status = upd.get("status")
                if status in ("initializing", "ready", "starting"):
                    with cpu_progress_placeholder.container():
                        st.info(upd.get("message", ""))
                elif status == "period_progress":
                    with cpu_progress_placeholder.container():
                        period = upd.get("period", 0)
                        total = max(1, upd.get("total_periods", 1))
                        st.info(upd.get("message", ""))
                        st.progress(min(1.0, period / total))

                        # Clean progress display
                        st.caption(
                            f"**Period {period}/{total}** | Portfolio: ${upd.get('portfolio_value', 0.0):.3f}"
                        )
                elif status == "period_plot_update" and not cpu_processed_plot:
                    # Update CPU plot (heavier update, only one per loop iteration)
                    fig = upd.get("figure")
                    if fig is not None:
                        cpu_plot_container.pyplot(fig, width="stretch")
                        cpu_processed_plot = True
                elif status == "plot_ready":
                    # Display the plot ready state - ensures synchronization checkpoint (always show)
                    fig = upd.get("figure")
                    if fig is not None:
                        cpu_plot_container.pyplot(fig, width="stretch")
                    with cpu_progress_placeholder.container():
                        st.success(upd.get("message", ""))
                elif status == "ready":
                    with cpu_progress_placeholder.container():
                        st.warning(upd.get("message", ""))
                    cpu_started = True
                elif status == "starting":
                    with cpu_progress_placeholder.container():
                        st.info(upd.get("message", ""))
                    cpu_started = True
                elif status in [
                    "starting_optimization",
                    "reusing_baseline",
                    "applying_turnover",
                ]:
                    with cpu_progress_placeholder.container():
                        st.info(upd.get("message", ""))
                elif status == "completed":
                    # Display final plot immediately with completion (if included)
                    fig = upd.get("figure")
                    if fig is not None:
                        cpu_plot_container.pyplot(fig, width="stretch")

                    with cpu_progress_placeholder.container():
                        st.success(
                            f"CPU completed in {upd.get('total_elapsed_time', 0.0):.2f}s"
                        )
                    cpu_done = True
                    break
                elif status == "error":
                    with cpu_progress_placeholder.container():
                        st.error(upd.get("message", "CPU error"))
                    cpu_done = True
                    break
        except queue.Empty:
            pass

        # Check if both are ready and show synchronized start message
        if gpu_started and cpu_started and not gpu_done and not cpu_done:
            with gpu_progress_placeholder.container():
                st.success(UIText.RACE_STARTED_GPU)
            with cpu_progress_placeholder.container():
                st.success(UIText.RACE_STARTED_CPU)
            gpu_started = False  # Reset to avoid repeated messages
            cpu_started = False

        time.sleep(PerformanceParams.MAIN_LOOP_DELAY)

    gpu_thread.join()
    cpu_thread.join()

    # Gather results
    out = {}
    try:
        out["GPU"] = gpu_result_q.get_nowait()
    except queue.Empty:
        out["GPU"] = _create_error_result("GPU", "No result received")
    try:
        out["CPU"] = cpu_result_q.get_nowait()
    except queue.Empty:
        out["CPU"] = _create_error_result("CPU", "No result received")
    return out


def main():
    """Main Streamlit app"""

    # Header
    st.markdown(
        f'<div class="main-header"> {DefaultValues.BLUEPRINT_NAME} - Backtesting Rebalance Strategies</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "**Interactive dynamic rebalancing strategies with real-time GPU vs CPU comparison**"
    )

    if not IMPORTS_OK:
        st.error(f"❌ Import Error: {IMPORT_ERROR}")
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

        st.text_input("Regime Name", value=DefaultValues.REGIME_NAME)
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
        kde_device = st.selectbox("KDE Device", ["GPU", "CPU"], index=1)

        # Rebalancing Strategy
        st.subheader("🧭 Rebalancing Strategy")
        strategy_display = {
            "pct_change": "Percentage Change",
            "drift_from_optimal": "Drift from Optimal",
            "max_drawdown": "Maximum Drawdown",
            "no_re_optimize": "No Re-Optimization",
        }
        strategy_key = st.selectbox(
            "Strategy",
            list(strategy_display.keys()),
            index=0,
            format_func=lambda k: strategy_display[k],
        )

        # Strategy information expandable section
        with st.expander(f"ℹ️ About {strategy_display[strategy_key]}", expanded=False):
            if strategy_key == "pct_change":
                st.markdown(
                    """
                **Percentage Change**

                Rebalances the portfolio when the period return falls below a specified threshold.

                **How it works:**
                - Monitors portfolio performance each evaluation period
                - Triggers rebalancing when returns drop below the threshold (e.g., -5%)
                - Helps protect against significant losses by adapting to poor performance

                **Best for:**
                - Risk management during market downturns
                - Portfolios sensitive to drawdowns
                - Conservative rebalancing approach

                **Example:** With threshold = -0.05, rebalancing occurs when portfolio loses more than 5% in a period.
                """
                )
            elif strategy_key == "drift_from_optimal":
                st.markdown(
                    """
                **Drift from Optimal**

                Rebalances when current portfolio weights deviate too much from the optimal allocation.

                **How it works:**
                - Calculates the distance between current and optimal weights
                - Uses L1 (Manhattan) or L2 (Euclidean) norm to measure deviation
                - Triggers rebalancing when drift exceeds the threshold

                **Best for:**
                - Maintaining target allocation discipline
                - Portfolios with clear strategic targets
                - Regular rebalancing to control drift

                **Example:** With threshold = 0.002 and L2 norm, rebalancing occurs when weight deviation exceeds 0.2%.
                """
                )
            elif strategy_key == "max_drawdown":
                st.markdown(
                    """
                **Maximum Drawdown**

                Rebalances when portfolio experiences a significant peak-to-trough decline.

                **How it works:**
                - Tracks the maximum portfolio value achieved
                - Calculates drawdown as decline from peak value
                - Triggers rebalancing when drawdown exceeds threshold

                **Best for:**
                - Downside risk management
                - Protecting against large losses
                - Crisis-responsive rebalancing

                **Example:** With threshold = 0.2, rebalancing occurs when portfolio drops 20% from its peak.
                """
                )
            else:  # no_re_optimize
                st.markdown(
                    """
                **No Re-Optimization**

                Maintains initial portfolio allocation without any rebalancing.

                **How it works:**
                - Sets portfolio allocation once at the beginning
                - Never triggers rebalancing regardless of performance
                - Pure buy-and-hold strategy

                **Best for:**
                - Baseline comparison against active strategies
                - Low-maintenance portfolios
                - Testing the value of rebalancing

                **Note:** This serves as a benchmark to compare against dynamic rebalancing strategies.
                """
                )

        threshold = st.number_input(
            "Threshold",
            value=(
                -0.005
                if strategy_key == "pct_change"
                else (0.002 if strategy_key == "drift_from_optimal" else 0.2)
            ),
            step=0.001,
            format="%.6f",
        )
        norm_choice = None
        if strategy_key == "drift_from_optimal":
            norm_choice = st.selectbox("Drift Norm", [1, 2], index=1)

        # Windows & Costs
        st.subheader("⏱️ Windows & Costs")
        col1, col2 = st.columns(2)
        with col1:
            look_back_window = st.number_input(
                "Look-back Window (days)",
                value=252,
                min_value=20,
                max_value=1250,
                step=10,
            )
        with col2:
            look_forward_window = st.number_input(
                "Look-forward Window (days)",
                value=21,
                min_value=5,
                max_value=125,
                step=1,
            )
        transaction_cost_factor = st.number_input(
            "Transaction Cost (fraction)",
            value=0.000,
            min_value=0.0,
            max_value=0.05,
            step=0.001,
            format="%.3f",
        )

        # Optional Constraints
        st.subheader(UIText.CONSTRAINTS_HEADER)

        # Add educational info about constraints
        with st.expander("ℹ️ Optional Constraints", expanded=False):
            st.markdown(
                """
            **🔄 Turnover:** Limits portfolio weight changes between periods (controls transaction costs)

            **🎯 Cardinality:** Limits number of assets with non-zero weights (creates focused portfolios, uses MILP)

            **⚠️ Hard CVaR Limit:** Sets absolute upper bound on portfolio risk (stricter than risk aversion)

            💡 **Note:** Constraints may increase solve times, especially cardinality.
            """
            )

        # Turnover constraint
        enable_turnover = st.checkbox(
            "Turnover Constraint",
            value=DefaultValues.ENABLE_TURNOVER_CONSTRAINT,
            help=UIText.TURNOVER_CONSTRAINT_HELP,
        )
        turnover_limit = None
        if enable_turnover:
            turnover_limit = st.number_input(
                "Turnover Limit (T_tar)",
                value=DefaultValues.TURNOVER_LIMIT,
                min_value=InputLimits.TURNOVER_LIMIT_RANGE[0],
                max_value=InputLimits.TURNOVER_LIMIT_RANGE[1],
                step=InputLimits.TURNOVER_LIMIT_STEP,
                format="%.3f",
            )

        # Cardinality constraint
        enable_cardinality = st.checkbox(
            "Cardinality Constraint",
            value=DefaultValues.ENABLE_CARDINALITY_CONSTRAINT,
            help=UIText.CARDINALITY_CONSTRAINT_HELP,
        )
        cardinality_limit = None
        if enable_cardinality:
            num_assets_in_dataset = get_dataset_num_assets(dataset_name)
            cardinality_limit = st.number_input(
                "Max Assets",
                value=min(DefaultValues.CARDINALITY_LIMIT, num_assets_in_dataset),
                min_value=InputLimits.CARDINALITY_LIMIT_RANGE[0],
                max_value=num_assets_in_dataset,
                step=InputLimits.CARDINALITY_LIMIT_STEP,
            )

        # Hard CVaR limit
        enable_cvar_limit = st.checkbox(
            "Hard CVaR Limit",
            value=DefaultValues.ENABLE_CVAR_LIMIT,
            help=UIText.CVAR_LIMIT_HELP,
        )
        cvar_hard_limit = None
        if enable_cvar_limit:
            cvar_hard_limit = st.number_input(
                "CVaR Limit",
                value=DefaultValues.CVAR_HARD_LIMIT,
                min_value=InputLimits.CVAR_HARD_LIMIT_RANGE[0],
                max_value=InputLimits.CVAR_HARD_LIMIT_RANGE[1],
                step=InputLimits.CVAR_HARD_LIMIT_STEP,
                format="%.4f",
            )



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

        # Run button
        st.markdown("---")
        run_btn = st.button(
            UIText.RUN_BUTTON.replace("Generate Efficient Frontier", "Run Rebalancing"),
            type="primary",
            use_container_width=True,
        )

    # Main content
    if run_btn:
        # Validate
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
                device=kde_device
            ),
            verbose=False
        )

        # Prepare parameters
        trading_range = (start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        cvar_params = CvarParameters(
            w_min=w_min,
            w_max=w_max,
            c_min=c_min,
            c_max=c_max,
            L_tar=L_tar,
            T_tar=turnover_limit,  # Use optional turnover constraint
            cvar_limit=cvar_hard_limit,  # Use optional hard CVaR limit
            risk_aversion=DefaultValues.RISK_AVERSION,  # Use default risk aversion
            confidence=confidence,
            cardinality=cardinality_limit,  # Use optional cardinality constraint
        )

        criteria = {"type": strategy_key, "threshold": float(threshold)}
        if strategy_key == "drift_from_optimal":
            criteria["norm"] = int(norm_choice or 2)

        st.markdown(
            '<div class="section-header"> GPU vs CPU Rebalancing </div>',
            unsafe_allow_html=True,
        )
        col_gpu, col_cpu = st.columns([1, 1], gap="medium")
        with col_gpu:
            st.markdown("### 🚀 GPU (cuOpt) Results")
            gpu_plot_container = st.empty()
            gpu_progress_placeholder = st.empty()
        with col_cpu:
            # In blog mode, hide CPU solver name
            cpu_header = (
                "### 🖥️ CPU Results"
                if blog_mode
                else f"### 🖥️ CPU ({cpu_solver_choice}) Results"
            )
            st.markdown(cpu_header)
            cpu_plot_container = st.empty()
            cpu_progress_placeholder = st.empty()

        # Run
        results = run_progressive_rebalancing(
            dataset_path=str(dataset_path),
            trading_range=trading_range,
            returns_compute_settings=returns_compute_settings,
            scenario_generation_settings=scenario_generation_settings,
            cvar_params=cvar_params,
            look_back_window=int(look_back_window),
            look_forward_window=int(look_forward_window),
            re_optimize_criteria=criteria,
            transaction_cost_factor=float(transaction_cost_factor),
            strategy_display_name=strategy_display[strategy_key],
            gpu_plot_container=gpu_plot_container,
            cpu_plot_container=cpu_plot_container,
            gpu_progress_placeholder=gpu_progress_placeholder,
            cpu_progress_placeholder=cpu_progress_placeholder,
            cpu_solver_choice=cpu_solver_choice,
            blog_mode=blog_mode,
        )

        # Summaries
        st.markdown(
            '<div class="section-header">📊 Summary</div>', unsafe_allow_html=True
        )
        g = results.get("GPU", {})
        c = results.get("CPU", {})

        # Display results in 3-column layout with speedup in the middle (equal spacing)
        col1, col2 = st.columns([1, 1])

        with col1:
            if g.get("success") and c.get("success"):
                st.metric("Number of Rebalances", g.get("rebal_count", 0))
                gt_solve = max(1e-9, g.get("total_solve_time", 0.0))
                ct_solve = max(1e-9, c.get("total_solve_time", 0.0))
                solve_speedup = ct_solve / gt_solve
                st.metric("⚡ GPU Solver Speedup", f"{solve_speedup:.1f}x faster")
            else:
                st.error("Speedup calculation failed")


        with col2:
            if g.get("success"):
                st.metric("⚡ GPU Solve Time", f"{g.get('total_solve_time', 0.0):.3f}s")
            else:
                st.error("GPU failed")
            if c.get("success"):
                st.metric("⚡ CPU Solve Time", f"{c.get('total_solve_time', 0.0):.3f}s")
            else:
                st.error("CPU failed")

        # Detailed tables
        with st.expander("📋 Detailed Period Results", expanded=False):
            if g.get("success") and isinstance(g.get("results_df"), pd.DataFrame):
                st.markdown("**GPU Period Results**")
                st.dataframe(
                    g["results_df"].reset_index().rename(columns={"index": "date"}),
                    hide_index=True,
                )
            if c.get("success") and isinstance(c.get("results_df"), pd.DataFrame):
                # In blog mode, hide CPU solver name
                cpu_results_header = (
                    "**CPU Period Results**"
                    if blog_mode
                    else f"**CPU ({cpu_solver_choice}) Period Results**"
                )
                st.markdown(cpu_results_header)
                st.dataframe(
                    c["results_df"].reset_index().rename(columns={"index": "date"}),
                    hide_index=True,
                )

    else:
        # Show instructions
        st.info(
            UIText.CONFIGURE_INSTRUCTION.replace(
                "Generate Efficient Frontier", "Run Rebalancing"
            )
        )

        st.markdown(
            """
        ### **Features:**
        - **Progressive Racing**: Both GPU and CPU solvers start simultaneously for fair comparison
        - **Multiple CPU Solvers**: Choose from HIGHS, CLARABEL, ECOS, OSQP, or SCS
        - **Real-Time Visualization**: See portfolio performance and rebalancing events as they happen
        - **Multiple Strategies**: Percentage change, drift from optimal, maximum drawdown triggers
        - **Period-by-Period Updates**: Track when re-optimization is triggered and portfolio updates
        - **Transaction Costs**: Model realistic trading costs in the backtest
        - **Live Metrics**: Real-time solve times, progress tracking, and performance comparison

        ### **How it works:**
        1. **Setup**: Configure your dataset, date range, CVaR parameters, and rebalancing strategy
        2. **Strategy Selection**: Choose trigger type (pct change, drift, drawdown) and threshold
        3. **Windows**: Set look-back (optimization) and look-forward (backtest) periods
        4. **Synchronized Start**: Both solvers initialize, then begin racing simultaneously
        5. **Progressive Updates**: Watch period-by-period as strategies evolve and trigger rebalancing
        6. **Performance Tracking**: Compare GPU vs CPU speed and final portfolio performance
        """
        )

        # Show example strategies
        with st.expander("📋 Example Strategy Configurations", expanded=False):
            st.markdown(
                """
            **Conservative (Low Rebalancing):**
            - Strategy: Percentage Change
            - Threshold: -0.01 (rebalance after 1% loss)
            - Look-back: 252 days (1 year of data)
            - Look-forward: 21 days (monthly evaluation)

            **Moderate (Medium Rebalancing):**
            - Strategy: Drift from Optimal
            - Threshold: 0.002 (rebalance when portfolio drifts)
            - Norm: L2 (Euclidean distance)
            - Look-back: 126 days (6 months)
            - Look-forward: 14 days (bi-weekly)

            **Aggressive (High Rebalancing):**
            - Strategy: Maximum Drawdown
            - Threshold: 0.05 (rebalance after 5% drawdown)
            - Look-back: 63 days (3 months)
            - Look-forward: 7 days (weekly evaluation)
            """
            )

        # Advanced settings info
        with st.expander("⚙️ Parameter Guidelines", expanded=False):
            st.markdown(
                """
            **Look-back Window:** Historical data for portfolio optimization
            - Longer periods (252+ days): More stable, slower adaptation
            - Shorter periods (63-126 days): More responsive, potentially noisier

            **Look-forward Window:** Period between strategy evaluations
            - Daily (1-7 days): High frequency, more trading costs
            - Weekly/Monthly (7-30 days): Lower frequency, practical for real trading

            **Thresholds:**
            - Percentage Change: Negative values (-0.001 to -0.05)
            - Drift from Optimal: Small positive values (0.001 to 0.01)
            - Maximum Drawdown: Positive values (0.01 to 0.2)
            """
            )

    # Add disclaimer at the bottom
    st.markdown("---")
    st.caption(
        "⚠️ This tool is for educational and research purposes. Past performance does not guarantee future results."
    )


if __name__ == "__main__":
    main()
