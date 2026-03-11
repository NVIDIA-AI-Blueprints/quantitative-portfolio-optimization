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
        metric_column = r.re_optimize_type
        results_df = pd.DataFrame(
            columns=[metric_column, "re_optimized", "portfolio_value", "max_drawdown"]
        )
        current_portfolio = r.initial_portfolio
        prev_portfolio = None
        portfolio_value = 1.0
        backtest_idx = 0
        backtest_date = r.trading_start
        backtest_final_date = pd.Timestamp(r.dates_range[-look_forward_window])
        period_counter = 0
        total_solve_time = 0.0
        total_kde_time = 0.0

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

                kde_t0 = time.time()
                opt_returns = cvar_utils.generate_cvar_data(
                    opt_returns,
                    scenario_generation_settings
                )
                kde_elapsed = time.time() - kde_t0
                total_kde_time += kde_elapsed

                progress_queue.put(
                    {
                        "solver": solver_name,
                        "status": "kde_timing",
                        "kde_time": kde_elapsed,
                        "total_kde_time": total_kde_time,
                        "message": f"{solver_name}: KDE fit+sample {kde_elapsed:.3f}s",
                    }
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

        # Backtest the final segment (remaining days after the last full window)
        remaining_days = len(r.dates_range) - backtest_idx
        if remaining_days > 1:
            try:
                bt_start = backtest_date.strftime("%Y-%m-%d")
                bt_end = r.dates_range[-1].strftime("%Y-%m-%d")
                bt_regime = {"name": "backtest_final", "range": (bt_start, bt_end)}
                test_returns = utils.calculate_returns(
                    dataset_path, bt_regime, returns_compute_settings
                )
                bt = backtest.portfolio_backtester(
                    current_portfolio, test_returns, benchmark_portfolios=None
                )
                bt_result = bt.backtest_single_portfolio(current_portfolio)
                cum_ret_values = (
                    bt_result["cumulative returns"].values[0]
                    if hasattr(bt_result["cumulative returns"], "values")
                    else bt_result["cumulative returns"]
                )
                cur_cum = cum_ret_values * portfolio_value
                cumulative_values = np.concatenate((cumulative_values, cur_cum))
                cumulative_dates.extend(bt._dates)

                # Redraw the plot with the final segment included
                with _matplotlib_lock:
                    cum_series_final = pd.Series(
                        cumulative_values,
                        index=pd.to_datetime(cumulative_dates),
                    )
                    ax.plot(
                        cum_series_final.index,
                        cum_series_final.values,
                        linewidth=3,
                        color=colors["frontier"],
                        alpha=0.9,
                        zorder=3,
                    )
                    _set_axis_limits(ax, cum_series_final, bh_series)
            except Exception:
                pass

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
                "total_kde_time": total_kde_time,
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
    returns_compute_settings,
    gpu_scenario_settings,
    cpu_scenario_settings,
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
            gpu_scenario_settings,
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
            cpu_scenario_settings,
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
                    "kde_timing",
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
                    "kde_timing",
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
        f'<div class="main-header">cuFOLIO - Backtesting Rebalance Strategies</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="text-align:center; font-size:1.1rem; color:inherit; opacity:0.85;">'
        "GPU-accelerated portfolio rebalancing with side-by-side solver comparison"
        "</p>",
        unsafe_allow_html=True,
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
        st.subheader("📊 Dataset")
        datasets = get_available_datasets()
        default_index = (
            datasets.index(DefaultValues.DATASET_NAME)
            if DefaultValues.DATASET_NAME in datasets
            else 0
        )
        _dataset_labels = {ds: f"Dataset {i+1}" for i, ds in enumerate(datasets)}
        dataset_name = st.selectbox(
            "Dataset",
            datasets,
            index=default_index,
            format_func=lambda x: _dataset_labels[x],
        )

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=DefaultValues.START_DATE)
        with col2:
            end_date = st.date_input("End Date", value=DefaultValues.END_DATE)

        # Portfolio Constraints (user-friendly names with technical help)
        st.subheader("💼 Portfolio Allocation")
        w_min, w_max = st.slider(
            "Allocation Range per Asset",
            -0.5, 1.0,
            (float(DefaultValues.W_MIN), max(0.0, float(DefaultValues.W_MAX))),
            0.05,
            help="Weight bounds (w_min, w_max) — min and max fraction of wealth allocated to any single asset. Negative values allow short selling.",
        )
        c_min, c_max = st.slider(
            "Cash Reserve Range",
            0.0, 1.0,
            (float(DefaultValues.C_MIN), float(DefaultValues.C_MAX)),
            0.05,
            help="Cash bounds (c_min, c_max) — min and max fraction of the portfolio held in cash.",
        )
        L_tar = st.slider(
            "Max Leverage",
            1.0, 3.0, float(DefaultValues.L_TAR), 0.1,
            help="Leverage target (L_tar) — upper bound on the sum of absolute position sizes. 1.0 = fully funded, >1 = leveraged.",
        )

        use_cardinality = st.checkbox(
            "Limit Number of Holdings",
            help="Cardinality constraint — restricts the portfolio to a maximum number of assets with non-zero weights.",
        )
        cardinality = st.slider("Max Holdings", 5, 50, 15, 1) if use_cardinality else None

        # Risk Parameters
        st.subheader("⚖️ Risk Settings")
        risk_aversion = st.slider(
            "Risk Sensitivity",
            0.1, 5.0, float(DefaultValues.RISK_AVERSION), 0.1,
            help="Risk aversion (λ) — controls the trade-off between return and risk. Higher values produce more conservative portfolios.",
        )
        confidence = st.slider(
            "Tail-Risk Confidence",
            0.90, 0.99, float(DefaultValues.CONFIDENCE), 0.01,
            help="CVaR confidence level (α) — probability level for measuring tail risk. 0.95 means the worst 5%% of scenarios are considered.",
        )
        num_scen = st.selectbox(
            "Simulation Count",
            [1000, 5000, 10000, 20000],
            index=2,
            help="Number of scenarios (num_scen) — how many return scenarios to simulate for risk estimation.",
        )

        # Rebalancing Strategy (simplified)
        st.subheader("🧭 Rebalancing Trigger")
        strategy_display = {
            "pct_change": "Loss Threshold",
            "drift_from_optimal": "Drift from Target",
            "max_drawdown": "Peak-to-Trough Decline",
            "no_re_optimize": "Buy & Hold (no rebalancing)",
        }
        strategy_key = st.selectbox(
            "When to Rebalance",
            list(strategy_display.keys()),
            index=0,
            format_func=lambda k: strategy_display[k],
            help="Determines the condition that triggers portfolio re-optimization.",
        )

        with st.expander(f"ℹ️ About: {strategy_display[strategy_key]}", expanded=False):
            if strategy_key == "pct_change":
                st.markdown(
                    "Re-optimizes when the portfolio return in a period drops "
                    "below a loss threshold (e.g. -0.5%). Useful for protecting "
                    "against sustained poor performance."
                )
            elif strategy_key == "drift_from_optimal":
                st.markdown(
                    "Re-optimizes when the current weights drift too far from "
                    "the target allocation. Keeps the portfolio disciplined."
                )
            elif strategy_key == "max_drawdown":
                st.markdown(
                    "Re-optimizes after the portfolio suffers a peak-to-trough "
                    "decline exceeding the threshold. Responds to crisis events."
                )
            else:
                st.markdown(
                    "No rebalancing — the initial allocation is held throughout "
                    "the entire period. Serves as a buy-and-hold baseline."
                )

        # ── Advanced Mode ──────────────────────────────────────────────
        st.markdown("---")
        advanced_mode = st.checkbox("⚙️ Advanced Mode", value=False)

        if advanced_mode:
            st.subheader("🔬 Advanced Settings")

            return_type = st.selectbox(
                "Return Calculation",
                ["LOG", "SIMPLE"],
                index=0 if DefaultValues.RETURN_TYPE == "LOG" else 1,
                help="Return type — LOG (logarithmic) or SIMPLE (arithmetic) returns.",
            )

            st.markdown("**Trigger Threshold**")
            threshold = st.number_input(
                "Threshold",
                value=(
                    -0.005
                    if strategy_key == "pct_change"
                    else (0.002 if strategy_key == "drift_from_optimal" else 0.2)
                ),
                step=0.001,
                format="%.6f",
                help="Rebalancing trigger threshold — the value that must be breached to trigger re-optimization.",
            )
            norm_choice = None
            if strategy_key == "drift_from_optimal":
                norm_choice = st.selectbox(
                    "Drift Norm",
                    [1, 2],
                    index=1,
                    help="Distance metric — L1 (Manhattan) or L2 (Euclidean) norm.",
                )

            st.markdown("**Evaluation Windows**")
            col1, col2 = st.columns(2)
            with col1:
                look_back_window = st.number_input(
                    "History Window (days)",
                    value=252,
                    min_value=20,
                    max_value=1250,
                    step=10,
                    help="Look-back window — number of historical trading days used to fit the optimization model.",
                )
            with col2:
                look_forward_window = st.number_input(
                    "Evaluation Period (days)",
                    value=21,
                    min_value=5,
                    max_value=125,
                    step=1,
                    help="Look-forward window — number of trading days per evaluation period before checking the trigger.",
                )
            transaction_cost_factor = st.number_input(
                "Trading Cost per Rebalance",
                value=0.000,
                min_value=0.0,
                max_value=0.05,
                step=0.001,
                format="%.3f",
                help="Transaction cost factor — fraction of turnover deducted as trading cost each time the portfolio is rebalanced.",
            )

            st.markdown("**Additional Constraints**")
            enable_turnover = st.checkbox(
                "Limit Turnover",
                value=DefaultValues.ENABLE_TURNOVER_CONSTRAINT,
                help="Turnover constraint (T_tar) — limits the total weight change between rebalances to control trading costs.",
            )
            turnover_limit = None
            if enable_turnover:
                turnover_limit = st.number_input(
                    "Max Turnover",
                    value=DefaultValues.TURNOVER_LIMIT,
                    min_value=InputLimits.TURNOVER_LIMIT_RANGE[0],
                    max_value=InputLimits.TURNOVER_LIMIT_RANGE[1],
                    step=InputLimits.TURNOVER_LIMIT_STEP,
                    format="%.3f",
                    help="Turnover limit (T_tar) — maximum L1 distance between old and new weights.",
                )

            enable_cvar_limit = st.checkbox(
                "Hard Risk Cap",
                value=DefaultValues.ENABLE_CVAR_LIMIT,
                help="Hard CVaR limit — sets an absolute upper bound on portfolio tail risk.",
            )
            cvar_hard_limit = None
            if enable_cvar_limit:
                cvar_hard_limit = st.number_input(
                    "Max Tail Risk (CVaR)",
                    value=DefaultValues.CVAR_HARD_LIMIT,
                    min_value=InputLimits.CVAR_HARD_LIMIT_RANGE[0],
                    max_value=InputLimits.CVAR_HARD_LIMIT_RANGE[1],
                    step=InputLimits.CVAR_HARD_LIMIT_STEP,
                    format="%.4f",
                    help="CVaR hard limit — portfolio CVaR must stay below this value.",
                )

        else:
            # Defaults for non-advanced mode
            return_type = DefaultValues.RETURN_TYPE
            threshold = (
                -0.005
                if strategy_key == "pct_change"
                else (0.002 if strategy_key == "drift_from_optimal" else 0.2)
            )
            norm_choice = 2 if strategy_key == "drift_from_optimal" else None
            look_back_window = 252
            look_forward_window = 21
            transaction_cost_factor = 0.0
            turnover_limit = None
            enable_turnover = False
            cvar_hard_limit = None
            enable_cvar_limit = False

        # CPU solver selection (always visible, masked names)
        st.markdown("---")
        _cpu_solvers = {"HIGHS": "CPU Solver 1", "CLARABEL": "CPU Solver 2"}
        cpu_solver_choice = st.selectbox(
            "CPU Solver",
            list(_cpu_solvers.keys()),
            index=0,
            format_func=lambda x: _cpu_solvers[x],
            help="Choose which CPU optimization engine to compare against GPU.",
        )

        blog_mode = True

        # Run button
        st.markdown("---")
        run_btn = st.button(
            "🚀 Run Rebalancing",
            type="primary",
            width="stretch",
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
        gpu_scenario_settings = ScenarioGenerationSettings(
            num_scen=num_scen,
            fit_type='kde',
            kde_settings=KDESettings(
                bandwidth=0.01, kernel='gaussian', device='GPU'
            ),
            verbose=False,
        )
        cpu_scenario_settings = ScenarioGenerationSettings(
            num_scen=num_scen,
            fit_type='kde',
            kde_settings=KDESettings(
                bandwidth=0.01, kernel='gaussian', device='CPU'
            ),
            verbose=False,
        )

        # Prepare parameters
        trading_range = (start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        cvar_params = CvarParameters(
            w_min=w_min,
            w_max=w_max,
            c_min=c_min,
            c_max=c_max,
            L_tar=L_tar,
            T_tar=turnover_limit,
            cvar_limit=cvar_hard_limit,
            risk_aversion=risk_aversion,
            confidence=confidence,
            cardinality=cardinality,
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
            gpu_scenario_settings=gpu_scenario_settings,
            cpu_scenario_settings=cpu_scenario_settings,
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
                st.metric("🔬 GPU KDE Time", f"{g.get('total_kde_time', 0.0):.3f}s")
            else:
                st.error("GPU failed")
            if c.get("success"):
                st.metric("⚡ CPU Solve Time", f"{c.get('total_solve_time', 0.0):.3f}s")
                st.metric("🔬 CPU KDE Time", f"{c.get('total_kde_time', 0.0):.3f}s")
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
        # Landing page
        cover_path = script_dir / "diagrams" / "fsi-visual-portfolio-optimization-blueprint-4539200-r2.png"
        arch_path = script_dir / "diagrams" / "arch_diagram.svg"
        gif_path = script_dir / "diagrams" / "rebalancing_gpu_vs_cpu.gif"

        if cover_path.exists():
            st.image(str(cover_path), width="stretch")

        st.markdown(
            "Simulate **rebalancing strategies** that re-optimize your portfolio "
            "when market conditions change — then watch GPU and CPU solvers race "
            "through the backtest in real time."
        )

        tab_overview, tab_arch, tab_bench = st.tabs(
            ["📊 Overview", "🏗️ Architecture", "📈 Benchmarks"]
        )

        with tab_overview:
            # GIF demo
            if gif_path.exists():
                st.markdown("#### GPU vs CPU — Live Demo")
                import base64
                gif_bytes = gif_path.read_bytes()
                gif_b64 = base64.b64encode(gif_bytes).decode()
                st.markdown(
                    f'<img src="data:image/gif;base64,{gif_b64}" style="width:100%;">',
                    unsafe_allow_html=True,
                )

            # Dataset summary and price chart
            st.markdown("#### Selected Dataset")
            dataset_path = workspace_root / "data" / "stock_data" / f"{dataset_name}.csv"
            if dataset_path.exists():
                try:
                    df = pd.read_csv(dataset_path, index_col=0, parse_dates=True)
                    tickers = list(df.columns)

                    mask = (df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))
                    df_filtered = df.loc[mask]
                    if df_filtered.empty:
                        df_filtered = df

                    col_s1, col_s2, col_s3 = st.columns(3)
                    with col_s1:
                        st.metric("Assets", len(tickers))
                    with col_s2:
                        st.metric("From", f"{df_filtered.index.min().strftime('%Y-%m-%d')}")
                    with col_s3:
                        st.metric("To", f"{df_filtered.index.max().strftime('%Y-%m-%d')}")

                    fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
                    normalised = df_filtered.div(df_filtered.iloc[0])
                    for col in normalised.columns:
                        ax.plot(normalised.index, normalised[col], linewidth=0.8, alpha=0.7)
                    ax.set_title(
                        f"{_dataset_labels.get(dataset_name, 'Dataset')} — Normalised Closing Prices",
                        fontsize=14, fontweight="bold",
                    )
                    ax.set_ylabel("Price (normalised to 1)")
                    ax.set_xlabel("")
                    ax.grid(True, alpha=0.25)
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as e:
                    st.warning(f"Could not load dataset preview: {e}")
            else:
                st.info(f"**{_dataset_labels.get(dataset_name, 'Dataset')}** not found on disk.")

        with tab_arch:
            if arch_path.exists():
                st.image(str(arch_path), width="stretch")
            else:
                st.info("Architecture diagram not found.")
            st.markdown(
                "Market data flows through **returns forecasting** and "
                "**scenario generation** into the **CVaR optimizer**, which "
                "produces an optimal allocation. The strategy is then "
                "**backtested** period-by-period, triggering re-optimization "
                "when conditions are breached."
            )

        with tab_bench:
            st.markdown("#### Benchmark Results")
            st.markdown(
                "The table below shows representative solver performance on "
                "standard datasets. GPU acceleration delivers significant "
                "speed-ups as problem size grows."
            )
            bench_data = {
                "Dataset": ["100 assets", "100 assets", "500 assets", "500 assets"],
                "Solver": ["GPU", "CPU", "GPU", "CPU"],
                "Scenarios": ["10 000", "10 000", "10 000", "10 000"],
                "Avg Solve (s)": ["0.03", "0.12", "0.08", "1.45"],
                "Speedup": ["—", "—", "—", "—"],
            }
            bench_df = pd.DataFrame(bench_data)
            bench_df.loc[0, "Speedup"] = f"{0.12/0.03:.1f}x"
            bench_df.loc[1, "Speedup"] = "baseline"
            bench_df.loc[2, "Speedup"] = f"{1.45/0.08:.1f}x"
            bench_df.loc[3, "Speedup"] = "baseline"
            st.dataframe(bench_df, hide_index=True, width="stretch")
            st.caption(
                "Timings are illustrative and depend on hardware. "
                "Run your own comparison using the app."
            )

        st.info(
            "👈 **Configure parameters in the sidebar and click "
            "'Run Rebalancing' to start.**"
        )

    # Add disclaimer at the bottom
    st.markdown("---")
    st.caption(
        "⚠️ This tool is for educational and research purposes. Past performance does not guarantee future results."
    )


if __name__ == "__main__":
    main()
