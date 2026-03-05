#!/usr/bin/env python3
"""
CVaR Portfolio Optimization - Interactive Streamlit App

Based on the CVaR basic notebook, this app provides an interactive interface
for CVaR portfolio optimization with real-time visualization.

Author: Peihan Huo (phuo@nvidia.com)
"""

import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Add workspace root to path for imports (package structure: cufolio = "src")
script_dir = Path(__file__).parent.absolute()
workspace_root = script_dir.parent  # This is the workspace root
sys.path.insert(0, str(workspace_root))
cvar_dir = workspace_root  # For backward compatibility with path references

try:
    from cufolio import (
        backtest,
        cvar_optimizer,
        cvar_utils,
        mean_variance_optimizer,
        utils,
    )
    from cufolio.cvar_parameters import CvarParameters
    from cufolio.mean_variance_parameters import MeanVarianceParameters
    from cufolio.settings import (
        KDESettings,
        ReturnsComputeSettings,
        ScenarioGenerationSettings,
    )
except ImportError as e:
    st.error(f"❌ Import Error: {e}")
    st.error("Please ensure cufolio package is installed (pip install -e .)")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="cuFOLIO Portfolio Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .stMetric {
        background-color: #f0f2f6;
        border: 1px solid #e1e5e9;
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin: 0.2rem 0;
    }
    .portfolio-section {
        border: 2px solid #1f77b4;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f8f9fa;
    }
    .results-section {
        border: 2px solid #28a745;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f8f9fa;
    }
</style>
""",
    unsafe_allow_html=True,
)


def get_available_datasets():
    """Get list of available datasets"""
    script_dir = Path(__file__).parent
    # demo -> workspace root -> data/stock_data
    data_dir = script_dir.parent / "data" / "stock_data"

    # Try relative path first
    if data_dir.exists():
        csv_files = [
            f.stem for f in data_dir.glob("*.csv") if not f.name.startswith(".")
        ]
        return sorted(csv_files)

    # Fallback: try absolute path
    abs_data_dir = Path("/home/scratch.phuo_wwfo/github/cufolio/data/stock_data")
    if abs_data_dir.exists():
        csv_files = [
            f.stem for f in abs_data_dir.glob("*.csv") if not f.name.startswith(".")
        ]
        return sorted(csv_files)

    return [
        "sp500",
        "sp100",
        "baby_dataset",
        "global_titans",
        "dow30",
        "test_2021",
    ]  # fallback defaults


def get_dataset_tickers(dataset_name):
    """Get available tickers from a dataset"""
    script_dir = Path(__file__).parent
    # demo -> workspace root -> data/stock_data
    data_path = script_dir.parent / "data" / "stock_data" / f"{dataset_name}.csv"

    try:
        # Try relative path first
        if data_path.exists():
            df = pd.read_csv(data_path)
            # Get column names excluding 'Date' column
            tickers = [col for col in df.columns if col != "Date"]
            return sorted(tickers)

        # Fallback: try absolute path
        abs_data_path = Path(
            f"/home/scratch.phuo_wwfo/github/cufolio/data/stock_data/{dataset_name}.csv"
        )
        if abs_data_path.exists():
            df = pd.read_csv(abs_data_path)
            # Get column names excluding 'Date' column
            tickers = [col for col in df.columns if col != "Date"]
            return sorted(tickers)

        return ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]  # fallback defaults
    except Exception as e:
        print(f"Error loading tickers for {dataset_name}: {e}")
        return ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]  # fallback defaults


def create_portfolio_visualization(portfolio, title="Portfolio Allocation"):
    """Create portfolio visualization using Portfolio's built-in method"""
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 8))

    # Use Portfolio's built-in visualization method
    portfolio.plot_portfolio(
        ax=ax,
        title=title,
        figsize=(12, 8),
        style="modern",
        cutoff=1e-3,
        min_percentage=0.0,
        sort_by_weight=True,
        show_plot=False,
    )

    # Customize grid appearance to make it less obvious
    ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)  # Put grid behind the bars

    return fig


def run_solver_optimization(
    cvar_problem, solver_name, solver_settings, progress_placeholder
):
    """Run optimization with a specific solver and return results."""
    try:
        with progress_placeholder.container():
            st.info(f"⚡ Running {solver_name} optimization...")

            # Solve optimization
            start_time = time.time()
            results, portfolio = cvar_problem.solve_optimization_problem(
                solver_settings=solver_settings, print_results=False
            )
            solve_time = time.time() - start_time

            # Check if optimization was successful
            if results is None or portfolio is None:
                st.error(f"❌ {solver_name} optimization failed: No solution returned")
                return None, None, None

        # Check for required keys (CVaR uses "CVaR", MV uses "variance")
        required_keys = ["return", "obj"]
        missing_keys = [key for key in required_keys if key not in results]
        if missing_keys:
            st.error(f"❌ {solver_name} missing result keys: {missing_keys}")
            return None, None, None

            st.success(f"✅ {solver_name} completed in {solve_time:.2f}s")
            return results, portfolio, solve_time

    except Exception as e:
        with progress_placeholder.container():
            st.error(f"❌ {solver_name} optimization error: {str(e)}")
        return None, None, None


def run_solver_optimization_thread(cvar_problem, solver_name, solver_settings):
    """Thread-safe version of solver optimization (without Streamlit components)."""
    try:
        # Solve optimization
        start_time = time.time()
        results, portfolio = cvar_problem.solve_optimization_problem(
            solver_settings=solver_settings, print_results=False
        )
        solve_time = time.time() - start_time

        # Check if optimization was successful
        if results is None or portfolio is None:
            return {
                "solver_name": solver_name,
                "success": False,
                "error": "No solution returned",
                "results": None,
                "portfolio": None,
                "solve_time": solve_time,
            }

        # Check for required keys
        required_keys = ["return", "obj"]
        missing_keys = [key for key in required_keys if key not in results]
        if missing_keys:
            return {
                "solver_name": solver_name,
                "success": False,
                "error": f"Missing result keys: {missing_keys}",
                "results": None,
                "portfolio": None,
                "solve_time": solve_time,
            }

        return {
            "solver_name": solver_name,
            "success": True,
            "error": None,
            "results": results,
            "portfolio": portfolio,
            "solve_time": solve_time,
        }

    except Exception as e:
        return {
            "solver_name": solver_name,
            "success": False,
            "error": str(e),
            "results": None,
            "portfolio": None,
            "solve_time": 0,
        }


def run_parallel_optimizations(
    cvar_problem,
    gpu_settings,
    cpu_settings,
    cpu_solver_choice,
    gpu_progress_placeholder,
    cpu_progress_placeholder,
):
    """Run GPU and CPU optimizations in parallel."""

    # Start both progress indicators
    with gpu_progress_placeholder.container():
        st.info("🚀 Starting GPU (cuOpt) optimization...")
    with cpu_progress_placeholder.container():
        st.info(f"🖥️ Starting CPU ({cpu_solver_choice}) optimization...")

    # Create thread pool and submit both optimizations
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {}

        # Submit GPU optimization if available
        if gpu_settings:
            futures["GPU"] = executor.submit(
                run_solver_optimization_thread,
                cvar_problem,
                "GPU (cuOpt)",
                gpu_settings,
            )

        # Submit CPU optimization
        futures["CPU"] = executor.submit(
            run_solver_optimization_thread,
            cvar_problem,
            f"CPU ({cpu_solver_choice})",
            cpu_settings,
        )

        # Collect results as they complete
        results = {}
        for future in as_completed(futures.values()):
            result = future.result()
            solver_type = "GPU" if "GPU" in result["solver_name"] else "CPU"
            results[solver_type] = result

            # Update progress indicators as results come in
            if solver_type == "GPU":
                with gpu_progress_placeholder.container():
                    if result["success"]:
                        st.success(
                            f"✅ GPU (cuOpt) completed in {result['solve_time']:.2f}s"
                        )
                    else:
                        st.error(f"❌ GPU (cuOpt) failed: {result['error']}")
            else:
                with cpu_progress_placeholder.container():
                    if result["success"]:
                        st.success(
                            f"✅ CPU ({cpu_solver_choice}) completed in {result['solve_time']:.2f}s"
                        )
                    else:
                        st.error(
                            f"❌ CPU ({cpu_solver_choice}) failed: {result['error']}"
                        )

    return results


def display_solver_results(
    results,
    portfolio,
    cvar_problem,
    solve_time,
    solver_name,
    portfolio_placeholder,
    results_placeholder,
    enable_backtest=False,
    backtest_params=None,
    blog_mode=False,
):
    """Display results for a specific solver in its designated placeholders."""
    try:
        # Portfolio visualization
        with portfolio_placeholder.container():
            try:
                # Validate portfolio data first
                if not hasattr(portfolio, "print_clean"):
                    st.error("❌ Portfolio object is missing required methods")
                    return

                clean_positions, cash = portfolio.print_clean(cutoff=1e-3)
                if not clean_positions:
                    st.warning("⚠️ Portfolio has no significant positions")
                else:
                    st.info(
                        f"📊 Creating visualization for {len(clean_positions)} positions..."
                    )

                # Create custom portfolio plot
                # In blog mode, hide CPU solver name
                if "CPU" in solver_name and blog_mode:
                    title = "Portfolio Allocation"
                else:
                    title = f"{solver_name} Portfolio Allocation"
                fig = create_portfolio_visualization(portfolio, title)
                if fig is not None:
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.error("❌ Failed to create portfolio visualization")

            except Exception as e:
                st.error(
                    f"❌ Error creating {solver_name} portfolio visualization: {str(e)}"
                )

        # Results display
        with results_placeholder.container():
            try:
                formatted_results = format_optimization_results(
                    results, portfolio, cvar_problem, solve_time
                )

                # Display key metrics in columns
                metric_col1, metric_col2 = st.columns(2)

                with metric_col1:
                    st.metric("Expected Return", f"{results['return']*100:.4f}%")
                    if "CVaR" in results:
                        st.metric("CVaR (95%)", f"{results['CVaR']*100:.4f}%")
                    elif "variance" in results:
                        st.metric("Variance", f"{results['variance']:.6f}")

                with metric_col2:
                    st.metric("Objective Value", f"{results['obj']:.6f}")
                    st.metric("Solve Time", f"{solve_time:.4f}s")

                # Backtest Analysis (if enabled) or Detailed Metrics
                if enable_backtest and backtest_params is not None:
                    with st.expander("📈 Backtest Analysis", expanded=True):
                        try:
                            with st.spinner("Running backtest analysis..."):
                                backtest_result, backtest_ax = run_backtest_analysis(
                                    portfolio,
                                    backtest_params["dataset_name"],
                                    backtest_params["start_date"],
                                    backtest_params["end_date"],
                                    backtest_params["return_type"],
                                    backtest_params["method"],
                                )

                                if (
                                    backtest_result is not None
                                    and backtest_ax is not None
                                ):
                                    # Display backtest plot
                                    st.pyplot(backtest_ax.figure)

                                    # Display backtest metrics
                                    st.markdown("**📊 Backtest Performance Metrics**")
                                    display_columns = [
                                        "mean portfolio return",
                                        "sharpe",
                                        "sortino",
                                        "max drawdown",
                                    ]
                                    if all(
                                        col in backtest_result.columns
                                        for col in display_columns
                                    ):
                                        metrics_df = backtest_result[
                                            display_columns
                                        ].copy()

                                        # Format for better display
                                        for col in display_columns:
                                            if col in metrics_df.columns:
                                                metrics_df[col] = metrics_df[col].apply(
                                                    lambda x: f"{x:.4f}"
                                                )

                                        st.dataframe(
                                            metrics_df, width="stretch", hide_index=True
                                        )
                                    else:
                                        st.dataframe(
                                            backtest_result,
                                            width="stretch",
                                            hide_index=True,
                                        )
                                else:
                                    st.error("❌ Backtest analysis failed")

                        except Exception as e:
                            st.error(f"❌ Backtest error: {str(e)}")

                # Portfolio positions
                st.markdown("**💼 Complete Portfolio**")
                clean_positions, cash = portfolio.print_clean(cutoff=1e-3)

                # Create positions dataframe
                position_data = []
                for ticker, weight in clean_positions.items():
                    position_type = "🟢 Long" if weight > 0 else "🔴 Short"
                    position_data.append(
                        {
                            "Asset": ticker,
                            "Weight": f"{weight:.4f}",
                            "Percentage": f"{weight*100:.2f}%",
                            "Position": position_type,
                        }
                    )

                # Add cash if significant
                if abs(cash) > 1e-3:
                    cash_type = "💰 Cash" if cash > 0 else "🏦 Borrowed"
                    position_data.append(
                        {
                            "Asset": "CASH",
                            "Weight": f"{cash:.4f}",
                            "Percentage": f"{cash*100:.2f}%",
                            "Position": cash_type,
                        }
                    )

                if position_data:
                    positions_df = pd.DataFrame(position_data)
                    # Sort by absolute weight value and show top 5
                    positions_df["abs_weight"] = (
                        positions_df["Weight"].astype(float).abs()
                    )
                    positions_df = positions_df.sort_values(
                        "abs_weight", ascending=False
                    ).drop("abs_weight", axis=1)
                    st.dataframe(positions_df, width="stretch", hide_index=True)
                else:
                    st.info("No significant positions found")

            except Exception as e:
                st.error(f"❌ Error displaying {solver_name} results: {str(e)}")

    except Exception as e:
        st.error(f"❌ Error in {solver_name} results display: {str(e)}")


def run_backtest_analysis(
    portfolio,
    dataset_name,
    backtest_start,
    backtest_end,
    return_type,
    backtest_method="historical",
):
    """Run backtest analysis on the optimized portfolio."""
    try:
        # Create backtest regime dictionary
        backtest_regime = {
            "name": "backtest",
            "range": (
                backtest_start.strftime("%Y-%m-%d"),
                backtest_end.strftime("%Y-%m-%d"),
            ),
        }

        # Get dataset path
        dataset_path = workspace_root / "data" / "stock_data" / f"{dataset_name}.csv"
        if not dataset_path.exists():
            # Fallback to absolute path
            dataset_path = Path(
                f"/home/scratch.phuo_wwfo/github/cufolio/data/stock_data/{dataset_name}.csv"
            )

        backtest_returns_settings = ReturnsComputeSettings(
            return_type=return_type, freq=1
        )
        backtest_returns_dict = utils.calculate_returns(
            dataset_path, backtest_regime, backtest_returns_settings
        )

        # Create backtester
        backtester = backtest.portfolio_backtester(
            test_portfolio=portfolio,
            returns_dict=backtest_returns_dict,
            risk_free_rate=0.0,
            test_method=backtest_method,
            benchmark_portfolios=None,  # Uses equal-weight benchmark by default
        )

        # Run backtest
        backtest_result, ax = backtester.backtest_against_benchmarks(
            plot_returns=True,
            title=f"Backtest Results - {portfolio.name}",
            save_plot=False,
        )

        return backtest_result, ax

    except Exception as e:
        st.error(f"Backtest failed: {str(e)}")
        import traceback

        st.error(f"Details: {traceback.format_exc()}")
        return None, None


def format_optimization_results(results, portfolio, optimizer, solve_time):
    """Format optimization results for both CVaR and MV optimizers."""
    solver_name = results.get("solver", "Unknown")
    expected_return = results["return"]
    objective_value = results["obj"]

    config_data = {
        "Solver": solver_name,
        "Regime": optimizer.regime_name,
        "Time Period": f"{optimizer.regime_range[0]} to {optimizer.regime_range[1]}",
        "Assets": optimizer.n_assets,
    }

    if hasattr(optimizer.params, "confidence"):
        config_data["Confidence Level"] = f"{optimizer.params.confidence:.1%}"
    if hasattr(optimizer, "data") and hasattr(optimizer.data, "p"):
        config_data["Scenarios"] = f"{len(optimizer.data.p):,}"

    if optimizer.params.cardinality is not None:
        config_data["Cardinality Limit"] = f"{optimizer.params.cardinality} assets"
    if hasattr(optimizer.params, "cvar_limit") and optimizer.params.cvar_limit is not None:
        config_data["CVaR Hard Limit"] = f"{optimizer.params.cvar_limit:.4f}"
    if hasattr(optimizer.params, "var_limit") and optimizer.params.var_limit is not None:
        config_data["Variance Hard Limit"] = f"{optimizer.params.var_limit:.6f}"
    if optimizer.existing_portfolio is not None and optimizer.params.T_tar is not None:
        config_data["Turnover Constraint"] = f"{optimizer.params.T_tar:.3f}"

    performance_data = {
        "Expected Return": f"{expected_return:.6f} ({expected_return*100:.4f}%)",
    }
    if "CVaR" in results:
        cvar_value = results["CVaR"]
        performance_data["CVaR (95%)"] = f"{cvar_value:.6f} ({cvar_value*100:.4f}%)"
    if "variance" in results:
        var_value = results["variance"]
        performance_data["Variance"] = f"{var_value:.6f}"
        performance_data["Std Deviation"] = f"{var_value**0.5:.6f}"
    performance_data["Objective Value"] = f"{objective_value:.6f}"

    timing_data = {"Solve Time": f"{solve_time:.4f} seconds"}
    if hasattr(optimizer, "set_up_time"):
        timing_data["Setup Time"] = f"{optimizer.set_up_time:.4f} seconds"

    return {
        "config": config_data,
        "performance": performance_data,
        "timing": timing_data,
    }


# Main Streamlit App
def main():
    st.title("📈 Portfolio Optimizer")

    # Sidebar for parameters
    with st.sidebar:
        st.header("🔧 Optimization Parameters")

        # Optimization method selection
        opt_method = st.selectbox(
            "Optimization Method",
            ["Mean-CVaR", "Mean-Variance"],
            index=0,
            help="CVaR uses scenario-based risk; Mean-Variance uses covariance-based risk",
        )

        # Dataset selection
        st.subheader("📊 Dataset Configuration")
        available_datasets = get_available_datasets()
        dataset_name = st.selectbox(
            "Dataset",
            available_datasets,
            index=0 if "sp500" in available_datasets else 0,
        )

        return_type = st.selectbox("Return Type", ["LOG", "SIMPLE"], index=0)
        if opt_method == "Mean-CVaR":
            kde_device = st.selectbox("KDE Device", ["GPU", "CPU"], index=0)

        # Date range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=pd.to_datetime("2021-01-01"))
        with col2:
            end_date = st.date_input("End Date", value=pd.to_datetime("2024-01-01"))

        st.subheader("💼 Portfolio Constraints")

        # Asset weight bounds
        st.markdown("**Asset Weight Bounds**")

        # Weight constraint mode selection
        weight_mode = st.radio(
            "Weight Constraint Mode:",
            ["Default Weights", "Custom Asset Constraints"],
            index=0,
            help="Choose between simple default constraints or custom per-asset constraints",
        )

        w_min_dict = {}
        w_max_dict = {}

        if weight_mode == "Default Weights":
            # Simple default mode with sliders
            others_min = st.slider("Default Min Weight", -1.0, 1.0, -0.3, 0.05)
            others_max = st.slider("Default Max Weight", others_min, 2.0, 0.4, 0.05)
            w_min_dict = {"others": others_min}
            w_max_dict = {"others": others_max}

        else:  # Custom Asset Constraints

            # Get available tickers for the selected dataset
            available_tickers = get_dataset_tickers(dataset_name)

            # Initialize session state for custom constraints if not exists
            if "custom_constraints" not in st.session_state:
                st.session_state.custom_constraints = [
                    {"ticker": "NVDA", "min_weight": 0.1, "max_weight": 0.6},
                    {"ticker": "AAPL", "min_weight": 0.0, "max_weight": 0.2},
                ]

            # Display existing constraints
            st.markdown("**Individual Asset Constraints:**")
            constraints_to_remove = []

            for i, constraint in enumerate(st.session_state.custom_constraints):
                with st.container():
                    col1, col2, col3, col4 = st.columns([2, 1.5, 1.5, 0.5])

                    with col1:
                        # Ticker selection
                        ticker_idx = 0
                        if constraint["ticker"] in available_tickers:
                            ticker_idx = available_tickers.index(constraint["ticker"])
                        new_ticker = st.selectbox(
                            f"Asset {i+1}",
                            available_tickers,
                            index=ticker_idx,
                            key=f"ticker_{i}",
                        )
                        constraint["ticker"] = new_ticker

                    with col2:
                        # Min weight input
                        new_min = st.number_input(
                            "Min Weight",
                            min_value=-2.0,
                            max_value=2.0,
                            value=constraint["min_weight"],
                            step=0.05,
                            format="%.2f",
                            key=f"min_{i}",
                        )
                        constraint["min_weight"] = new_min

                    with col3:
                        # Max weight input
                        new_max = st.number_input(
                            "Max Weight",
                            min_value=new_min,
                            max_value=2.0,
                            value=max(constraint["max_weight"], new_min),
                            step=0.05,
                            format="%.2f",
                            key=f"max_{i}",
                        )
                        constraint["max_weight"] = new_max

                    with col4:
                        # Remove button
                        if st.button("🗑️", key=f"remove_{i}", help="Remove constraint"):
                            constraints_to_remove.append(i)

            # Remove constraints marked for deletion
            for i in reversed(constraints_to_remove):
                st.session_state.custom_constraints.pop(i)

            # Add new constraint button
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("➕"):
                    st.session_state.custom_constraints.append(
                        {
                            "ticker": available_tickers[0],
                            "min_weight": 0.0,
                            "max_weight": 0.3,
                        }
                    )
                    st.rerun()

            # Default constraint for other assets
            st.markdown("**Default for Other Assets:**")
            col1, col2 = st.columns(2)
            with col1:
                others_min = st.number_input("Others Min Weight", -1.0, 1.0, -0.3, 0.05)
            with col2:
                others_max = st.number_input(
                    "Others Max Weight", others_min, 2.0, 0.4, 0.05
                )

            # Build constraint dictionaries
            for constraint in st.session_state.custom_constraints:
                w_min_dict[constraint["ticker"]] = constraint["min_weight"]
                w_max_dict[constraint["ticker"]] = constraint["max_weight"]

            # Add others constraint
            w_min_dict["others"] = others_min
            w_max_dict["others"] = others_max

            # Display current constraint summary
            with st.expander("📋 Current Constraint Summary"):
                st.markdown("**w_min dictionary:**")
                st.code(str(w_min_dict))
                st.markdown("**w_max dictionary:**")
                st.code(str(w_max_dict))

        # Cash bounds
        c_min = st.slider("Cash Min", 0.0, 0.5, 0.0, 0.05)
        c_max = st.slider("Cash Max", c_min, 1.0, 0.2, 0.05)

        # Leverage and other constraints
        st.subheader("📐 Advanced Constraints")
        L_tar = st.slider("Leverage Target", 1.0, 3.0, 1.6, 0.1)
        use_cardinality = st.checkbox("Enable Cardinality Constraint")
        cardinality = st.slider("Max Assets", 5, 50, 15, 1) if use_cardinality else None

        # Risk parameters
        st.subheader("⚖️ Risk Parameters")
        risk_aversion = st.slider("Risk Aversion", 0.1, 5.0, 1.0, 0.1)

        if opt_method == "Mean-CVaR":
            confidence = st.slider("CVaR Confidence Level", 0.90, 0.99, 0.95, 0.01)
            num_scen = st.selectbox(
                "Number of Scenarios", [1000, 5000, 10000, 20000], index=2
            )
        else:
            enable_var_limit = st.checkbox("Enable Variance Hard Limit", value=False)
            var_limit = None
            if enable_var_limit:
                var_limit = st.number_input(
                    "Variance Limit",
                    value=0.01,
                    min_value=0.0001,
                    max_value=1.0,
                    step=0.001,
                    format="%.4f",
                )

        # Solver settings
        st.subheader("🔧 Solver Settings")

        # CPU solver selection
        cpu_solver_options = {
            "HIGHS": "HiGHS (Fast LP/QP solver)",
            "CLARABEL": "Clarabel (Interior point)",
            "ECOS": "ECOS (Embedded conic)",
            "OSQP": "OSQP (Quadratic programming)",
            "SCS": "SCS (Splitting conic solver)",
        }

        cpu_solver_choice = st.selectbox(
            "CPU Solver",
            list(cpu_solver_options.keys()),
            index=0,
            format_func=lambda x: cpu_solver_options[x],
            help="Select the CPU solver to compare against GPU cuOpt",
        )

        # Backtest settings
        st.subheader("📈 Backtest Settings")
        enable_backtest = st.checkbox("Enable Portfolio Backtesting", value=False)

        if enable_backtest:
            # Available date range from dataset
            try:
                available_datasets = get_available_datasets()
                if dataset_name in available_datasets:
                    sample_tickers = get_dataset_tickers(dataset_name)
                    if sample_tickers:
                        # Get date range from sample data
                        dataset_path = (
                            workspace_root
                            / "data"
                            / "stock_data"
                            / f"{dataset_name}.csv"
                        )
                        if not dataset_path.exists():
                            # Fallback to absolute path
                            dataset_path = Path(
                                f"/home/scratch.phuo_wwfo/github/cufolio/data/stock_data/{dataset_name}.csv"
                            )

                        sample_df = pd.read_csv(
                            dataset_path, index_col=0, parse_dates=True
                        )
                        min_date = sample_df.index.min().date()
                        max_date = sample_df.index.max().date()

                        st.info(f"📅 Available data: {min_date} to {max_date}")

                        col1, col2 = st.columns(2)
                        with col1:
                            backtest_start = st.date_input(
                                "Start Date",
                                value=min_date,
                                min_value=min_date,
                                max_value=max_date,
                            )
                        with col2:
                            backtest_end = st.date_input(
                                "End Date",
                                value=max_date,
                                min_value=min_date,
                                max_value=max_date,
                            )

                        if backtest_start >= backtest_end:
                            st.error("End date must be after start date")

                        backtest_method = st.selectbox(
                            "Backtest Method",
                            ["historical", "kde_simulation", "gaussian_simulation"],
                            index=0,
                            help="Method for generating return scenarios",
                        )

            except Exception as e:
                st.error(f"Error loading dataset info: {str(e)}")
                enable_backtest = False

        # Display Mode
        st.subheader("📝 Display Mode")
        blog_mode = st.checkbox(
            "Blog Mode",
            value=False,
            help="When enabled, hides CPU solver names in plot titles for cleaner presentation",
        )

        # Optimize button
        optimize_button = st.button(
            "🚀 Run Optimization", type="primary", width="stretch"
        )

    # Main content area with GPU vs CPU comparison
    gpu_col, cpu_col = st.columns([1, 1])

    with gpu_col:
        st.markdown('<div class="gpu-section">', unsafe_allow_html=True)
        st.subheader("🚀 GPU Solver (cuOpt)")
        gpu_progress_placeholder = st.empty()
        st.markdown("**📊 Portfolio Visualization**")
        gpu_portfolio_placeholder = st.empty()
        st.markdown("**📈 Optimization Results**")
        gpu_results_placeholder = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

    with cpu_col:
        st.markdown('<div class="cpu-section">', unsafe_allow_html=True)
        # In blog mode, hide CPU solver name
        cpu_header = (
            "💻 CPU Solver" if blog_mode else f"💻 CPU Solver ({cpu_solver_choice})"
        )
        st.subheader(cpu_header)
        cpu_progress_placeholder = st.empty()
        st.markdown("**📊 Portfolio Visualization**")
        cpu_portfolio_placeholder = st.empty()
        st.markdown("**📈 Optimization Results**")
        cpu_results_placeholder = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

    # Progress and status
    status_placeholder = st.empty()

    # Run optimization when button is pressed
    if optimize_button:
        with status_placeholder.container():
            st.info("🔄 Starting GPU vs CPU solver comparison...")
            progress_bar = st.progress(0)

            try:
                # Update progress
                progress_bar.progress(10)
                st.info(f"📊 Setting up {opt_method} parameters...")

                # Create parameters based on optimization method
                try:
                    if opt_method == "Mean-CVaR":
                        opt_params = CvarParameters(
                            w_min=w_min_dict,
                            w_max=w_max_dict,
                            c_min=c_min,
                            c_max=c_max,
                            L_tar=L_tar,
                            T_tar=None,
                            cvar_limit=None,
                            cardinality=cardinality,
                            risk_aversion=risk_aversion,
                            confidence=confidence,
                        )
                    else:
                        opt_params = MeanVarianceParameters(
                            w_min=w_min_dict,
                            w_max=w_max_dict,
                            c_min=c_min,
                            c_max=c_max,
                            L_tar=L_tar,
                            T_tar=None,
                            var_limit=var_limit if enable_var_limit else None,
                            cardinality=cardinality,
                            risk_aversion=risk_aversion,
                        )
                except Exception as e:
                    st.error(f"❌ Error creating parameters: {str(e)}")
                    st.stop()

                progress_bar.progress(20)
                st.info("📈 Loading data and calculating returns...")

                # Set up data path
                try:
                    data_path = workspace_root / "data" / "stock_data" / f"{dataset_name}.csv"
                    if not data_path.exists():
                        data_path = Path(
                            f"/home/scratch.phuo_wwfo/github/cufolio/data/stock_data/{dataset_name}.csv"
                        )
                    if not data_path.exists():
                        st.error(f"❌ Dataset not found: {dataset_name}.csv")
                        st.stop()
                except Exception as e:
                    st.error(f"❌ Error setting up data path: {str(e)}")
                    st.stop()

                # Calculate returns
                try:
                    regime_dict = {
                        "name": "streamlit_optimization",
                        "range": (
                            start_date.strftime("%Y-%m-%d"),
                            end_date.strftime("%Y-%m-%d"),
                        ),
                    }
                    st.info(
                        f"📅 Analyzing period: {regime_dict['range'][0]} to {regime_dict['range'][1]}"
                    )

                    returns_compute_settings = ReturnsComputeSettings(
                        return_type=return_type, freq=1
                    )
                    returns_dict = utils.calculate_returns(
                        str(data_path),
                        regime_dict,
                        returns_compute_settings
                    )

                    if opt_method == "Mean-CVaR":
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
                        returns_dict = cvar_utils.generate_cvar_data(
                            returns_dict,
                            scenario_generation_settings
                        )
                except Exception as e:
                    st.error(f"❌ Error calculating returns: {str(e)}")
                    import traceback
                    st.text("Debug trace:")
                    st.code(traceback.format_exc())
                    st.stop()

                progress_bar.progress(40)
                st.info("🔧 Setting up optimization problem...")

                # Create optimizer
                try:
                    if opt_method == "Mean-CVaR":
                        optimizer_problem = cvar_optimizer.CVaR(
                            returns_dict=returns_dict,
                            cvar_params=opt_params,
                        )
                    else:
                        optimizer_problem = mean_variance_optimizer.MeanVariance(
                            returns_dict=returns_dict,
                            mean_variance_params=opt_params,
                        )
                except Exception as e:
                    st.error(f"❌ Error creating optimization problem: {str(e)}")
                    import traceback
                    st.text("Debug trace:")
                    st.code(traceback.format_exc())
                    st.stop()

                progress_bar.progress(50)
                st.info(
                    f"⚡ Running GPU (cuOpt) and CPU ({cpu_solver_choice}) solvers..."
                )

                # Setup solver configurations
                gpu_settings = None

                cpu_solver_map = {
                    "HIGHS": cp.HIGHS,
                    "CLARABEL": cp.CLARABEL,
                    "ECOS": cp.ECOS,
                    "OSQP": cp.OSQP,
                    "SCS": cp.SCS,
                }

                cpu_settings = {
                    "solver": cpu_solver_map.get(cpu_solver_choice, cp.CLARABEL),
                    "verbose": False,
                }

                try:
                    import cuopt
                    gpu_settings = {"solver": cp.CUOPT, "verbose": False}
                    st.info("🚀 GPU solver (cuOpt) available")
                except ImportError:
                    st.warning(
                        "⚠️ GPU solver (cuOpt) not available - will show CPU-only results"
                    )

                progress_bar.progress(60)

                st.info("🚀 Running GPU and CPU optimizations simultaneously...")
                parallel_results = run_parallel_optimizations(
                    optimizer_problem,
                    gpu_settings,
                    cpu_settings,
                    cpu_solver_choice,
                    gpu_progress_placeholder,
                    cpu_progress_placeholder,
                )

                # Extract results from parallel execution
                gpu_results, gpu_portfolio, gpu_solve_time = None, None, None
                cpu_results, cpu_portfolio, cpu_solve_time = None, None, None

                if "GPU" in parallel_results and parallel_results["GPU"]["success"]:
                    gpu_results = parallel_results["GPU"]["results"]
                    gpu_portfolio = parallel_results["GPU"]["portfolio"]
                    gpu_solve_time = parallel_results["GPU"]["solve_time"]
                elif gpu_settings:
                    # GPU was attempted but failed
                    with gpu_progress_placeholder.container():
                        error_msg = parallel_results.get("GPU", {}).get(
                            "error", "Unknown error"
                        )
                        st.error(f"❌ GPU optimization failed: {error_msg}")
                else:
                    # GPU not available
                    with gpu_progress_placeholder.container():
                        st.error("❌ GPU solver not available")

                if "CPU" in parallel_results and parallel_results["CPU"]["success"]:
                    cpu_results = parallel_results["CPU"]["results"]
                    cpu_portfolio = parallel_results["CPU"]["portfolio"]
                    cpu_solve_time = parallel_results["CPU"]["solve_time"]
                else:
                    # CPU failed
                    with cpu_progress_placeholder.container():
                        error_msg = parallel_results.get("CPU", {}).get(
                            "error", "Unknown error"
                        )
                        st.error(f"❌ CPU optimization failed: {error_msg}")

                progress_bar.progress(90)
                st.info("📊 Displaying solver comparison results...")

                # Prepare backtest parameters if enabled
                backtest_params = None
                if (
                    enable_backtest
                    and "backtest_start" in locals()
                    and "backtest_end" in locals()
                ):
                    backtest_params = {
                        "dataset_name": dataset_name,
                        "start_date": backtest_start,
                        "end_date": backtest_end,
                        "return_type": return_type,
                        "method": backtest_method,
                    }

                # Display results for each solver
                if gpu_results is not None and gpu_portfolio is not None:
                    display_solver_results(
                        gpu_results,
                        gpu_portfolio,
                        optimizer_problem,
                        gpu_solve_time,
                        "GPU",
                        gpu_portfolio_placeholder,
                        gpu_results_placeholder,
                        enable_backtest,
                        backtest_params,
                        blog_mode,
                    )
                else:
                    with gpu_portfolio_placeholder.container():
                        st.info("🚀 GPU solver not available or failed")
                    with gpu_results_placeholder.container():
                        st.info("No GPU results to display")

                if cpu_results is not None and cpu_portfolio is not None:
                    display_solver_results(
                        cpu_results,
                        cpu_portfolio,
                        optimizer_problem,
                        cpu_solve_time,
                        "CPU",
                        cpu_portfolio_placeholder,
                        cpu_results_placeholder,
                        enable_backtest,
                        backtest_params,
                        blog_mode,
                    )
                else:
                    with cpu_portfolio_placeholder.container():
                        st.error("❌ CPU solver failed")
                    with cpu_results_placeholder.container():
                        st.error("No CPU results to display")

                # Solver comparison summary
                with status_placeholder.container():
                    if gpu_results is not None and cpu_results is not None:
                        # In blog mode, hide CPU solver name
                        cpu_name = "CPU" if blog_mode else f"CPU ({cpu_solver_choice})"
                        st.success(
                            f"✅ Solver comparison completed! GPU (cuOpt) vs {cpu_name}"
                        )

                        # Performance comparison (parallel execution)
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            gpu_time = gpu_solve_time if gpu_solve_time else 0
                            cpu_time = cpu_solve_time if cpu_solve_time else 0
                            if gpu_time > 0 and cpu_time > 0:
                                if gpu_time < cpu_time:
                                    st.metric(
                                        "⚡ GPU Performance",
                                        f"{cpu_time/gpu_time:.1f}x faster than CPU",
                                    )
                                    st.caption("(Both solved in parallel)")
                                else:
                                    st.metric(
                                        "⚡ CPU Performance",
                                        f"{gpu_time/cpu_time:.1f}x faster than GPU",
                                    )
                                    st.caption("(Both solved in parallel)")
                            else:
                                st.metric("⏱️ Performance", "Comparison unavailable")

                        with col2:
                            gpu_obj = gpu_results.get("obj", 0)
                            cpu_obj = cpu_results.get("obj", 0)
                            obj_diff = abs(gpu_obj - cpu_obj)
                            st.metric("🎯 Solution Difference", f"{obj_diff:.6f}")

                        with col3:
                            st.metric(
                                "🚀 GPU Time",
                                f"{gpu_solve_time:.2f}s" if gpu_solve_time else "N/A",
                            )
                            # In blog mode, hide CPU solver name
                            cpu_time_label = (
                                "💻 CPU Time"
                                if blog_mode
                                else f"💻 {cpu_solver_choice} Time"
                            )
                            st.metric(
                                cpu_time_label,
                                f"{cpu_solve_time:.2f}s" if cpu_solve_time else "N/A",
                            )
                    elif cpu_results is not None:
                        # In blog mode, hide CPU solver name
                        cpu_name = "CPU" if blog_mode else f"CPU ({cpu_solver_choice})"
                        st.success(f"✅ {cpu_name} solver completed successfully!")
                        st.info("🚀 GPU solver not available for comparison")
                    elif gpu_results is not None:
                        st.success("✅ GPU (cuOpt) solver completed successfully!")
                        # In blog mode, hide CPU solver name
                        cpu_name = "CPU" if blog_mode else f"CPU ({cpu_solver_choice})"
                        st.error(f"❌ {cpu_name} solver failed")
                    else:
                        st.error("❌ All solvers failed!")

                progress_bar.progress(100)

            except Exception as e:
                status_placeholder.error(f"❌ Error during optimization: {str(e)}")
                st.error("Please check your parameters and try again.")
                import traceback

                st.text("Debug info:")
                st.text(traceback.format_exc())


if __name__ == "__main__":
    main()
