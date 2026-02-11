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

"""Utility functions for portfolio optimization and data processing."""

import os

from typing import Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf

from .cvar_parameters import CvarParameters
from .mean_variance_parameters import MeanVarianceParameters
from .settings import ReturnsComputeSettings, ScenarioGenerationSettings


def get_input_data(filepath):
    """Load input data from file."""
    _, file_extension = os.path.splitext(filepath)
    file_extension = file_extension.lower()

    if file_extension == ".csv":
        df = pd.read_csv(filepath, index_col=0)
    elif file_extension == ".parquet":
        df = pd.read_parquet(filepath)
    elif file_extension in [".xls", ".xlsx"]:
        df = pd.read_excel(filepath)
    elif file_extension == ".json":
        df = pd.read_json(filepath)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")
    df = df.dropna(axis=1)
    return df


def calculate_returns(
    input_dataset: Union[pd.DataFrame, str],
    regime_dict: dict = None,
    returns_compute_settings: ReturnsComputeSettings = None,
):
    """
    Preprocess price data and compute returns for a specified time period.

    Calculates log returns (or other return types) and computes the mean
    and covariance matrix for downstream portfolio optimization.

    Parameters
    ----------
    input_dataset : pd.DataFrame or str
        Price data as a DataFrame or path to the input dataset file.
    regime_dict : dict, optional
        Market regime specification with format {'name': str, 'range': (start, end)}.
    returns_compute_settings : ReturnsComputeSettings, optional
        Configuration for return calculation including return type, frequency,
        and computation device. Uses defaults if not provided.

    Returns
    -------
    dict
        Dictionary containing computed returns, mean, covariance, and metadata.
    """
    if returns_compute_settings is None:
        returns_compute_settings = ReturnsComputeSettings()

    return_type = returns_compute_settings.return_type.upper()
    freq = returns_compute_settings.freq

    if isinstance(input_dataset, str):
        input_data = get_input_data(input_dataset)
    else:
        input_data = input_dataset

    if regime_dict is None or regime_dict.get("range") is None:
        input_data = input_data
        regime_dict = {"name": "Default", "range": (input_data.index[0], input_data.index[-1])}
    else:
        start, end = regime_dict["range"]
        input_data = input_data.loc[start:end]
        
    input_data = input_data.dropna(axis=1)

    if return_type == "LOG":
        returns_dataframe = calculate_log_returns(input_data, freq)
    elif return_type == "PNL":
        returns_dataframe = input_data
    elif return_type == "LINEAR":
        returns_dataframe = compute_linear_returns(input_data, freq)
    elif return_type == "ABSOLUTE":
        returns_dataframe = compute_absolute_returns(input_data, freq)
    else:
        raise NotImplementedError("Invalid return type!")

    returns_array = returns_dataframe.to_numpy()
    m = np.mean(returns_array, axis=0)
    cov = np.cov(returns_array.transpose())

    returns_dict = {
        "return_type": return_type,
        "returns": returns_dataframe,
        "regime": regime_dict,
        "dates": returns_dataframe.index,
        "mean": m,
        "covariance": cov,
        "tickers": list(input_data.columns),
    }

    return returns_dict


def calculate_log_returns(price_data, freq=1):
    """compute the log returns given a price dataframe"""
    # compute the log returns
    returns_dataframe = np.log(price_data / price_data.shift(freq))

    return returns_dataframe.dropna(how="all").fillna(0)


def compute_linear_returns(price_data, freq=1):
    """
    compute the simple returns using freq. For example, 
    freq = 1 means (today - yesterday) / yesterday.
    """
    returns_dataframe = price_data.pct_change(freq)
    returns_dataframe = returns_dataframe.dropna(how="all")
    returns_dataframe = returns_dataframe.fillna(0)

    return returns_dataframe


def compute_absolute_returns(price_data, freq=1):
    """
    compute the absolute returns using freq. 
    For example, freq = 1 means today - yesterday.
    """
    returns_dataframe = price_data.diff(freq)
    returns_dataframe = returns_dataframe.dropna(how="all")
    returns_dataframe = returns_dataframe.fillna(0)

    return returns_dataframe


def plot_efficient_frontier(
    risk_measure,
    result_dataframe,
    single_asset_portfolio,
    custom_portfolios,
    key_portfolios,
    verbose=False,
    title=None,
    show_plot=True,
    EF_plot_png_name=None,
    notional=1e7,
):
    """
    plot the efficient frontier using the optimization results of different
    risk-aversion levels in Seaborn.

    Parameters:
    :risk_measure: str
    :result_dataframe: Pandas DataFrame - (num_risks_levels, ?) where each row
        records the result of the optimization w.r.t. a certain risk level
    :single_asset_portfolio: Pandas DataFrame - (n_assets, #performance metrics)
        each row records the performance of the portfolio made up of one single asset
    :key_portfolios: dict - {portfolio_name: marker} of names of the portfolios
        (and corresponding markers) to highlight on the efficient frontier
        (e.g. min var, max Sharpe, max return, etc.)
    :custom_portfolios: Pandas DataFrame - (#user inputs, #performance metrics)
        each row records the performance of a custom portfolio from user input
    :show_plot: bool - whether to show plot
    :EF_plot_png_name: str - save the figure under the name EF_plot_png_name
    """
    # Apply consistent styling
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("paper", font_scale=0.9)
    sns.set_palette(palette="Blues_d")
    plt.figure(figsize=(10, 7), dpi=300)

    # Create scaled versions of the data for plotting
    result_dataframe_scaled = result_dataframe.copy()
    result_dataframe_scaled[f"{risk_measure}_percent"] = (
        result_dataframe_scaled[risk_measure] * 100
    )
    result_dataframe_scaled["return_scaled"] = (
        result_dataframe_scaled["return"] * notional
    )

    if key_portfolios is not None:
        # plot the markers for the key portfolios
        example_portfolio = pd.DataFrame({}, columns=result_dataframe.columns)
        for portfolio_name, marker in key_portfolios.items():
            portfolio_idx = get_portfolio(result_dataframe, portfolio_name)
            example_portfolio = pd.concat(
                [example_portfolio, result_dataframe.iloc[portfolio_idx].to_frame().T]
            )
            portfolio_data_scaled = (
                result_dataframe_scaled.iloc[portfolio_idx].to_frame().T
            )
            sns.scatterplot(
                data=portfolio_data_scaled,
                x=f"{risk_measure}_percent",
                y="return_scaled",
                marker=marker,
                s=100,
                color="darkorange",
                label=portfolio_name,
                legend=True,
                zorder=2,
            )
        example_portfolio = example_portfolio.reset_index()

        if verbose:
            # create the annotation box for the key portfolios
            _ = []  # annotated_points (unused)
            _ = []  # annotation_list (unused)

            offset_list = [(-15, -150), (20, -70), (-15, -70)]

            for row_idx, row in example_portfolio.iterrows():
                point = (row.loc[risk_measure] * 100, row.loc["return"] * notional)

                annotation = ""
                weights_dict, cash = row["optimal portfolio"]
                for ticker, weight in weights_dict.items():
                    if weight > 5e-2 or weight < -5e-2:
                        annotation += ticker + f": {weight: .2f}\n"

                annotation += f"cash: {cash: .2f}"
                annotation = annotation.rstrip("\n")

                plt.annotate(
                    annotation,
                    xy=point,
                    ha="left",
                    xytext=offset_list[row_idx],
                    textcoords="offset points",
                    fontsize=8,
                    bbox=dict(
                        boxstyle="round,pad=0.4", facecolor="#e8dff5", edgecolor="black"
                    ),
                    arrowprops=dict(
                        arrowstyle="->", connectionstyle="arc3,rad=0.3", color="black"
                    ),
                )

    # create line for efficient frontier
    sns.lineplot(
        data=result_dataframe_scaled,
        x=f"{risk_measure}_percent",
        y="return_scaled",
        linewidth=3,
        zorder=1,
        label="Optimal Portfolios",
    )
    plt.legend()

    custom_portfolio_markers = ["s", "^", "v", "<", ">", "p", "h"]
    if not custom_portfolios.empty:
        for i in range(0, len(custom_portfolios)):
            portfolio = custom_portfolios.iloc[i]
            annotation = portfolio["portfolio_name"]
            plt.scatter(
                x=portfolio[risk_measure] * 100,  # Convert to percentage
                y=portfolio["return"] * notional,  # Scale by notional
                marker=custom_portfolio_markers[i],
                color=".2",
                zorder=4,
                label=annotation,
            )
    plt.legend()

    # scatter plot the single asset portfolios
    single_asset_scaled = single_asset_portfolio.copy()
    single_asset_scaled[f"{risk_measure}_percent"] = (
        single_asset_scaled[risk_measure] * 100
    )
    single_asset_scaled["return_scaled"] = single_asset_scaled["return"] * notional

    sns.scatterplot(
        data=single_asset_scaled,
        x=f"{risk_measure}_percent",
        y="return_scaled",
        hue="variance",
        size="variance",
        palette="icefire",
        legend=False,
        zorder=3,
    )

    for i in range(0, len(single_asset_portfolio)):
        plt.annotate(
            f"{single_asset_portfolio.index[i]}",
            (
                single_asset_portfolio[risk_measure][i] * 100,
                single_asset_portfolio["return"][i] * notional,
            ),
            textcoords="offset points",
            xytext=(2, 3) if i % 2 == 0 else (-4, -6),
            fontsize=7,
            ha="center",
        )

    # Set axis labels with proper scaling
    plt.xlabel("Conditional Value at Risk (CVaR %)", fontsize=10)
    plt.ylabel(f"Expected Return (${notional / 1e6:.0f}M Notional)", fontsize=10)

    if not title:
        plt.title(
            f"Efficient Frontier with {len(single_asset_portfolio)} Stocks",
            fontsize=11,
            pad=15,
        )
    else:
        plt.title(title, fontsize=11, pad=15)
    if EF_plot_png_name:
        plt.savefig(EF_plot_png_name)
    if show_plot:
        plt.show()


def get_portfolio(result, portfolio_name):
    """Extract specific portfolio from optimization results."""
    portfolio_name = portfolio_name.lower()
    if portfolio_name == "min_var":
        min_value = result["risk"].min()
        idx = result[result["risk"] == min_value].index[0]
    elif portfolio_name == "max_sharpe":
        max_sharpe = result["sharpe"].max()
        idx = result[result["sharpe"] == max_sharpe].index[0]
    elif portfolio_name == "max_return":
        max_return = result["return"].max()
        idx = result[result["return"] == max_return].index[-1]
    else:
        raise ValueError(
            "portfolio_name should be a string (e.g. min_var, max_sharpe, max_return)"
        )

    return idx


def portfolio_plot_with_backtest(
    portfolio,
    backtester,
    cut_off_date,
    backtest_plot_title,
    save_plot=False,
    results_dir="results",
):
    """
    Create side-by-side portfolio allocation and backtest performance plots.

    Displays portfolio allocation as a horizontal bar chart alongside
    cumulative returns comparison with benchmarks.

    Parameters
    ----------
    portfolio : Portfolio
        Portfolio object to display allocation for
    backtester : portfolio_backtester
        Backtester object containing test portfolio and benchmarks
    cut_off_date : str
        Date to mark with vertical line on backtest plot
    backtest_plot_title : str
        Title for the backtest plot
    save_plot : bool, default False
        Whether to save the combined plot to results directory
    results_dir : str, default "results"
        Directory path where plots will be saved
    """
    # Apply consistent styling without whitegrid for portfolio plot
    sns.set_context("paper", font_scale=0.9)

    # Create subplots with appropriate sizing for side-by-side display
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=300)

    # Plot portfolio allocation
    ax1 = portfolio.plot_portfolio(ax=ax1, show_plot=False)

    # Completely reset and apply very subtle grid to portfolio plot
    ax1.grid(False)  # Turn off any existing grid first
    ax1.grid(True, axis="x", alpha=0.1, color="#E0E0E0", linestyle="-", linewidth=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_color("#E0E0E0")
    ax1.spines["bottom"].set_color("#E0E0E0")
    ax1.set_axisbelow(True)

    # Apply whitegrid style only to backtest plot
    with plt.style.context("seaborn-v0_8-whitegrid"):
        # Plot backtest results
        _, ax2 = backtester.backtest_against_benchmarks(
            plot_returns=True,
            ax=ax2,
            cut_off_date=cut_off_date,
            title=backtest_plot_title,
            save_plot=False,
        )

    # Ensure backtest grid is subtle and consistent
    ax2.grid(True, alpha=0.1, color="#E0E0E0", linewidth=0.3)
    ax2.set_axisbelow(True)

    plt.tight_layout()

    # Save combined plot if requested
    if save_plot:
        import os

        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)

        # Generate filename
        portfolio_name = (
            portfolio.name.replace(" ", "_").lower() if portfolio.name else "portfolio"
        )
        test_method = backtester.test_method.replace("_", "")

        filename = f"combined_{portfolio_name}_{test_method}_analysis.png"
        filepath = os.path.join(results_dir, filename)

        # Save with high quality
        plt.savefig(
            filepath,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )

        print(f"Combined plot saved: {filepath}")

    plt.show()


def compare_results(*results_list):
    """
    Compare and display results from multiple solvers in tabular format.

    Args:
        *results_list: Result dictionaries from different solvers.
    """
    results = [r for r in results_list if r is not None]
    if not results:
        print("No results available")
        return

    # Find common numeric keys, sorted: solve time, obj, then rest
    common = set.intersection(*[set(r.keys()) for r in results])
    keys = sorted(
        [k for k in common if k != "solver" and isinstance(results[0].get(k), (int, float))],
        key=lambda x: (0 if x == "solve time" else 1 if x == "obj" else 2, x)
    )

    # Print table
    print("\n" + "=" * 70)
    print("SOLVER COMPARISON")
    print("=" * 70)
    print(f"{'Solver':<15}" + "".join(f" {k:<12}" for k in keys))
    print("-" * 70)
    for r in results:
        print(f"{r.get('solver', 'Unknown'):<15}" + "".join(f" {(r.get(k) or 0):<12.6f}" for k in keys))

    # Objective differences
    if len(results) > 1 and "obj" in keys:
        print("\nObjective Differences:")
        for i, r1 in enumerate(results):
            for r2 in results[i + 1:]:
                print(f"  {r1.get('solver')} vs {r2.get('solver')}: {abs(r1.get('obj', 0) - r2.get('obj', 0)):.8f}")

    print()  # Add blank line for better readability


def download_data(dataset_dir):
    """
    Download the data for the given dataset name.
    """

    tickers = [
        'A', 'AAPL', 'ABT', 'ACGL', 'ACN', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK', 'AEE', 'AEP', 'AES', 'AFL', 'AIG', 'AIZ', 'AJG', 'AKAM', 'ALB', 'ALGN',
        'ALL', 'AMAT', 'AMD', 'AME', 'AMGN', 'AMT', 'AMZN', 'AON', 'AOS', 'APA', 'APD', 'APH', 'ARE', 'ATO', 'AVB', 'AVY', 'AXON', 'AXP', 'AZO',
        'BA', 'BAC', 'BALL', 'BAX', 'BBWI', 'BBY', 'BDX', 'BEN', 'BG', 'BIIB', 'BIO', 'BK', 'BKNG', 'BKR', 'BLK', 'BMY', 'BRO', 'BSX', 'BWA', 'BXP',
        'C', 'CAG', 'CAH', 'CAT', 'CB', 'CBRE', 'CCI', 'CCL', 'CDNS', 'CHD', 'CHRW', 'CI', 'CINF', 'CL', 'CLX', 'CMA', 'CMCSA', 'CME', 'CMI', 'CMS',
        'CNC', 'CNP', 'COF', 'COO', 'COP', 'COR', 'COST', 'CPB', 'CPRT', 'CPT', 'CRL', 'CRM', 'CSCO', 'CSGP', 'CSX', 'CTAS', 'CTRA', 'CTSH', 'CVS', 'CVX',
        'D', 'DD', 'DE', 'DECK', 'DGX', 'DHI', 'DHR', 'DIS', 'DLR', 'DLTR', 'DOC', 'DOV', 'DPZ', 'DRI', 'DTE', 'DUK', 'DVA', 'DVN',
        'EA', 'EBAY', 'ECL', 'ED', 'EFX', 'EG', 'EIX', 'EL', 'ELV', 'EMN', 'EMR', 'EOG', 'EQIX', 'EQR', 'EQT', 'ES', 'ESS', 'ETN', 'ETR', 'EVRG',
        'EW', 'EXC', 'EXPD', 'EXR', 'F', 'FAST', 'FCX', 'FDS', 'FDX', 'FE', 'FFIV', 'FI', 'FICO', 'FIS', 'FITB', 'FMC', 'FRT',
        'GD', 'GE', 'GEN', 'GILD', 'GIS', 'GL', 'GLW', 'GOOG', 'GOOGL', 'GPC', 'GPN', 'GRMN', 'GS', 'GWW',
        'HAL', 'HAS', 'HBAN', 'HD', 'HIG', 'HOLX', 'HON', 'HPQ', 'HRL', 'HSIC', 'HST', 'HSY', 'HUBB', 'HUM',
        'IBM', 'IDXX', 'IEX', 'IFF', 'ILMN', 'INCY', 'INTC', 'INTU', 'IP', 'IPG', 'IRM', 'ISRG', 'IT', 'ITW', 'IVZ',
        'J', 'JBHT', 'JBL', 'JCI', 'JKHY', 'JNJ', 'JPM', 'K', 'KEY', 'KIM', 'KLAC', 'KMB', 'KMX', 'KO', 'KR',
        'L', 'LEN', 'LH', 'LHX', 'LIN', 'LKQ', 'LLY', 'LMT', 'LNT', 'LOW', 'LRCX', 'LUV', 'LVS',
        'MAA', 'MAR', 'MAS', 'MCD', 'MCHP', 'MCK', 'MCO', 'MDLZ', 'MDT', 'MET', 'MGM', 'MHK', 'MKC', 'MKTX', 'MLM', 'MMC', 'MMM', 'MNST', 'MO', 'MOH',
        'MOS', 'MPWR', 'MRK', 'MS', 'MSFT', 'MSI', 'MTB', 'MTCH', 'MTD', 'MU',
        'NDAQ', 'NDSN', 'NEE', 'NEM', 'NFLX', 'NI', 'NKE', 'NOC', 'NRG', 'NSC', 'NTAP', 'NTRS', 'NUE', 'NVDA', 'NVR',
        'O', 'ODFL', 'OKE', 'OMC', 'ON', 'ORCL', 'ORLY', 'OXY',
        'PAYX', 'PCAR', 'PCG', 'PEG', 'PEP', 'PFE', 'PFG', 'PG', 'PGR', 'PH', 'PHM', 'PKG', 'PLD', 'PNC', 'PNR', 'PNW', 'POOL', 'PPG', 'PPL', 'PRU',
        'PSA', 'PTC', 'PWR', 'QCOM',
        'RCL', 'REG', 'REGN', 'RF', 'RHI', 'RJF', 'RL', 'RMD', 'ROK', 'ROL', 'ROP', 'ROST', 'RSG', 'RTX', 'RVTY',
        'SBAC', 'SBUX', 'SCHW', 'SHW', 'SJM', 'SLB', 'SNA', 'SNPS', 'SO', 'SPG', 'SPGI', 'SRE', 'STE', 'STLD', 'STT', 'STX', 'STZ', 'SWK', 'SWKS', 'SYK',
        'SYY', 'T', 'TAP', 'TDY', 'TECH', 'TER', 'TFC', 'TFX', 'TGT', 'TJX', 'TMO', 'TPR', 'TRMB', 'TROW', 'TRV', 'TSCO', 'TSN', 'TT', 'TTWO', 'TXN',
        'TXT', 'TYL', 'UDR', 'UHS', 'UNH', 'UNP', 'UPS', 'URI', 'USB',
        'VLO', 'VMC', 'VRSN', 'VRTX', 'VTR', 'VTRS', 'VZ',
        'WAB', 'WAT', 'WDC', 'WEC', 'WELL', 'WFC', 'WM', 'WMB', 'WMT', 'WRB', 'WST', 'WTW', 'WY', 'WYNN',
        'XEL', 'XOM', 'YUM', 'ZBH', 'ZBRA'
    ]

    start_date = "2005-01-01"
    end_date = "2025-01-01"

    data = yf.download(tickers, start=start_date, end=end_date, timeout = 30)

    data = data['Close'].dropna(axis = 1)

    data.to_csv(dataset_dir)


def create_synthetic_stock_dataset(
    training_directory: str, regime_name: str, regime_range: tuple, num_synthetic: int
):
    """Create synthetic stock dataset based on training data.

    Args:
        training_directory (str): Path to the training data directory.
        regime_name (str): Name of the market regime.
        regime_range (tuple): Date range for the regime (start_date, end_date).
        num_synthetic (int): Number of synthetic datasets to generate.

    Returns:
        str: Path to the saved synthetic dataset file.

    Raises:
        ValueError: If num_synthetic is less than or equal to 0.

    Example:
        >>> training_dir = "data/stock_data/sp500.csv"
        >>> regime = "bull_market"
        >>> date_range = ("2020-01-01", "2021-12-31")
        >>> save_path = create_synthetic_stock_dataset(
        ...     training_dir,
        ...     regime,
        ...     date_range,
        ...     num_synthetic=100
        ... )
        >>> print(save_path)  # data/stock_data/synthetic-bull_market-size_500.csv
    """
    from . import scenario_generation  # Lazy import

    if num_synthetic <= 0:
        raise ValueError("Please provide a valid integer for num_synthetic!")

    synthetic_data = scenario_generation.generate_synthetic_stock_data(
        dataset_directory=training_directory,
        num_synthetic=num_synthetic,
        fit_range=regime_range,
        generate_range=regime_range,
    )
    dataset_size = len(synthetic_data.columns)

    save_name = "synthetic-" + regime_name + f"-size_{dataset_size}.csv"
    save_path = os.path.join(os.path.dirname(training_directory), save_name)
    synthetic_data.to_csv(save_path)

    return save_path


def optimize_market_regimes(
    input_file_name: str,
    returns_compute_settings: ReturnsComputeSettings,
    all_regimes: dict,
    params: Union[CvarParameters, MeanVarianceParameters],
    solver_settings_list: list[dict],
    scenario_generation_settings: ScenarioGenerationSettings = None,
    results_csv_file_name: str = None,
    num_synthetic: int = 0,
    print_results: bool = True,
):
    """
    Compare optimization performance across different regimes and solvers.

    Automatically detects whether to use CVaR or Mean-Variance optimization
    based on the type of parameters passed.

    Parameters
    ----------
    input_file_name : str
        Path to input data file.
    returns_compute_settings : ReturnsComputeSettings
        Configuration for computing returns from price data.
    all_regimes : dict
        Dictionary of regimes to test with format {'regime_name': regime_range}.
    params : CvarParameters or MeanVarianceParameters
        Optimization parameters. The type determines which optimizer to use:
        - CvarParameters: Uses CVaR optimization (requires scenario_generation_settings)
        - MeanVarianceParameters: Uses Mean-Variance optimization
    solver_settings_list : list[dict]
        List of solver configurations to test.
    scenario_generation_settings : ScenarioGenerationSettings, optional
        Configuration for generating return scenarios. Required when using
        CvarParameters, ignored for MeanVarianceParameters.
    results_csv_file_name : str, optional
        CSV filename to save results.
    num_synthetic : int, optional
        Number of synthetic data copies to generate (0 = none).
        Only applicable for CVaR optimization.
    print_results : bool, optional
        Whether to print optimization results.

    Returns
    -------
    pd.DataFrame
        Results dataframe with solver performance metrics per regime.

    Raises
    ------
    ValueError
        If CvarParameters is passed without scenario_generation_settings.
        If params is neither CvarParameters nor MeanVarianceParameters.
    """
    # Determine optimization type and risk measure based on params type
    if isinstance(params, CvarParameters):
        if scenario_generation_settings is None:
            raise ValueError(
                "scenario_generation_settings is required when using Mean-CVaR optimization"
            )
        risk_measure = "CVaR"
        from . import cvar_optimizer  # Lazy import
        from . import cvar_utils  # For generate_cvar_data
    elif isinstance(params, MeanVarianceParameters):
        risk_measure = "variance"
        from . import mean_variance_optimizer  # Lazy import
    else:
        raise ValueError(
            f"params must be either CvarParameters or MeanVarianceParameters, "
            f"got {type(params).__name__}"
        )

    if len(solver_settings_list) == 0:
        raise ValueError("Please provide at least one solver settings!")

    # Helper function to extract solver name from settings
    def get_solver_name(settings):
        """Extract solver name from solver settings dict."""
        if "solver" in settings:
            solver_obj = settings["solver"]
            return str(solver_obj).replace("cp.", "").replace("solvers.", "")
        else:
            raise ValueError(f"Please provide a solver name in the format 'solver': <solver_name>")

    # Build column names dynamically based on solvers
    columns = ["regime"]
    solver_names = []
    for settings in solver_settings_list:
        solver_name = get_solver_name(settings)
        solver_names.append(solver_name)
        columns.extend(
            [
                f"{solver_name}-obj",
                f"{solver_name}-solve_time",
                f"{solver_name}-return",
                f"{solver_name}-{risk_measure}",
                f"{solver_name}-optimal_portfolio",
            ]
        )

    result_rows = []

    for regime_name, regime_range in all_regimes.items():
        print("=" * 70)
        print(f"Processing Regime: {regime_name}")
        print("=" * 70)

        # Handle synthetic data generation (CVaR only)
        if risk_measure == "CVaR" and num_synthetic > 0:
            input_data_directory = create_synthetic_stock_dataset(
                input_file_name, regime_name, regime_range, num_synthetic
            )
        else:
            input_data_directory = input_file_name

        # Create returns_dict for the current regime
        curr_regime = {"name": regime_name, "range": regime_range}
        returns_dict = calculate_returns(
            input_data_directory, curr_regime, returns_compute_settings
        )

        # Generate CVaR scenario data if needed
        if risk_measure == "CVaR":
            returns_dict = cvar_utils.generate_cvar_data(
                returns_dict, scenario_generation_settings
            )

        # Initialize result row for this regime
        result_row = {"regime": regime_name}

        # Solve with each solver
        for idx, solver_settings in enumerate(solver_settings_list):
            solver_name = solver_names[idx]
            print(f"\n--- Testing Solver: {solver_name} ---")

            # Set up optimization problem based on risk measure
            if risk_measure == "CVaR":
                problem = cvar_optimizer.CVaR(
                    returns_dict=returns_dict, cvar_params=params
                )
            else:
                problem = mean_variance_optimizer.MeanVariance(
                    returns_dict=returns_dict, mean_variance_params=params
                )

            # Solve optimization problem
            try:
                result, portfolio = problem.solve_optimization_problem(
                    solver_settings, print_results=print_results
                )

                # Store results with solver-specific column names
                result_row[f"{solver_name}-obj"] = result["obj"]
                result_row[f"{solver_name}-solve_time"] = result["solve time"]
                result_row[f"{solver_name}-return"] = result["return"]
                result_row[f"{solver_name}-{risk_measure}"] = result.get(
                    risk_measure, result.get(risk_measure.lower(), None)
                )
                result_row[f"{solver_name}-optimal_portfolio"] = portfolio.print_clean(
                    verbose=False
                )

                print(
                    f"  ✓ {solver_name} - Objective: {result['obj']:.6f}, "
                    f"Time: {result['solve time']:.4f}s"
                    f"--------------------------------"
                )

            except Exception as e:
                print(f"  ✗ {solver_name} failed: {str(e)}")
                result_row[f"{solver_name}-obj"] = None
                result_row[f"{solver_name}-solve_time"] = None
                result_row[f"{solver_name}-return"] = None
                result_row[f"{solver_name}-{risk_measure}"] = None
                result_row[f"{solver_name}-optimal_portfolio"] = None

        result_rows.append(result_row)

    # Create DataFrame from collected rows
    result_dataframe = pd.DataFrame(result_rows, columns=columns)

    print("\n" + "=" * 70)
    print("Optimization Complete!")
    print("=" * 70)
    print("\n")

    if results_csv_file_name:
        result_dataframe.to_csv(results_csv_file_name, index=False)
        print(f"Results saved to: {results_csv_file_name}")

    return result_dataframe