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
Base optimization classes and utilities for portfolio optimization.

Provides abstract base classes and common functionality shared across
different optimization algorithms, including weight constraint handling,
problem setup dispatch, solve dispatch, and portfolio state management.
"""

import copy
import os
import pickle
import time

import numpy as np
import pandas as pd

from .portfolio import Portfolio
from .settings import ApiSettings


class BaseOptimizer:
    """
    Base class for portfolio optimization algorithms.

    Provides common functionality for different optimization methods including
    weight constraint handling, problem setup/solve dispatch, parameter
    storage, and result printing.

    Attributes
    ----------
    returns_dict : dict
        Dictionary containing return data and asset information
    tickers : list
        Asset ticker symbols
    n_assets : int
        Number of assets in the portfolio
    risk_measure : str
        Risk measure type (e.g., "CVaR", "variance")
    weights_previous : np.ndarray
        Previous portfolio weights for turnover calculations
    """

    def __init__(
        self,
        returns_dict,
        params,
        api_settings=None,
        existing_portfolio=None,
        risk_measure="",
    ):
        """
        Initialize base optimizer with return data and portfolio state.

        Parameters
        ----------
        returns_dict : dict
            Dictionary containing asset returns data and tickers
        params : BaseParameters
            Optimization parameters (deep-copied internally)
        api_settings : ApiSettings, optional
            API configuration. Uses CVXPY with bounds if not provided.
        existing_portfolio : Portfolio or None
            Previous portfolio for turnover calculations.
            If None, creates uniform weights.
        risk_measure : str
            Risk measure identifier (e.g., "CVaR", "variance")
        """
        self.returns_dict = returns_dict
        self.tickers = returns_dict["tickers"]
        self.n_assets = len(self.tickers)
        self.risk_measure = risk_measure

        if not existing_portfolio:
            self.weights_previous = np.ones(self.n_assets) / self.n_assets
        else:
            self.weights_previous = existing_portfolio

        if api_settings is None:
            api_settings = ApiSettings()

        self.api_settings = api_settings
        self.api_choice = api_settings.api

        self.regime_name = returns_dict["regime"]["name"]
        self.regime_range = returns_dict["regime"]["range"]
        self.covariance = returns_dict["covariance"]
        self.existing_portfolio = existing_portfolio

        self.params = self._store_params(params)
        self.optimal_portfolio = None

    # ------------------------------------------------------------------
    # Parameter storage
    # ------------------------------------------------------------------

    def _store_params(self, params):
        """
        Deep-copy params and convert w_min / w_max to ndarray format.

        Parameters
        ----------
        params : BaseParameters
            Optimization parameter object

        Returns
        -------
        BaseParameters
            Deep-copied parameters with weight bounds as ndarrays
        """
        params_copy = copy.deepcopy(params)
        params_copy.w_min = self._update_weight_constraints(params_copy.w_min)
        params_copy.w_max = self._update_weight_constraints(params_copy.w_max)
        return params_copy

    def _update_weight_constraints(self, weight_constraints):
        """
        Convert weight constraints to numpy array format.

        Handles multiple input formats for weight constraints:
        - numpy array: used directly
        - dict: maps ticker names to constraint values
        - float: uniform constraint for all assets

        Parameters
        ----------
        weight_constraints : np.ndarray, dict, or float
            Weight constraint specification in various formats

        Returns
        -------
        np.ndarray
            Weight constraints as numpy array (length n_assets)

        Raises
        ------
        ValueError
            If constraint format is invalid or missing ticker specifications
        """

        if isinstance(weight_constraints, np.ndarray):
            updated_weight_constraints = weight_constraints

        elif isinstance(weight_constraints, dict):
            updated_weight_constraints = np.zeros(self.n_assets)
            for ticker_idx, ticker in enumerate(self.tickers):
                if ticker in weight_constraints.keys():
                    updated_weight_constraints[ticker_idx] = weight_constraints[ticker]
                elif "others" in weight_constraints.keys():
                    updated_weight_constraints[ticker_idx] = weight_constraints[
                        "others"
                    ]
                else:
                    raise ValueError(
                        "Must specify a weight constraint for each ticker or 'others'"
                    )

        elif isinstance(weight_constraints, float):
            updated_weight_constraints = np.full(self.n_assets, weight_constraints)
        else:
            raise ValueError("Invalid weight constraints")

        return updated_weight_constraints

    # ------------------------------------------------------------------
    # Problem setup dispatch
    # ------------------------------------------------------------------

    def _setup_optimization_problem(self):
        """
        Set up the optimization problem based on the selected API choice.

        Times the setup process, optionally scales risk aversion, then
        delegates to the API-specific setup method (_setup_cvxpy_problem
        or _setup_cuopt_problem).
        """
        set_up_start = time.time()

        if self.api_settings.scale_risk_aversion:
            self._scale_risk_aversion()

        if self.api_choice == "cvxpy":
            self._setup_cvxpy_problem()
            self._assign_cvxpy_parameter_values()

            pickle_path = self.api_settings.pickle_save_path
            if pickle_path is not None:
                self._save_problem_pickle(pickle_path)

        elif self.api_choice == "cuopt_python":
            self._validate_cuopt_setup()
            (
                self._cuopt_problem,
                self._cuopt_variables,
                self.cuopt_timing_dict,
            ) = self._setup_cuopt_problem()
        else:
            raise ValueError(f"Unsupported api_choice: {self.api_choice}")

        set_up_end = time.time()
        self.set_up_time = set_up_end - set_up_start

    def _validate_cuopt_setup(self):
        """Hook for subclasses to validate state before cuOpt setup.

        Override in subclasses to raise errors for unsupported
        configurations (e.g., quadratic constraints in a linear solver).
        """
        pass

    def _scale_risk_aversion(self):
        """Scale risk aversion heuristically. Must be implemented by subclasses."""
        raise NotImplementedError

    def _setup_cvxpy_problem(self):
        """Build the CVXPY optimization problem. Must be implemented by subclasses."""
        raise NotImplementedError

    def _setup_cuopt_problem(self):
        """Build the cuOpt optimization problem. Must be implemented by subclasses."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # CVXPY parameter assignment
    # ------------------------------------------------------------------

    def _assign_cvxpy_parameter_values(self):
        """
        Assign values to all CVXPY parameters from current data and
        parameter settings.

        Handles the common parameters (weight bounds, cash bounds,
        risk_aversion, L_tar, T_tar, cardinality) then delegates to
        _assign_subclass_cvxpy_params for optimizer-specific parameters.
        """
        if self.api_settings.weight_constraints_type == "parameter":
            self.w_min_param.value = self.params.w_min
            self.w_max_param.value = self.params.w_max

        if self.api_settings.cash_constraints_type == "parameter":
            self.c_min_param.value = self.params.c_min
            self.c_max_param.value = self.params.c_max

        self.risk_aversion_param.value = self.params.risk_aversion
        self.L_tar_param.value = self.params.L_tar

        if self.params.T_tar is not None:
            self.T_tar_param.value = self.params.T_tar

        if self.params.cardinality is not None:
            self.cardinality_param.value = self.params.cardinality

        self._assign_subclass_cvxpy_params()

    def _assign_subclass_cvxpy_params(self):
        """Hook for subclass-specific CVXPY parameter assignment."""
        pass

    # ------------------------------------------------------------------
    # Pickle save
    # ------------------------------------------------------------------

    def _save_problem_pickle(self, pickle_save_path: str):
        """Save the CVXPY optimization problem to a pickle file."""
        try:
            os.makedirs(os.path.dirname(pickle_save_path), exist_ok=True)
            with open(pickle_save_path, "wb") as f:
                pickle.dump(self.optimization_problem, f)
            print(f"Problem saved to: {pickle_save_path}")
        except Exception as e:
            print(f"Warning: Failed to save problem to pickle: {e}")

    # ------------------------------------------------------------------
    # Solve dispatch
    # ------------------------------------------------------------------

    def _get_cvxpy_risk_metric_value(self):
        """Return the risk metric value from the CVXPY solution.

        Must be implemented by subclasses (e.g. CVaR risk, variance).
        """
        raise NotImplementedError

    def _solve_cvxpy_problem(self, solver_settings: dict):
        """
        Solve the CVXPY optimization problem.

        Parameters
        ----------
        solver_settings : dict
            Solver configuration dict for CVXPY.Problem.solve().

        Returns
        -------
        result_row : pd.Series
        weights : np.ndarray
        cash : float
        """
        self.optimization_problem.solve(**solver_settings)
        weights = self.w.value
        cash = self.c.value

        solver_stats = getattr(self.optimization_problem, "solver_stats", None)
        reported_solve_time = (
            getattr(solver_stats, "solve_time", None)
            if solver_stats is not None
            else None
        )

        solver_time = (
            float(reported_solve_time)
            if reported_solve_time is not None
            else self.optimization_problem._solve_time
        )

        self.cvxpy_api_overhead = (
            self.optimization_problem._solve_time - solver_time
            if reported_solve_time is not None
            else None
        )

        result_row = pd.Series(
            [
                self.regime_name,
                str(solver_settings["solver"]),
                solver_time,
                self.expected_ptf_returns.value,
                self._get_cvxpy_risk_metric_value(),
                self.optimization_problem.value,
            ],
            index=self._result_columns,
        )

        return result_row, weights, cash

    def _solve_cuopt_problem(self, solver_settings: dict = None):
        """Solve the cuOpt optimization problem. Must be implemented by subclasses."""
        raise NotImplementedError

    def _print_results(
        self,
        result_row: pd.Series,
        portfolio: Portfolio,
        time_results: dict,
        min_percentage: float = 1,
    ):
        """Print optimization results. Must be implemented by subclasses."""
        raise NotImplementedError

    def solve_optimization_problem(
        self, solver_settings: dict = None, print_results: bool = True
    ):
        """
        Unified solve method that calls the appropriate API-specific solver.

        Parameters
        ----------
        solver_settings : dict, optional
            Solver configuration. Format depends on API choice:
            - CVXPY: {"solver": cp.CLARABEL, "verbose": True}
            - cuOpt: {"time_limit": 60}
        print_results : bool, default True
            Enable formatted result output to console.

        Returns
        -------
        result_row : pd.Series
            Performance metrics.
        portfolio : Portfolio
            Optimized portfolio with weights and cash allocation.
        """
        time_results = {}

        if self.api_choice == "cvxpy":
            if solver_settings is None or solver_settings.get("solver") is None:
                raise ValueError("A solver must be provided for CVXPY API")
            result_row, weights, cash = self._solve_cvxpy_problem(solver_settings)
            portfolio_name = str(solver_settings["solver"]) + "_optimal"
        elif self.api_choice == "cuopt_python":
            result_row, weights, cash = self._solve_cuopt_problem(solver_settings)
            portfolio_name = "cuOpt_optimal"
        else:
            raise ValueError(f"Unsupported api_choice: {self.api_choice}")

        portfolio = Portfolio(
            name=portfolio_name,
            tickers=self.tickers,
            weights=weights,
            cash=cash,
            time_range=self.regime_range,
        )

        if print_results:
            self._print_results(result_row, portfolio, time_results, min_percentage=1)

        return result_row, portfolio

    # ------------------------------------------------------------------
    # Cone data extraction
    # ------------------------------------------------------------------

    def _get_cone_data_filename(self):
        """Return the filename for cone data pickle. Must be implemented by subclasses."""
        raise NotImplementedError

    def _extract_problem_cone_data(self, problem_data_dir: str):
        """Extract cone data from the CVXPY problem and save to pickle.

        Parameters
        ----------
        problem_data_dir : str
            Directory path where the pickle file will be saved.

        Returns
        -------
        P, q, A, b, dims
            Cone problem data components.
        """
        data = self.optimization_problem.get_problem_data("SCS")
        P = data[0].get("P", None)
        q = data[0].get("c", None)
        A = data[0].get("A", None)
        b = data[0].get("b", None)
        dims = data[0].get("dims", None)

        os.makedirs(problem_data_dir, exist_ok=True)

        filename = self._get_cone_data_filename()
        pickle_file_path = os.path.join(problem_data_dir, filename)

        with open(pickle_file_path, "wb") as f:
            pickle.dump(data, f)

        print(f"Problem data saved to: {pickle_file_path}")

        return P, q, A, b, dims

    # ------------------------------------------------------------------
    # cuOpt timing utility
    # ------------------------------------------------------------------

    def _print_cuopt_timing(self, timing_dict):
        """Print detailed timing information for cuOpt problem setup loops.

        Parameters
        ----------
        timing_dict : dict
            Dictionary containing timing information for each setup phase.
        """
        print("\ncuOpt SETUP TIMING BREAKDOWN")
        print(f"{'-' * 40}")
        total_time = sum(timing_dict.values())
        for phase, time_taken in timing_dict.items():
            percentage = (time_taken / total_time * 100) if total_time > 0 else 0
            print(
                f"{phase.replace('_', ' ').title():<25}: {time_taken:.6f}s "
                f"({percentage:.1f}%)"
            )
        print(f"{'-' * 40}")
        print(f"{'Total Setup Time':<25}: {total_time:.6f}s (100.0%)")
        print()
