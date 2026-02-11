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

import copy
import os
import pickle
import time
from typing import Optional

import cvxpy as cp
import numpy as np
import pandas as pd

from . import base_optimizer
from .mean_variance_parameters import MeanVarianceParameters
from .portfolio import Portfolio
from .settings import ApiSettings

"""
Module: Mean-Variance Optimization
==================================
This module implements data structures and a class for Mean-Variance
portfolio optimization (Markowitz optimization).

A Mean-Variance optimizer chooses asset weights that maximize expected return
while minimizing portfolio variance (or, equivalently, minimizing a
risk-penalized loss).

Key features
------------
* Set up problem using different interfaces (CVXPY with bounds/parameters, cuOpt).
* Build models with customizable constraints based on MeanVarianceParameters.
* Print optimization results with detailed performance metrics and allocation.

Public classes
--------------
``MeanVariance``
    Main Mean-Variance portfolio optimizer class that supports multiple solver
    interfaces (CVXPY and cuOpt). Handles Mean-Variance optimization with
    customizable constraints including weight bounds, cash allocation, leverage
    limits, variance hard limits, turnover restrictions, and cardinality constraints.

Usage Examples
--------------
Standard CVXPY solver (uses bounds by default):
    >>> optimizer = MeanVariance(returns_dict, mean_variance_params)
    >>> result, portfolio = optimizer.solve_optimization_problem(
    ...     {"solver": cp.CLARABEL}
    ... )

cuOpt GPU solver:
    >>> api_settings = ApiSettings(api="cuopt_python")
    >>> optimizer = MeanVariance(returns_dict, mean_variance_params, api_settings=api_settings)
    >>> result, portfolio = optimizer.solve_optimization_problem({
    ...     "time_limit": 60
    ... })

CVXPY with parameters:
    >>> api_settings = ApiSettings(
    ...     api="cvxpy",
    ...     weight_constraints_type="parameter",
    ...     cash_constraints_type="parameter"
    ... )
    >>> optimizer = MeanVariance(returns_dict, mean_variance_params, api_settings=api_settings)
    >>> result, portfolio = optimizer.solve_optimization_problem(
    ...     {"solver": cp.CLARABEL}
    ... )
"""


class MeanVariance(base_optimizer.BaseOptimizer):
    """
    Mean-Variance portfolio optimizer with multiple API support.

    Solves Mean-Variance (Markowitz) optimization problems with the following constraints:
        - Weight bounds
        - Cash bounds
        - Leverage constraint
        - Hard variance limit (optional)
        - Turnover constraint (optional)
        - Cardinality constraint (optional)

    Key features:
    - Risk-adjusted return optimization using variance as risk measure
    - Supports both CVXPY and cuOpt Python APIs
    - GPU acceleration available via cuOpt
    - Performance monitoring with timing metrics
    - Automatic setup based on API choice
    """

    def __init__(
        self,
        returns_dict: dict,
        mean_variance_params: MeanVarianceParameters,
        api_settings: Optional[ApiSettings] = None,
        existing_portfolio: Optional[Portfolio] = None,
    ):
        """Initialize Mean-Variance optimizer with data and constraints.

        Parameters
        ----------
        returns_dict : dict
            Input data containing regime info, mean returns, and covariance matrix.
        mean_variance_params : MeanVarianceParameters
            Constraint parameters and optimization settings (deep-copied).
        api_settings : ApiSettings, optional
            API configuration including solver choice and constraint types.
            Uses CVXPY with bounds if not provided.
        existing_portfolio : Portfolio, optional
            An existing portfolio to measure the turnover from.
        """
        super().__init__(returns_dict, existing_portfolio, "variance")

        if api_settings is None:
            api_settings = ApiSettings()

        self.api_settings = api_settings
        self.api_choice = api_settings.api

        self.regime_name = returns_dict["regime"]["name"]
        self.regime_range = returns_dict["regime"]["range"]
        self.mean = returns_dict["mean"]
        self.covariance = returns_dict["covariance"]
        self.existing_portfolio = existing_portfolio
        self.params = self._store_mean_variance_params(mean_variance_params)

        # Set up the optimization problem based on API choice
        self._setup_optimization_problem()

        self.optimal_portfolio = None

        self._result_columns = [
            "regime",
            "solver",
            "solve time",
            "return",
            "variance",
            "obj",
        ]

    def _store_mean_variance_params(self, mean_variance_params: MeanVarianceParameters):
        """
        Store the Mean-Variance parameters in the optimizer.

        If w_min and w_max are input as floats, convert them to ndarrays
        with the same value repeated for all assets.
        """
        params_copy = copy.deepcopy(mean_variance_params)

        params_copy.w_min = self._update_weight_constraints(params_copy.w_min)
        params_copy.w_max = self._update_weight_constraints(params_copy.w_max)

        return params_copy

    def _setup_optimization_problem(self):
        """
        Set up the optimization problem based on the selected API choice.
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
            if self.params.var_limit is not None:
                # CVXPY supports a quadratic constraint (variance <= var_limit),
                # but cuOpt's QP interface here supports linear constraints only.
                raise NotImplementedError(
                    "cuOpt Python API does not support 'var_limit' (quadratic constraint). "
                )
            (
                self._cuopt_problem,
                self._cuopt_variables,
                self.cuopt_timing_dict,
            ) = self._setup_cuopt_problem()
        else:
            raise ValueError(f"Unsupported api_choice: {self.api_choice}")

        set_up_end = time.time()
        self.set_up_time = set_up_end - set_up_start

    def _scale_risk_aversion(self):
        """
        Heuristically scale risk aversion parameter by the ratio of
        the maximum return over standard deviation for single-asset portfolios.
        """
        std_devs = np.sqrt(np.diag(self.covariance))
        std_devs = np.maximum(std_devs, 1e-10)
        return_risk_ratios = self.mean / std_devs

        self._risk_aversion_scalar = np.max(return_risk_ratios)

        self.params.update_risk_aversion(
            self.params.risk_aversion * self._risk_aversion_scalar
        )

    def _setup_cvxpy_problem(self):
        """
        Build the mean-variance optimization problem using CVXPY.

        Supports the following types of problems:
            1. (QP) 'basic mean-variance': basic Markowitz problem
            2. (QP) 'mean-variance with limit': hard limit on variance
            3. (QP) 'mean-variance with turnover'
            4. (MIQP) 'mean-variance with cardinality' (not implemented)
        """
        num_assets = self.n_assets

        # Create variables based on constraint type settings
        if self.api_settings.weight_constraints_type == "bounds":
            self.w = cp.Variable(
                num_assets,
                name="weights",
                bounds=[self.params.w_min, self.params.w_max],
            )
        else:
            self.w = cp.Variable(num_assets, name="weights")
            self.w_min_param = cp.Parameter(num_assets, name="w_min")
            self.w_max_param = cp.Parameter(num_assets, name="w_max")

        if self.api_settings.cash_constraints_type == "bounds":
            self.c = cp.Variable(
                1, name="cash", bounds=[self.params.c_min, self.params.c_max]
            )
        else:
            self.c = cp.Variable(1, name="cash")
            self.c_min_param = cp.Parameter(name="c_min")
            self.c_max_param = cp.Parameter(name="c_max")

        # Create parameters for optimization parameters
        self.risk_aversion_param = cp.Parameter(nonneg=True, name="risk_aversion")
        self.L_tar_param = cp.Parameter(nonneg=True, name="L_tar")
        self.T_tar_param = cp.Parameter(nonneg=True, name="T_tar")
        self.var_limit_param = cp.Parameter(nonneg=True, name="var_limit")
        self.cardinality_param = cp.Parameter(name="cardinality")

        # Set up expressions for optimization
        self.expected_ptf_returns = self.mean.T @ self.w
        self.portfolio_variance = cp.quad_form(self.w, self.covariance)

        # Add variable bounds constraints (only if using parameter constraints)
        constraints = []
        if self.api_settings.weight_constraints_type == "parameter":
            constraints.extend(
                [
                    self.w_min_param <= self.w,
                    self.w <= self.w_max_param,
                ]
            )
        if self.api_settings.cash_constraints_type == "parameter":
            constraints.extend(
                [
                    self.c_min_param <= self.c,
                    self.c <= self.c_max_param,
                ]
            )

        # Set up common constraints
        if self.params.cardinality is not None:
            raise NotImplementedError("MIQP (cardinality constraint) is not implemented yet.")
        else:
            constraints.extend(
                [
                    cp.sum(self.w) + self.c == 1,
                    cp.norm1(self.w) <= self.L_tar_param,
                ]
            )

        # Set up objective
        if self.params.var_limit is None:
            obj = cp.Minimize(
                self.risk_aversion_param * self.portfolio_variance
                - self.expected_ptf_returns
            )
        else:
            obj = cp.Maximize(self.expected_ptf_returns)
            constraints.append(self.portfolio_variance <= self.var_limit_param)

        # Set up turnover constraint
        if self.existing_portfolio is not None and self.params.T_tar is not None:
            w_prev = np.array(self.existing_portfolio.weights)
            z = self.w - w_prev
            constraints.append(cp.norm(z, 1) <= self.T_tar_param)

        # Set up group constraints
        if self.params.group_constraints is not None:
            for group_constraint in self.params.group_constraints:
                tickers_index = [
                    self.tickers.index(ticker)
                    for ticker in group_constraint["tickers"]
                ]
                constraints.append(
                    cp.sum(self.w[tickers_index])
                    <= group_constraint["weight_bounds"]["w_max"]
                )
                constraints.append(
                    cp.sum(self.w[tickers_index])
                    >= group_constraint["weight_bounds"]["w_min"]
                )

        self.optimization_problem = cp.Problem(obj, constraints)

    def _assign_cvxpy_parameter_values(self):
        """
        Assign values to all CVXPY parameters from current data and parameter settings.
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

        if self.params.var_limit is not None:
            self.var_limit_param.value = self.params.var_limit

        if self.params.cardinality is not None:
            self.cardinality_param.value = self.params.cardinality

    def _setup_cuopt_problem(self):
        """
        Set up Mean-Variance optimization problem using cuOpt Python API.

        Uses QuadraticExpression for the quadratic objective term w'Σw.

        Returns
        -------
        problem : cuopt Problem instance
        variables : dict
        timing_dict : dict
        """
        from cuopt.linear_programming.problem import (
            CONTINUOUS,
            MINIMIZE,
            Problem,
            LinearExpression,
            QuadraticExpression,
        )

        num_assets = self.n_assets
        timing = {}

        # Step 1: Create problem
        t0 = time.time()
        problem = Problem("Mean-Variance Portfolio QP")
        timing["create_problem"] = time.time() - t0

        variables = {}

        # Step 2a: Add weight variables w[i]
        t0 = time.time()
        w_vars = []
        for i in range(num_assets):
            w_var = problem.addVariable(
                lb=float(self.params.w_min[i]),
                ub=float(self.params.w_max[i]),
                vtype=CONTINUOUS,
                name=f"w_{i}",
            )
            w_vars.append(w_var)
        variables["w"] = w_vars
        timing["add_weight_vars"] = time.time() - t0

        # Step 2b: Add cash variable
        t0 = time.time()
        variables["c"] = problem.addVariable(
            lb=float(self.params.c_min),
            ub=float(self.params.c_max),
            vtype=CONTINUOUS,
            name="cash",
        )
        timing["add_cash_var"] = time.time() - t0

        # Step 2c: Auxiliary variables for leverage (w_pos, w_neg)
        t0 = time.time()
        w_pos_vars = []
        w_neg_vars = []
        for i in range(num_assets):
            w_pos = problem.addVariable(lb=0.0, vtype=CONTINUOUS, name=f"w_pos_{i}")
            w_neg = problem.addVariable(lb=0.0, vtype=CONTINUOUS, name=f"w_neg_{i}")
            w_pos_vars.append(w_pos)
            w_neg_vars.append(w_neg)
        variables["w_pos"] = w_pos_vars
        variables["w_neg"] = w_neg_vars
        timing["add_aux_vars"] = time.time() - t0

        # Step 3a: Budget constraint: sum(w) + c = 1
        t0 = time.time()
        budget_vars = w_vars + [variables["c"]]
        budget_coeffs = [1.0] * (num_assets + 1)
        budget_expr = LinearExpression(budget_vars, budget_coeffs, 0.0)
        problem.addConstraint(budget_expr == 1.0, name="budget")
        timing["budget_constraint"] = time.time() - t0

        # Step 3b: Decomposition constraints: w[i] - w_pos[i] + w_neg[i] = 0
        t0 = time.time()
        for i in range(num_assets):
            decomp_vars = [w_vars[i], w_pos_vars[i], w_neg_vars[i]]
            decomp_coeffs = [1.0, -1.0, 1.0]
            decomp_expr = LinearExpression(decomp_vars, decomp_coeffs, 0.0)
            problem.addConstraint(decomp_expr == 0.0, name=f"decomp_{i}")
        timing["decomp_constraints"] = time.time() - t0

        # Step 3c: Leverage constraint: sum(w_pos + w_neg) <= L_tar
        t0 = time.time()
        leverage_vars = w_pos_vars + w_neg_vars
        leverage_coeffs = [1.0] * (2 * num_assets)
        leverage_expr = LinearExpression(leverage_vars, leverage_coeffs, 0.0)
        problem.addConstraint(leverage_expr <= self.params.L_tar, name="leverage")
        timing["leverage_constraint"] = time.time() - t0

        # Step 3d: Turnover constraint (if applicable)
        if self.existing_portfolio is not None and self.params.T_tar is not None:
            t0 = time.time()
            w_prev = np.array(self.existing_portfolio.weights)
            turnover_pos_vars = []
            turnover_neg_vars = []

            for i in range(num_assets):
                to_pos = problem.addVariable(
                    lb=0.0, vtype=CONTINUOUS, name=f"turnover_pos_{i}"
                )
                to_neg = problem.addVariable(
                    lb=0.0, vtype=CONTINUOUS, name=f"turnover_neg_{i}"
                )
                turnover_pos_vars.append(to_pos)
                turnover_neg_vars.append(to_neg)

            # Decomposition constraints: w[i] - to_pos[i] + to_neg[i] = w_prev[i]
            for i in range(num_assets):
                decomp_vars = [w_vars[i], turnover_pos_vars[i], turnover_neg_vars[i]]
                decomp_coeffs = [1.0, -1.0, 1.0]
                decomp_expr = LinearExpression(decomp_vars, decomp_coeffs, 0.0)
                problem.addConstraint(
                    decomp_expr == float(w_prev[i]), name=f"turnover_decomp_{i}"
                )

            # Turnover constraint: sum(to_pos + to_neg) <= T_tar
            turnover_vars = turnover_pos_vars + turnover_neg_vars
            turnover_coeffs = [1.0] * (2 * num_assets)
            turnover_expr = LinearExpression(turnover_vars, turnover_coeffs, 0.0)
            problem.addConstraint(turnover_expr <= self.params.T_tar, name="turnover")

            variables["turnover_pos"] = turnover_pos_vars
            variables["turnover_neg"] = turnover_neg_vars
            timing["turnover_constraints"] = time.time() - t0
        else:
            timing["turnover_constraints"] = 0.0

        # Step 3e: Group constraints (if applicable)
        if self.params.group_constraints is not None:
            t0 = time.time()
            for group_idx, group_constraint in enumerate(self.params.group_constraints):
                tickers_index = [
                    self.tickers.index(ticker)
                    for ticker in group_constraint["tickers"]
                ]
                if len(tickers_index) > 0:
                    group_vars = [w_vars[i] for i in tickers_index]
                    group_coeffs = [1.0] * len(tickers_index)
                    group_expr = LinearExpression(group_vars, group_coeffs, 0.0)
                    problem.addConstraint(
                        group_expr <= group_constraint["weight_bounds"]["w_max"],
                        name=f"group_{group_idx}_upper",
                    )
                    problem.addConstraint(
                        group_expr >= group_constraint["weight_bounds"]["w_min"],
                        name=f"group_{group_idx}_lower",
                    )
            timing["group_constraints"] = time.time() - t0
        else:
            timing["group_constraints"] = 0.0

        # Step 4: Build objective using QuadraticExpression
        #   minimize: risk_aversion * (w' Σ w) - (μ' w)
        #
        # cuOpt format:
        #   QuadraticExpression(quad_vars1, quad_vars2, quad_coeffs, lin_vars, lin_coeffs, constant)

        # 4a: Build linear terms: -μ'w
        t0 = time.time()
        lin_vars = w_vars
        lin_coeffs = [-float(self.mean[i]) for i in range(num_assets)]
        timing["build_linear_arrays"] = time.time() - t0

        # 4b: Build quadratic terms for risk_aversion * (w' Σ w)
        t0 = time.time()
        # quad_vars1: each var repeated n times consecutively
        quad_vars1 = [w for w in w_vars for _ in range(num_assets)]
        # quad_vars2: full list repeated n times
        quad_vars2 = w_vars * num_assets
        timing["build_quad_vars"] = time.time() - t0

        # 4c: Build quad_coeffs from covariance matrix
        t0 = time.time()
        q_matrix = self.params.risk_aversion * self.covariance
        quad_coeffs = q_matrix.ravel().tolist()
        timing["build_quad_coeffs"] = time.time() - t0

        # 4d: Create QuadraticExpression in one shot
        t0 = time.time()
        objective_expr = QuadraticExpression(
            quad_vars1, quad_vars2, quad_coeffs,
            lin_vars, lin_coeffs, 0.0
        )
        timing["create_quad_expr"] = time.time() - t0

        # 4e: Set objective
        t0 = time.time()
        problem.setObjective(objective_expr, sense=MINIMIZE)
        timing["set_objective"] = time.time() - t0

        # Print setup summary
        print(f"{'=' * 50}")
        print("cuOpt MEAN-VARIANCE (QP) PROBLEM SETUP COMPLETED")
        print(f"{'=' * 50}")
        print(f"Variables: {num_assets} weights + 1 cash + {2 * num_assets} leverage aux")
        print(f"Quadratic terms: {len(quad_coeffs)}")
        print(f"Linear terms: {len(lin_coeffs)}")
        print("Problem Type: QP (Quadratic Programming)")
        print(f"{'=' * 50}")

        return problem, variables, timing

    def _solve_cuopt_problem(self, solver_settings: dict = None):
        """
        Solve Mean-Variance optimization using cuOpt.

        Parameters
        ----------
        solver_settings : dict, optional
            cuOpt solver configuration.

        Returns
        -------
        result_row : pd.Series
        weights : np.ndarray
        cash : float
        """
        from cuopt.linear_programming.solver_settings import SolverSettings

        settings = SolverSettings()
        if solver_settings:
            for param, value in solver_settings.items():
                if param != "solver":  # Skip CVXPY-specific params
                    settings.set_parameter(param, value)

        total_start = time.time()
        self._cuopt_problem.solve(settings)
        total_end = time.time()
        total_time = total_end - total_start
        solve_time = self._cuopt_problem.SolveTime
        self.cuopt_api_overhead = total_time - solve_time

        if self._cuopt_problem.Status.name != "Optimal":
            raise RuntimeError(
                f"cuOpt failed to find optimal solution. Status: "
                f"{self._cuopt_problem.Status.name}"
            )

        weights = np.array([var.getValue() for var in self._cuopt_variables["w"]])
        cash = self._cuopt_variables["c"].getValue()

        expected_return = np.dot(self.mean, weights)
        variance_value = weights @ self.covariance @ weights

        objective_value = self._cuopt_problem.ObjValue

        result_row = pd.Series(
            [
                self.regime_name,
                "cuOpt",
                solve_time,
                expected_return,
                variance_value,
                objective_value,
            ],
            index=self._result_columns,
        )

        print(f"cuOpt solution found in {solve_time:.4f} seconds")
        print(f"Status: {self._cuopt_problem.Status.name}")
        print(f"Objective value: {objective_value:.6f}")

        return result_row, weights, cash

    def _solve_cvxpy_problem(self, solver_settings: dict):
        """
        Solve the optimization problem using the user-specified solver.

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
                self.portfolio_variance.value,
                self.optimization_problem.value,
            ],
            index=self._result_columns,
        )

        return result_row, weights, cash

    def _save_problem_pickle(self, pickle_save_path: str):
        """Save the CVXPY optimization problem to a pickle file."""
        try:
            os.makedirs(os.path.dirname(pickle_save_path), exist_ok=True)
            with open(pickle_save_path, "wb") as f:
                pickle.dump(self.optimization_problem, f)
            print(f"Mean-Variance problem saved to: {pickle_save_path}")
        except Exception as e:
            print(f"Warning: Failed to save Mean-Variance problem to pickle: {e}")

    def _print_mean_variance_results(
        self,
        result_row: pd.Series,
        portfolio: Portfolio,
        time_results: dict,
        min_percentage: float = 1,
    ):
        """Display Mean-Variance optimization results and portfolio allocation."""
        solver_name = result_row["solver"]
        solve_time = result_row["solve time"]
        expected_return = result_row["return"]
        variance_value = result_row["variance"]
        objective_value = result_row["obj"]

        print(f"\n{'=' * 60}")
        print("MEAN-VARIANCE OPTIMIZATION RESULTS")
        print(f"{'=' * 60}")

        print("PROBLEM CONFIGURATION")
        print(f"{'-' * 30}")
        print(f"Solver:              {solver_name}")
        print(f"Regime:              {self.regime_name}")
        print(f"Time Period:         {self.regime_range[0]} to {self.regime_range[1]}")
        print(f"Assets:              {self.n_assets}")

        if self.params.cardinality is not None:
            print(f"Cardinality Limit:   {self.params.cardinality} assets")
        if self.params.var_limit is not None:
            print(f"Variance Hard Limit: {self.params.var_limit:.6f}")
        if self.existing_portfolio is not None and self.params.T_tar is not None:
            print(f"Turnover Constraint: {self.params.T_tar:.3f}")

        print("\nPERFORMANCE METRICS")
        print(f"{'-' * 30}")
        print(f"Expected Return:     {expected_return:.6f} ({expected_return * 100:.4f}%)")
        print(f"Variance:            {variance_value:.6f}")
        print(f"Std Deviation:       {np.sqrt(variance_value):.6f}")
        print(f"Objective Value:     {objective_value:.6f}")

        print("\nSOLVING PERFORMANCE")
        print(f"{'-' * 30}")
        if hasattr(self, "set_up_time"):
            print(f"Setup Time:          {self.set_up_time:.4f} seconds")
        if hasattr(self, "cvxpy_api_overhead") and self.cvxpy_api_overhead is not None:
            print(f"CVXPY API Overhead:  {self.cvxpy_api_overhead:.4f} seconds")
        if hasattr(self, "cuopt_api_overhead"):
            print(f"cuOpt API Overhead:  {self.cuopt_api_overhead:.4f} seconds")
        print(f"Solve Time:          {solve_time:.4f} seconds")

        for key, value in time_results.items():
            print(f"{key.title():20} {value:.4f} seconds")

        print("\nOPTIMAL PORTFOLIO ALLOCATION")
        print(f"{'-' * 30}")
        portfolio.print_clean(verbose=True, min_percentage=min_percentage)

        print(f"{'=' * 60}\n")

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
            Performance metrics: regime, solve_time, return, variance, objective.
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
            self._print_mean_variance_results(
                result_row, portfolio, time_results, min_percentage=1
            )

        return result_row, portfolio

    def _extract_problem_cone_data(self, problem_data_dir: str):
        """Extract the cone data from the problem and save to pickle file."""
        data = self.optimization_problem.get_problem_data("SCS")
        P = data[0].get("P", None)
        q = data[0].get("c", None)
        A = data[0].get("A", None)
        b = data[0].get("b", None)
        dims = data[0].get("dims", None)

        os.makedirs(problem_data_dir, exist_ok=True)

        regime_name = getattr(self, "regime_name", "unknown")
        filename = f"mean_variance_{regime_name}.pkl"
        pickle_file_path = os.path.join(problem_data_dir, filename)

        with open(pickle_file_path, "wb") as f:
            pickle.dump(data, f)

        print(f"Problem data saved to: {pickle_file_path}")

        return P, q, A, b, dims

