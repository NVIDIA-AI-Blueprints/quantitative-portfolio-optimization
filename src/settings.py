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

"""Configuration settings for portfolio optimization workflows.

This module provides Pydantic-based settings classes for configuring:
- Returns computation (return type, frequency, device)
- Scenario generation (number of scenarios, fitting method, KDE settings)
- API selection (solver API, constraint types)

These settings classes provide validation, default values, and type safety
for the various configuration dictionaries used throughout the optimization
workflow.
"""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class KDESettings(BaseModel):
    """Configuration for Kernel Density Estimation (KDE) sampling.

    Used within ScenarioGenerationSettings when fit_type is 'kde'.

    Attributes
    ----------
    bandwidth : float
        Bandwidth parameter for the kernel. Controls smoothness of the
        estimated density. Smaller values give more detailed estimates.
        Default is 0.01.
    kernel : str
        Kernel function to use. Supported: 'gaussian', 'tophat', 'epanechnikov',
        'exponential', 'linear', 'cosine'. Default is 'gaussian'.
    device : str
        Computation device. 'GPU' uses cuML for acceleration, 'CPU' uses
        scikit-learn. Default is 'GPU'.

    Examples
    --------
    >>> kde_settings = KDESettings(bandwidth=0.05, kernel='gaussian', device='GPU')
    >>> print(kde_settings.bandwidth)
    0.05
    """

    model_config = ConfigDict(validate_assignment=True)

    bandwidth: float = Field(default=0.01, gt=0, description="KDE bandwidth parameter")
    kernel: Literal[
        "gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"
    ] = Field(default="gaussian", description="Kernel function type")
    device: Literal["GPU", "CPU"] = Field(
        default="GPU", description="Computation device"
    )

    @field_validator("device", mode="before")
    @classmethod
    def normalize_device(cls, v: str) -> str:
        """Normalize device string to uppercase."""
        if isinstance(v, str):
            return v.upper()
        return v


class ScenarioGenerationSettings(BaseModel):
    """Configuration for scenario generation in CVaR optimization.

    Controls how return scenarios are generated from historical data,
    including the number of scenarios, fitting method, and KDE parameters.

    Attributes
    ----------
    num_scen : int
        Number of return scenarios to simulate. Default is 10000.
    fit_type : str
        Distribution fitting method:
        - 'gaussian': Fit multivariate normal distribution
        - 'kde': Kernel density estimation for non-parametric fitting
        - 'no_fit': Use historical returns directly as scenarios
        Default is 'kde'.
    kde_settings : KDESettings, optional
        Settings for KDE fitting. Only used when fit_type='kde'.
        If None and fit_type='kde', default KDESettings are used.
    verbose : bool
        Whether to print progress information. Default is False.

    Examples
    --------
    >>> settings = ScenarioGenerationSettings(
    ...     num_scen=10000,
    ...     fit_type='kde',
    ...     kde_settings=KDESettings(bandwidth=0.01, kernel='gaussian', device='GPU'),
    ...     verbose=False
    ... )
    >>> print(settings.num_scen)
    10000

    >>> # Using model_dump() to get dictionary representation
    >>> settings_dict = settings.model_dump()
    """

    model_config = ConfigDict(validate_assignment=True)

    num_scen: int = Field(
        default=10000, gt=0, description="Number of scenarios to simulate"
    )
    fit_type: Literal["gaussian", "kde", "no_fit"] = Field(
        default="kde", description="Distribution fitting method"
    )
    kde_settings: Optional[KDESettings] = Field(
        default=None, description="KDE-specific settings"
    )
    verbose: bool = Field(default=False, description="Print progress information")

    @model_validator(mode="after")
    def set_default_kde_settings(self) -> "ScenarioGenerationSettings":
        """Set default KDE settings when fit_type is 'kde' and kde_settings is None."""
        if self.fit_type == "kde" and self.kde_settings is None:
            self.kde_settings = KDESettings()
        return self


class ReturnsComputeSettings(BaseModel):
    """Configuration for computing asset returns from price data.

    Specifies how returns are calculated from raw price data, including
    the return type, frequency, and computation device.

    Attributes
    ----------
    return_type : str
        Type of returns to compute:
        - 'LOG': Logarithmic returns (log(P_t / P_{t-freq}))
        - 'LINEAR': Simple percentage returns ((P_t - P_{t-freq}) / P_{t-freq})
        - 'ABSOLUTE': Absolute price changes (P_t - P_{t-freq})
        - 'PNL': Input data is already P&L, no transformation needed
        Default is 'LOG'.
    freq : int
        Frequency for return calculation. 1 means daily returns,
        5 means weekly returns (for daily data). Default is 1.
    returns_compute_device : str
        Computation device. 'GPU' for GPU acceleration, 'CPU' for CPU.
        Default is 'CPU'.
    verbose : bool
        Whether to print progress information. Default is False.

    Examples
    --------
    >>> settings = ReturnsComputeSettings(return_type='LOG', freq=1)
    >>> print(settings.return_type)
    LOG

    >>> # For weekly returns from daily data
    >>> weekly_settings = ReturnsComputeSettings(return_type='LINEAR', freq=5)
    """

    model_config = ConfigDict(validate_assignment=True)

    return_type: Literal["LOG", "LINEAR", "ABSOLUTE", "PNL"] = Field(
        default="LOG", description="Return calculation method"
    )
    freq: int = Field(default=1, gt=0, description="Return calculation frequency")
    returns_compute_device: Literal["GPU", "CPU"] = Field(
        default="CPU", description="Computation device"
    )
    verbose: bool = Field(default=False, description="Print progress information")

    @field_validator("return_type", mode="before")
    @classmethod
    def normalize_return_type(cls, v: str) -> str:
        """Normalize return_type string to uppercase."""
        if isinstance(v, str):
            return v.upper()
        return v

    @field_validator("returns_compute_device", mode="before")
    @classmethod
    def normalize_device(cls, v: str) -> str:
        """Normalize device string to uppercase."""
        if isinstance(v, str):
            return v.upper()
        return v


class ApiSettings(BaseModel):
    """Configuration for optimization API selection.

    Specifies which optimization backend to use and how constraints
    should be formulated.

    Attributes
    ----------
    api : str
        Optimization API to use:
        - 'cvxpy': Use CVXPY for convex optimization (supports multiple solvers)
        - 'cuopt_python': Use NVIDIA cuOpt for GPU-accelerated optimization
        Default is 'cvxpy'.
    weight_constraints_type : str
        How weight constraints are specified in CVXPY:
        - 'parameter': Constraints use CVXPY Parameters (can be updated)
        - 'bounds': Constraints are fixed bounds
        Only applies when api='cvxpy'. Default is 'bounds'.
    cash_constraints_type : str
        How cash constraints are specified in CVXPY:
        - 'parameter': Constraints use CVXPY Parameters (can be updated)
        - 'bounds': Constraints are fixed bounds
        Only applies when api='cvxpy'. Default is 'bounds'.
    scale_risk_aversion : bool
        Whether to heuristically scale the risk aversion parameter by the ratio
        of maximum return/risk for single-asset portfolios. Default is True.
    pickle_save_path : str, optional
        Path to save the CVXPY problem as a pickle file. Only applies when
        api='cvxpy'. If None, problem is not saved. Default is None.

    Examples
    --------
    >>> settings = ApiSettings(
    ...     api='cvxpy',
    ...     weight_constraints_type='parameter',
    ...     cash_constraints_type='parameter'
    ... )
    >>> print(settings.api)
    cvxpy

    >>> # For cuOpt usage
    >>> cuopt_settings = ApiSettings(api='cuopt_python')

    >>> # CVXPY with pickle save enabled
    >>> settings = ApiSettings(
    ...     api='cvxpy',
    ...     pickle_save_path='problems/cvar_problem.pkl'
    ... )
    """

    model_config = ConfigDict(validate_assignment=True)

    api: Literal["cvxpy", "cuopt_python"] = Field(
        default="cvxpy", description="Optimization API to use"
    )
    weight_constraints_type: Literal["parameter", "bounds"] = Field(
        default="bounds",
        description="Weight constraint formulation (CVXPY only)",
    )
    cash_constraints_type: Literal["parameter", "bounds"] = Field(
        default="bounds",
        description="Cash constraint formulation (CVXPY only)",
    )
    scale_risk_aversion: bool = Field(
        default=True,
        description="Heuristically scale risk aversion by max return/risk ratio",
    )
    pickle_save_path: Optional[str] = Field(
        default=None,
        description="Path to save CVXPY problem pickle (CVXPY only)",
    )

