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
from typing import Optional, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator


class BaseParameters(BaseModel):
    """
    Base class for portfolio optimization parameters.

    Contains shared fields, validators, and update methods used by both
    CVaR and Mean-Variance optimization parameter classes.

    Weight bounds ``w_min`` / ``w_max`` can be:
    - numpy arrays (length n_assets) for per-asset bounds
    - dict mapping asset names to bounds
    - float for uniform bounds across all assets
    - None for no bounds

    Optional constraints (T_tar, cardinality) default to None when
    not specified.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Weight / cash bounds
    w_min: Union[np.ndarray, dict, float] = 0.0
    w_max: Union[np.ndarray, dict, float] = 1.0
    c_min: float = 0.0
    c_max: float = 1.0

    # Risk model parameters
    risk_aversion: float = 1.0

    # Soft / hard constraint targets
    L_tar: float = 1.6
    T_tar: Optional[float] = None

    # Cardinality constraint
    cardinality: Optional[int] = None

    # Group constraints:
    # [{'group_name': group_name,
    #   'tickers': tickers
    #   'weight_bounds': {'w_min': w_min, 'w_max': w_max}}]
    group_constraints: Optional[list[dict]] = None

    @field_validator("c_min")
    @classmethod
    def validate_c_min(cls, value: float) -> float:
        if value < 0:
            raise ValueError("Cash lower bound (c_min) must be non-negative.")
        return value

    @field_validator("c_max")
    @classmethod
    def validate_c_max(cls, value: float) -> float:
        if not (0 <= value <= 1):
            raise ValueError("Cash upper bound (c_max) must be in [0, 1].")
        return value

    @field_validator("risk_aversion")
    @classmethod
    def validate_risk_aversion(cls, value: float) -> float:
        if value < 0:
            raise ValueError("Risk aversion must be non-negative.")
        return value

    @field_validator("cardinality")
    @classmethod
    def validate_cardinality(cls, value: Optional[int]) -> Optional[int]:
        if value is not None:
            if not isinstance(value, int) or value <= 0:
                raise ValueError("Cardinality must be a positive integer.")
        return value

    def update_w_min(self, new_w_min: Union[np.ndarray, dict, float]) -> None:
        self.w_min = new_w_min

    def update_w_max(self, new_w_max: Union[np.ndarray, dict, float]) -> None:
        if isinstance(new_w_max, (int, float)) and new_w_max > 1:
            raise ValueError("Scalar upper bound for weights must be <= 1.")
        self.w_max = new_w_max

    def update_c_min(self, new_c_min: float) -> None:
        if new_c_min < 0:
            raise ValueError("Cash lower bound (c_min) must be non-negative.")
        self.c_min = new_c_min

    def update_c_max(self, new_c_max: float) -> None:
        if not (0 <= new_c_max <= 1):
            raise ValueError("Cash upper bound (c_max) must be in [0, 1].")
        self.c_max = new_c_max

    def update_risk_aversion(self, new_risk_aversion: float) -> None:
        if new_risk_aversion < 0:
            raise ValueError("Risk aversion must be non-negative.")
        self.risk_aversion = new_risk_aversion

    def update_L_tar(self, new_L_tar: float) -> None:
        self.L_tar = new_L_tar

    def update_T_tar(self, new_T_tar: Optional[float]) -> None:
        self.T_tar = new_T_tar

    def update_cardinality(self, new_cardinality: Optional[int]) -> None:
        if new_cardinality is not None:
            if not isinstance(new_cardinality, int) or new_cardinality <= 0:
                raise ValueError("Cardinality must be a positive integer.")
        self.cardinality = new_cardinality

    def update_group_constraints(
        self, new_group_constraints: Optional[list[dict]]
    ) -> None:
        self.group_constraints = new_group_constraints
