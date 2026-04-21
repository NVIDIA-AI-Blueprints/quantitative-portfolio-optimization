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
from pydantic import field_validator

from .base_parameters import BaseParameters


class CvarParameters(BaseParameters):
    """
    User-tunable parameters and constraint limits for CVaR optimization.

    Extends BaseParameters with CVaR-specific fields: ``confidence`` level
    for the CVaR risk measure and an optional hard ``cvar_limit``.
    """

    # Override weight bound defaults for CVaR
    w_min: Union[np.ndarray, dict, float] = 1.0
    w_max: Union[np.ndarray, dict, float] = 0.0

    # CVaR-specific fields
    confidence: float = 0.95
    cvar_limit: Optional[float] = None

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        if not (0 < value <= 1):
            raise ValueError(
                "Confidence level must be in (0, 1], e.g. 0.95 for 95% CVaR."
            )
        return value

    def update_cvar_limit(self, new_cvar_limit: float):
        self.cvar_limit = new_cvar_limit

    def update_confidence(self, new_confidence: float):
        if not (0 < new_confidence <= 1):
            raise ValueError(
                "Confidence level must be in (0, 1], e.g. 0.95 for 95% CVaR."
            )
        self.confidence = new_confidence
