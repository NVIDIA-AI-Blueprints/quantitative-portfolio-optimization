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
from typing import Optional

from .base_parameters import BaseParameters


class MeanVarianceParameters(BaseParameters):
    """
    User-tunable parameters and constraint limits for Mean-Variance optimization.

    Extends BaseParameters with a Mean-Variance-specific optional hard limit
    on portfolio variance (``var_limit``).
    """

    # Mean-Variance-specific field
    var_limit: Optional[float] = None

    def validate_var_limit(self, value: Optional[float]) -> Optional[float]:
        if value is not None:
            if not isinstance(value, float) or value <= 0:
                raise ValueError("Variance limit must be positive.")
        return value
