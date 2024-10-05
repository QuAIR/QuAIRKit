# !/usr/bin/env python3
# Copyright (c) 2023 QuAIR team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""
The source file of the class for the distance.
"""

import torch

from ..core import Operator, State, utils

__all__ = ["TraceDistance", "StateFidelity"]

class TraceDistance(Operator):
    r"""The class of the loss function to compute the trace distance.

    This interface can make you using the trace distance as the loss function.

    Args:
        target_state: The target state to be used to compute the trace distance.

    """
    def __init__(self, target_state: State):
        super().__init__()
        self.target_state = target_state

    def forward(self, state: State) -> torch.Tensor:
        r"""Compute the trace distance between the input state and the target state.

        The value computed by this function can be used as a loss function to optimize.

        Args:
            state: The input state which will be used to compute the trace distance with the target state.

        Raises:
            NotImplementedError: The backend is wrong or not supported.

        Returns:
            The trace distance between the input state and the target state.
        """
        assert (
            state.num_systems == self.target_state.num_systems
        ), f"The number of systems does not match: received {state.num_systems}, expect {self.target_state.num_systems}."

        if not (state.backend == self.target_state.backend == 'state_vector'):
            return utils.qinfo._trace_distance(
                state.density_matrix, self.target_state.density_matrix
            )

        inner_prod = torch.linalg.vecdot(state.ket.squeeze(-1), self.target_state.ket.squeeze(-1))
        fidelity = torch.abs(inner_prod) ** 2
        return torch.sqrt(1 - fidelity)

class StateFidelity(Operator):
    r"""The class of the loss function to compute the state fidelity.

    This interface can make you using the state fidelity as the loss function.

    Args:
        target_state: The target state to be used to compute the state fidelity.
    """
    def __init__(self, target_state: State):
        super().__init__()
        self.target_state = target_state

    def forward(self, state: State) -> torch.Tensor:
        r"""Compute the state fidelity between the input state and the target state.

        The value computed by this function can be used as a loss function to optimize.

        Args:
            state: The input state which will be used to compute the state fidelity with the target state.

        Raises:
            NotImplementedError: The backend is wrong or not supported.

        Returns:
            The state fidelity between the input state and the target state.
        """
        assert (
            state.num_systems == self.target_state.num_systems
        ), f"The number of systems does not match: received {state.num_systems}, expect {self.target_state.num_systems}."
        
        if not (state.backend == self.target_state.backend == 'state_vector'):
            return utils.qinfo._state_fidelity(
                state.density_matrix, self.target_state.density_matrix
            )
        inner_prod = torch.linalg.vecdot(state.ket.squeeze(-1), self.target_state.ket.squeeze(-1))
        return torch.abs(inner_prod)
